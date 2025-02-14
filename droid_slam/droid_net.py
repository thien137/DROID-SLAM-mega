import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indices

from torch_scatter import scatter_mean

from typing import *

def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    """ upsame disparity map """
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)

class GraphAgg(nn.Module):
    """'We then pool the hidden state over all features which share the same source
    view ii and predict a pixel-wise damping factor (eta). We use the softplus opesrator to ensure
    that the damping factor is positive. Additionally, we use the pooled features to predict a 8x8 
    mask (upmask) which can be used to upsample the inverse depth estimate.'

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        # features with same source view ix?
        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        # batch, 7-frame video sequence, 128 channels, height, width?
        net = net.view(batch, num, 128, ht, wd)
        # scatter mean groups features by source view index ix and computes
        # the mean across frames that share the same index -> average feature
        # representation for frames that belong to the same source
        net = scatter_mean(net, ix, dim=1)
        # view on (batch, num, 128, ht, wd)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        # damping factor
        eta = self.eta(net).view(batch, -1, ht, wd)
        # 8x8 upmask
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class UpdateModule(nn.Module):
    """Learned update operator

    Args:
        nn (_type_): _description_
    """
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DroidNet(nn.Module):
    
    fnet: BasicEncoder 
    cnet: BasicEncoder
    update: UpdateModule
    
    def __init__(self):
        super(DroidNet, self).__init__()
        # feature encoder 
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        # context encoder
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        # update operator
        self.update = UpdateModule()

    def extract_features(self, images):
        """run feature extraction networks

        Args:
            images (_type_): videos

        Returns:
            fmaps (_type): feature map
            net (_type_): context features
            inp (_type_): context features (relu activation)
        """

        # normalize images 
        # TODO: magic numbers
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Feature network results
        fmaps = self.fnet(images)
        
        # Context network results
        net = self.cnet(images)
        
        # Split context channel results 
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        return fmaps, net, inp

    def forward(self, 
                Gs: Tensor, 
                images: Tensor, 
                disps: Tensor, 
                intrinsics: Tensor, # TODO: Remove this, focal length is unknown
                graph: Optional[OrderedDict]=None, 
                num_steps: int=12, 
                fixedp: int=2):
        """Estimates SE3 or Sim3 between pair of frames

        Args:
            Gs (Tensor): _description_
            images (Tensor): _description_
            disps (Tensor): _description_
            intrinsics (Tensor): _description_
            graph (Optional[OrderedDict], optional): _description_. Defaults to None.
            num_steps (int, optional): _description_. Defaults to 12.
            fixedp (int, optional): _description_. Defaults to 2.

        Returns:
            _type_: _description_
        """
        
        # indices of images in graph
        u = keyframe_indices(graph)
        # ii[x] -> jj[x], ie. (ii[x], jj[x]) \in Edges
        ii, jj, kk = graph_to_edge_list(graph)

        # List of initial images indices
        ii = ii.to(device=images.device, dtype=torch.long)
        # List of overlapping "edges" images
        jj = jj.to(device=images.device, dtype=torch.long)

        # 1/8 feature maps, Split context channels (net, inp)
        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        # Correlation creation
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        # height, width
        ht, wd = images.shape[-2:]
        # p_i downsampled
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        # p_ij correspondences (initialize previous solution)
        # TODO: focal length is unknown, probably don't need to change this
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        # p_ij targets
        target = coords1.clone()

        # list of results for forward
        # TODO: should add intrinsics somewhere here
        Gs_list, disp_list, residual_list = [], [], []
        for step in range(num_steps):
            # TODO: should add intrinsics somewhere here
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # extract motion features
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0

            # concatenate the previous BA solution with the flow fieldto use feedback from 
            # the previous iteration (starts at 0 since target = coords1)
            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            # net = ConvGRU hidden state
            # delta = r
            # weight = w
            net, delta, weight, eta, upmask = \
                self.update(net, inp, corr, motion, ii, jj)

            # p_ij* = p_ij + delta
            target = coords1 + delta

            # update camera poses and disparity maps
            for i in range(2):
                Gs, disps = BA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)
            
            # previous BA solution
            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)

        # TODO: Should output intrinsics
        return Gs_list, disp_list, residual_list
