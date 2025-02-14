import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch_scatter import scatter_mean

"""

'In the first ego-motion pretraining stage, we predicted flows and confidence maps (using the losses
in Eq. 7) with synthetic data of static scenes, i.e., without any dynamic video data. This stage helps
model effectively learn pairwise flows and corresponding confidence induced only by ego-motion. In the
second dynamic fine-tuning stage, we freeze the parameters of F and finetune F_m on synthetic dynamic
videos, conditioning F_m on the features from our pretrained F during each iteration to predict movement
probability map m_i, supervising through both camera and cross-entropy losse. This stage decorrelates
llearning scene dynamics from learning 2D correspondences, and thus leads to more stable and effective
training behavior for the differentiable BA framework.' 

'Within the motion module, we first perform 2D spatial average pooling to provide the model with global
spatial information; we tthen perform average pooling alon the time axis to fuse information from I_ii
and all its neighboring keyframes I_j (where j \in N(i))'

"""

class MotionNet(nn.Module):
    """Motion module F_m from MegaSAM to predict an object movement probability map"""
    
    def __init__(self):
        super(MotionNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, pading=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1)
        )
        # Post average temporal pool
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Sequential( 
            nn.Conv2d(128+128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, padding=1)
        )

    def forward(self,
                ii, # images
                nii, # neighboring images for ii 
                net: Tensor, # hidden state
                weight: Tensor # confidence
    ) -> Tensor:
        batch, num, ch, ht, wd = net.shape
        
        # TODO: might need to change this later if shapes don't match
        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        weight = weight.view(batch*num, -1, ht, wd)
        
        # Spatial Pool
        spa = F.adaptive_avg_pool2d(net, (1, 1))
        
        # Concatenate inputs
        movmap = torch.cat((weight, net, spa))
        
        # Convolution + Temporal Sequence for Confidence
        movmap = self.conv1(movmap)
        # temporal view 
        movmap = movmap.view(batch, num, -1, ht, wd)
        # temporal mean on image and neighboring keyframes
        # TODO: this scatter_mean probably does not do what I want it to
        movmap = scatter_mean(net, ii + nii[ii], dim=1) # -> (batch, num, 128, ht, wd)
        movmap = movmap.view(-1, 128, ht, wd)
        
        # Convolution
        movmap = self.conv2(movmap)
        
        # Concatenate net 
        movmap = self.conv3(movmap)
        
        return movmap.view(output_dim)
        
        