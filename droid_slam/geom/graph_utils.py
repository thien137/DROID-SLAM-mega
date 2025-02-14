
import torch
from torch import Tensor
import numpy as np
from collections import OrderedDict

import lietorch
from data_readers.rgbd_utils import compute_distance_matrix_flow, compute_distance_matrix_flow2

from typing import *

def graph_to_edge_list(graph: OrderedDict[int, List[int]]) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert graph to edge list

    Args:
        graph (OrderedDict[int, List[int]]): vertices and their edges

    Returns:
        ii (Tensor): start images indices
        jj (Tensor): end images indices
        kk (Tensor): 
    """
    ii, jj, kk = [], [], []
    for s, u in enumerate(graph):
        for v in graph[u]:
            ii.append(u)
            jj.append(v)
            kk.append(s)

    ii = torch.as_tensor(ii)
    jj = torch.as_tensor(jj)
    kk = torch.as_tensor(kk)
    return ii, jj, kk

def keyframe_indices(graph: OrderedDict[int, List[int, int]]) -> Tensor:
    """Return tensor of image indices contained in graph

    Args:
        graph (OrderedDict[int, List[int, int]]): vertices and their edges

    Returns:
        Tensor: indices of images contained in graph
    """
    return torch.as_tensor([u for u in graph])

def meshgrid(m: int, 
             n: int, 
             device: str='cuda') -> Tuple[Tensor, Tensor]:
    """Constructs flattened meshgrid

    Args:
        m (int): height of grid
        n (int): width of grid
        device (str): Defaults to 'cuda'

    Returns:
        ii (Tensor): meshgrid indices along height
        jj (Tensor): meshgrid indices along width
    """
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(n))
    return ii.reshape(-1).to(device), jj.reshape(-1).to(device)

def neighbourhood_graph(n: int, 
                        r: int):
    """Generates meshgrid indices or edges connecting neighbors

    Args:
        n (int): number of images
        r (int): radius

    Returns:
        ii (Tensor): Starting image indices
        jj (Tensor): Ending image indices
    """
    
    ii, jj = meshgrid(n, n)
    d = (ii - jj).abs()
    keep = (d >= 1) & (d <= r)
    return ii[keep], jj[keep]

def build_frame_graph(poses: Tensor, 
                      disps: Tensor, 
                      intrinsics: Tensor, 
                      num: int=16, 
                      thresh: int=24.0, 
                      r: int=2):
    """Construct a frame graph between co-visible frames (V, E)

    Args:
        poses (Tensor): camera poses
        disps (Tensor): disparity maps
        intrinsics (Tensor): camera intrinsics
        num (int, optional): _description_. Defaults to 16.
        thresh (int, optional): _description_. Defaults to 24.0.
        r (int, optional): _description_. Defaults to 2.

    Returns:
        graph OrderedList[int, List[int, int]]: graph of vertices and their images
    """
    N = poses.shape[1]
    poses = poses[0].cpu().numpy()
    disps = disps[0][:,3::8,3::8].cpu().numpy()
    intrinsics = intrinsics[0].cpu().numpy() / 8.0
    d = compute_distance_matrix_flow(poses, disps, intrinsics)

    count = 0
    graph = OrderedDict()
    
    for i in range(N):
        graph[i] = []
        d[i,i] = np.inf
        for j in range(i-r, i+r+1):
            if 0 <= j < N and i != j:
                graph[i].append(j)
                d[i,j] = np.inf
                count += 1

    while count < num:
        ix = np.argmin(d)
        i, j = ix // N, ix % N

        if d[i,j] < thresh:
            graph[i].append(j)
            d[i,j] = np.inf
            count += 1
        else:
            break
    
    return graph
