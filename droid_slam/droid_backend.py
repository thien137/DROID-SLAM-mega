import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph

"""

'The backend performs global bundle adjustment over the entire history of keyframes.
During each iteration, we rebuild the frame graph using the flow between all pairs
of keyframes, represented as NxN sdistance matrix. We first add edges between
temporally adjacent keyframes. We then sample new edges from the distance matrix
in order of increasing flow. With each selected edge, we suppress neighboring edges
within a distance of 2, where distance is defined as the Chebyshev distance between index 
pairs.

We then apply the update operator to the entire frame graph, often consisting of thousand
of frames and edages. Storing the full set of correlation volumes would quickly exceed video
memory. Instead, we use the memory efficient implementation proposed in RAFT.

During training, we implement dense bundle adjustment in PyTorch to leverage the automatic
diff

"""

class DroidBackend:
    """Performs global bundle adjustment over the entire history of keyframes"""
    
    net: "DroidNet"
    video: "DepthVideo"
    
    def __init__(self, net, video, args):
        # Depth Video
        self.video = video
        # Update operation
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.upsample = args.upsample
        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        
        # TODO
        self.backend_radius = args.backend_radius
        
        # Non-maximum Suppression
        self.backend_nms = args.backend_nms
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if not self.video.stereo and not torch.any(self.video.disps_sens):
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=16*t, upsample=self.upsample)

        graph.add_proximity_factors(rad=self.backend_radius, 
                                    nms=self.backend_nms, 
                                    thresh=self.backend_thresh, 
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True
