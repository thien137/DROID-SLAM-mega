# TODO Implement Consistent Video Depth

import numpy as np
import torch
from torch import Tensor

def cvd(Gs, fs, flows, depth, uncertainty):
    """

    'We fix estimated camera parameters and perform first-order global
    optimization over video depth and corresponding uncertainty parameters
    by minimizing flow and depth losses through pairwise 2D optical flows'

    Args:
        Gs (_type_): _description_
        fs (_type_): _description_
        flows (_type_): _description_
        depth (_type_): _description_
        uncertainty (_type_): _description_

    Returns:
        _type_: _description_
    """
    # TODO: Implemen this
    
    return depth, uncertainty