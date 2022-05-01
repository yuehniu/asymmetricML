"""DNN operators in low-rank format
Desc:
    This file contains several operators in low-rank format
    - lowrankRelU: ReLU function in low-rank format
    - lowrankPooling: Pooling function in low-rank format

Author:
Note:
    We do not need to create low-rank format for other linear operators
    such as Conv, Linear, etc. Since data decomposition is only performed
    in non-linear layers.
    Please refer to: https://arxiv.org/abs/2110.01229 for more explanation.
"""
import torch
import torch.nn as nn
from .utils import svd_approx


def _get_lowrank( data ):
    """A utility function to extract low-rank part from input
    :param data input data to be processed (batch, channel, height, width)
    :return low-rank output
    """
    bs, c, h, w = data.shape
    data_flat = torch.reshape(data, [bs, c, h * w])

    # =========================================================================
    # To extract low-rank part, we
    # 1) perform SVD on inputs;
    # 2) use r principle channels (Vh) to reconstruct data
    # 3) align dimension
    # for b in range(bs):
    #     U, s, Vh = torch.svd( data_flat[b, :, :] )
    #     S = torch.diag( s )
    #     data_lowrank = U[ :, 0:r ] @ S[ 0:r, 0:r ] @ torch.transpose( Vh[:, 0:r], 0, 1 )
    #     data_flat[ b, :, : ].copy_( data_lowrank )
    data_lowrank, rank = svd_approx( data_flat )

    return torch.reshape( data_lowrank, [bs, c, h, w] ), rank


class lowrankReLUop( torch.autograd.Function ):
    @staticmethod
    def forward( ctx, input ):
        """
        First apply normal ReLU op, then extract low-rank part
        :param ctx running contexts
        :param input input to a ReLU layer (batch, channel, height, width)
        :return ReLU output with only low-rank part
        """
        ctx.save_for_backward( input )
        output = torch.nn.functional.relu( input )
        output, rank = _get_lowrank( output )

        return output, torch.tensor( rank, requires_grad=False )

    @staticmethod
    def backward( ctx, gradout, gradrank ):
        """
        Backward is same as normal ReLU (though only low-rank data is used)
        :param ctx running contexts
        :param gradout gradients on outputs (batch, channel, height, width)
        :return gradients on inputs
        """
        input, = ctx.saved_tensors
        gradin = gradout.clone()
        gradin[ input < 0 ] = 0.0
        return gradin, None
