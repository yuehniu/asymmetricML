"""Data Pre-Processing
Desc:
    Functions for pre-processing data, like low-rank decomposition, adding noise, etc.

Note:
"""
import torch
from .utils import svd_approx


def data_lowrank( inputs, r=1 ):
    """extract low-rank part of inputs
    :param inputs data  ( [batch, ichannel, height, width] )
    :param r ranks ( int )
    :return low-rank parts of inputs ( [batch, ichannel, height, width] )
    """
    bs, c, h, w = inputs.shape
    inputs_flat = torch.reshape( inputs, [bs, c, h * w] )

    # =========================================================================
    # To extract low-rank part, we
    # 1) perform SVD on inputs;
    # 2) use r principle channels (Vh) to reconstruct data
    # 3) align dimension
    # U, s, Vh = torch.svd( inputs_flat[:, :, :] ) (maybe too computation intensive)
    # for b in range( bs ):
    #     U, s, Vh = torch.svd( inputs_flat[b, :, :] )
    #     S = torch.diag( s )
    #     input_lowrank = U[ :, 0:r ] @ S[ 0:r, 0:r ] @ torch.transpose( Vh[ :, 0:r ], 0, 1 )
    #     inputs_flat[ b, :, : ].copy_( input_lowrank )
    inputs_lowrank, r_actual = svd_approx( inputs_flat, r )

    inputs_lowrank = torch.reshape( inputs_lowrank, [bs, c, h, w] )

    return inputs_lowrank, r_actual


def data_withnoise( inputs, nsr=0.1 ):
    """add noise to inputs
    :param inputs data  ( [batch, ichannel, height, width] )
    :param nsr noise-signal ratio ( float )
    :return data with noise
    """
    noise_mean = 0.0
    noise_std = nsr * torch.ones_like( inputs )
    noise = torch.normal( noise_mean, noise_std )
    inputs += noise

    return inputs
