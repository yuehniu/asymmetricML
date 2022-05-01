"""Customized layers
Desc:
    This script is to create some customized layers.

Author:
    Yue (Julien) Niu

Note:
"""
import torch


class noisyAct( torch.nn.Module ):
    def __init__( self, mean, std ):
        super( noisyAct, self ).__init__()
        self.mean = mean
        self.std = std

    def forward( self, input ):
        noise_std = self.std * torch.ones_like( input )
        noise = torch.normal( self.mean, noise_std )
        output = input + noise
        return output
