"""
Attack model in model inversion (mi) attacks.
It is essentially a GAN model with 1) generator; and 2) discriminator.

Ref:
    attack model: http://arxiv.org/abs/1911.07135

Author:

Note:
"""
import torch
import torch.nn as nn


class miGenerator( nn.Module ):
    def __init__( self, nc_base, nc_latent, nc_dec ):
        """
        :param nc_base number of kernels in first conv layer
        :param nc_latent number of channel in latent vectors
        :param nc_dec number of channel to decoder
        """
        super( miGenerator, self ).__init__()
        self.prior = nn.Sequential(
            nn.Conv2d( 3, nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( nc_base ),
            nn.ReLU(),
            nn.Conv2d( nc_base, 2*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 2*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 2*nc_base, 4*nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 4*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, padding=1 ),
            # nn.BatchNorm2d( 4*nc_base ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=2, padding=1  ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=4, padding=1 ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=8, padding=1 ),
            # nn.ReLU(),
            # nn.Conv2d( 4*nc_base, 4*nc_base, 3, 1, dilation=16, padding=1 ),
        )
        self.enc = nn.Sequential(
            nn.ConvTranspose2d( nc_latent, 2*nc_dec, 4, 1, 0, bias=False  ),  # output: 4x4
            nn.BatchNorm2d( 2*nc_dec ),
            nn.ReLU(),
            nn.ConvTranspose2d( 2*nc_dec, nc_dec, 4, 2, 1, bias=False ),  # output: 8x8
            nn.BatchNorm2d( nc_dec ),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d( nc_dec, nc_dec, 4, 2, 1, bias=False ),  # output: 16x16
            # nn.ConvTranspose2d( 2*nc_dec, nc_dec, 4, 2, 1 ),  # output: 16x16
            nn.BatchNorm2d( nc_dec ),
            nn.ReLU(),
            nn.ConvTranspose2d( nc_dec, nc_dec//2, 4, 2, 1, bias=False ),  # output: 32x32
            nn.BatchNorm2d( nc_dec//2 ),
            nn.ReLU(),
            nn.Conv2d( nc_dec//2, nc_dec//4, 3, padding=1, bias=False ),
            nn.BatchNorm2d( nc_dec//4 ),
            nn.ReLU(),
            nn.Conv2d( nc_dec//4, 3, 3, padding=1, bias=False ),
            nn.Tanh(),
        )

    def forward( self, xu, z ):
        # x = torch.cat( ( self.prior( xu ), self.enc( z ) ), 1 )
        x = self.enc( z ) + self.prior( xu )
        return self.dec( x )


class miDiscriminator( nn.Module ):
    def __init__( self, nc_base ):
        """
        :param nc_base number of kernels in first conv layer
        """
        super( miDiscriminator, self ).__init__()

        self.model = nn.Sequential(
            nn.Conv2d( 3, nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( nc_base ),
            nn.ReLU(),
            nn.Conv2d( nc_base, 2*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 2*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 2*nc_base, 4*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 4*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 4*nc_base, 8*nc_base, 3, 2, padding=1, bias=False ),
            nn.BatchNorm2d( 8*nc_base ),
            nn.ReLU(),
            nn.Conv2d( 8*nc_base, 1, 2, bias=False ),
            nn.Sigmoid(),
        )

    def forward( self, x ):
        return self.model( x )
