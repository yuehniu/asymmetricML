"""Mutual information utility functions
Desc:
    Functions for estimate mutual information.

Note:
"""
import torch


def get_activations( model, activation_list ):
    """ register hook functions to get actions when performing FWD/BWD
    :param: model, NN model class
    :param: activation_list, list for storing intermediate data
    """
    def hook1( model, input, output ):
        activation_list.append( output.detach() )

    def hook2( model, gradin, gradout ):
        activation_list.append( gradout )

    for m in model.children():
        if isinstance( m, torch.nn.Sequential ):
            for mm in m.children():
                if isinstance( mm, torch.nn.Conv2d ):
                    mm.register_forward_hook( hook1 )
                    mm.register_backward_hook( hook2 )
                    break  # only first conv layer
        else:  # ResNet blocks
            for mm in m.modules():
                if isinstance( mm, torch.nn.Conv2d ):
                    mm.register_forward_hook( hook1 )
                    mm.register_backward_hook( hook2 )
                    break # only first conv layer
