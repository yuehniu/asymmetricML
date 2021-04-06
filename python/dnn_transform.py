"""Transform normal DNN model to the asymmetric version
Description:
    This file is aimed to load a normal DNN model definition and transform it to the asymmetric version.
    It contains:
    - transform_model: the top interface of model transform;
    - transform_layer: the interface of each-layer transform;
    - asymConv2D: asymmetric version of Conv2D;
    - asymReLU: asymmetric version of ReLU;
    - asymPooling: asymmetric version of Pooling.

Author:
    Yue (Julien) Niu

Note:
"""
import math
import torch
import torch.nn as nn

from .dnn_sgx import sgxReLU, sgxReLUPooling, sgxConv, sgxShortCut
from .models.resnet import BasicBlock

def transform_model(model, sgxdnn_Obj, use_SGX=True):
    print("[INFO] Transform model to the asymmetric version")

    new_modules = []
    need_sgx_list = []
    m_prev = None
    for m in model.children():
        if isinstance( m, nn.Sequential ):
           for mm in m.children(): 
               new_m, need_sgx = transform_layer( mm,sgxdnn_Obj, use_SGX )
               if isinstance( m_prev, asymReLU ) and (isinstance(mm, nn.MaxPool2d) or isinstance( mm, nn.AvgPool2d )):
                   new_modules.pop(-1)
                   need_sgx_list.pop(-1)
               for mi, sgxi in zip(new_m, need_sgx):
                   new_modules.append(mi)
                   need_sgx_list.append(sgxi)
               m_prev = new_m[ -1 ]
        else:
            new_m, need_sgx = transform_layer(m, sgxdnn_Obj, use_SGX)
            if isinstance( m_prev, asymReLU ) and (isinstance(m, nn.MaxPool2d) or isinstance( m, nn.AvgPool2d )):
                new_modules.pop(-1)
                need_sgx_list.pop(-1)
            for mi, sgxi in zip(new_m, need_sgx):
                new_modules.append(mi)
                need_sgx_list.append(sgxi)
            m_prev = new_modules[ -1 ]

    return torch.nn.Sequential(*new_modules), need_sgx_list

def transform_layer(m, sgxdnn_Obj, use_SGX):
    new_m = []
    need_sgx = []
    if isinstance( m, BasicBlock ): # ResNet Block
        print("[INFO] Convert resnet block")
        config = []
        need_sgx_sub = []
        for mm in m.children():
            if isinstance( mm, nn.Conv2d ):
                config_conv = [ mm.in_channels, mm.out_channels, mm.kernel_size, 
                                mm.stride, mm.padding, mm.dilation ]
                config.append( config_conv )
                need_sgx_sub.append( True )
            if isinstance( mm, nn.ReLU ):
                need_sgx_sub.append( True )
            if isinstance( mm, nn.Sequential ):
                if len( list(mm.children()) ) == 0:
                    need_sgx_sub.append( True )
                    continue
                for mmm in mm.children():
                    if isinstance( mmm, nn.Conv2d ):
                        config_conv = [ mmm.in_channels, mmm.out_channels, mmm.kernel_size, 
                                        mmm.stride, mmm.padding, mmm.dilation ]
                        config.append( config_conv )
                        need_sgx_sub.append( True )
        new_m.append(asymResBlock( sgxdnn_Obj, config ) )
        need_sgx.append( need_sgx_sub )
        new_m.append( asymReLU( sgxdnn_Obj ) )
        need_sgx.append( True )
    if isinstance(m, nn.Conv2d):
        print("[INFO] Convert convolutional layer")
        config = [m.in_channels, m.out_channels, m.kernel_size, m.stride,
                  m.padding, m.dilation]
        if m.in_channels == 3:
            new_m.append(nn.Conv2d(*config))
            need_sgx.append(False)
        else:
            new_m.append(asymConv2D(sgxdnn_Obj, *config))
            need_sgx.append(True)
    elif isinstance(m, nn.BatchNorm2d):
        config = [m.num_features, m.eps, m.momentum]
        new_m.append(nn.BatchNorm2d(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.ReLU):
        print("[INFO] Convert ReLU layer")
        new_m.append(asymReLU(sgxdnn_Obj))
        need_sgx.append(True)
    elif isinstance(m, nn.MaxPool2d) or isinstance( m, nn.AvgPool2d):
        print("[INFO] Convert Max Pooling layer")
        config = [m.kernel_size, m.stride, m.padding]
        new_m.append(asymReLUPooling(sgxdnn_Obj, *config))
        need_sgx.append(True)
        #new_m.append(nn.MaxPool2d(*config))
        #need_sgx.append(False)
    elif isinstance(m, nn.Linear):
        print("[INFO] Convert Linear layer")
        config = [m.in_features, m.out_features]
        new_m.append(nn.Linear(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.Flatten):
        new_m.append(nn.Flatten())
        need_sgx.append(False)

    return new_m, need_sgx

class asymResBlock( nn.Module ):
    def __init__( self, sgxdnn_Obj, config ):
        super(asymResBlock, self).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.config = config

        self.conv0 = asymConv2D( sgxdnn_Obj, *(config[ 0 ]) )
        self.relu0 = asymReLU( sgxdnn_Obj )
        self.conv1 = asymConv2D( sgxdnn_Obj, *(config[ 1 ]) ) 
        if len(config) == 3:
            self.shortcut = asymConv2D( sgxdnn_Obj, *(config[ 2 ]), shortcut=True )
        else:
            self.shortcut = asymShortCut( sgxdnn_Obj )
        #self.relu1 = asymReLU( sgxdnn_Obj )

        self.type="asymResBlock"
    def forward( self, input ):
        out0 = self.relu0( self.conv0( input ) )
        out1 = self.conv1( out0 ) 
        out = out1 + self.shortcut( input ) 
        #out = self.relu1( out1 )
        return out

        
class asymConv2D(nn.Module):
    def __init__(self, sgxdnn_Obj, n_ichnls, n_ochnls, kernel_size, stride, padding, dilation, shortcut=False):
        super(asymConv2D, self).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.in_channels = n_ichnls
        self.out_channels = n_ochnls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.shortcut = shortcut
        self.type = "asymConv2D"

        self.weight = nn.Parameter(torch.Tensor(n_ochnls, n_ichnls, kernel_size[0], kernel_size[1]))
        self.bias = nn.Parameter(torch.Tensor(n_ochnls))

        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        self.bias.data.zero_()

    def forward(self, input):
        return sgxConv.apply(input, self.weight, self.bias, self.sgxdnn, self.shortcut)

class asymShortCut( nn.Module ):
    def __init__( self, sgxdnn_Obj ):
        super( asymShortCut, self ).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.type = "asymShortCut"

    def forward( self, input ):
        return sgxShortCut.apply( input, self.sgxdnn )

class asymReLU(nn.Module):
    def __init__(self, sgxdnn_Obj):
        super(asymReLU, self).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.type = "asymReLU"

    def forward(self, input):
        return sgxReLU.apply(input, self.sgxdnn)

class asymReLUPooling(nn.Module):
    def __init__(self, sgxdnn_Obj, kernel_size, stride, padding):
        super(asymReLUPooling, self).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.type = "asymReLUPooling"
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, input):
        return sgxReLUPooling.apply(input, self.sgxdnn)
