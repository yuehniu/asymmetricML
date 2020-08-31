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
import torch
import torch.nn as nn

from .dnn_sgx import sgxReLU

def transform_model(model, sgxdnn_Obj, use_SGX=True):
    print("[INFO] Transform model to the asymmetric version")

    new_modules = []
    need_sgx_list = []
    for m in model.modules():
        new_m, need_sgx = transform_layer(m, sgxdnn_Obj, use_SGX)
        for mi, sgxi in zip(new_m, need_sgx):
            new_modules.append(mi)
            need_sgx_list.append(sgxi)

    return torch.nn.Sequential(*new_modules), need_sgx_list

def transform_layer(m, sgxdnn_Obj, use_SGX):
    new_m = []
    need_sgx = []
    if isinstance(m, nn.Conv2d): #TODO
        print("[INFO] Convert convolutional layer")
        config = [m.in_channels, m.out_channels, m.kernel_size, m.stride,
                  m.padding, m.dilation, m.groups, m.padding_mode]
        new_m.append(nn.Conv2d(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.BatchNorm2d):
        config = [m.num_features, m.eps, m.momentum]
        new_m.append(nn.BatchNorm2d(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.ReLU): #TODO
        print("[INFO] Convert ReLU layer")
        new_m.append(asymReLU(sgxdnn_Obj))
        need_sgx.append(True)
    elif isinstance(m, nn.MaxPool2d): #TODO
        print("[INFO] Convert Max Pooling layer")
        config = [m.kernel_size, m.stride, m.padding, m.dilation]
        new_m.append(nn.MaxPool2d(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.Linear):
        print("[INFO] Convert Linear layer")
        config = [m.in_features, m.out_features]
        new_m.append(nn.Linear(*config))
        need_sgx.append(False)
    elif isinstance(m, nn.Flatten):
        new_m.append(nn.Flatten())
        need_sgx.append(False)

    return new_m, need_sgx

class asymReLU(nn.Module):
    def __init__(self, sgxdnn_Obj):
        super(asymReLU, self).__init__()
        self.sgxdnn = sgxdnn_Obj
        self.type = "asymReLU"

    def forward(self, input):
        return sgxReLU.apply(input, self.sgxdnn)
