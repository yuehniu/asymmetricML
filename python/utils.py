import torch
import torch.nn as nn
import math
import numpy as np
from .models import vgg11, vgg16, vgg19, resnet20, resnet32

dim_prev = []
def build_network(model_name, **kwargs):
    models = {
        'vgg11': vgg11,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'resnet20': resnet20,
        'resnet32': resnet32
    }

    return models[model_name](**kwargs)

def infer_memory_size(sgx_dnn, model, need_sgx, batchsize, dim_in):
    """
    Infer memory requirement in each layer in SGX
    :param sgx_dnn: class sgxDNN
    :param model: model definition (Asymmetric version)
    :param need_sgx: whether need SGX
    :param batchsize: batch size
    :param dim_in: dimension of input
    :return:
    """
    global dim_prev
    sgx_dnn.batchsize = batchsize
    sgx_dnn.dim_in = dim_in
    dim_block_first = dim_prev[:]
    for module, need in zip(model, need_sgx):
        if module.type == "asymResBlock":
            infer_memory_size( sgx_dnn, module.children(), need, batchsize, dim_in )
        if module.type == "asymShortCut":
            in_dim_cur = dim_prev
            out_dim_cur = in_dim_cur
            sgx_dnn.in_memory_desc[ module ]  = in_dim_cur
            sgx_dnn.out_memory_desc[ module ] = out_dim_cur
            sgx_dnn.lyr_config.append( None )
        if module.type == "asymReLU":
            in_dim_cur = dim_prev
            out_dim_cur = in_dim_cur
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            sgx_dnn.lyr_config.append(None)
        elif module.type == 'asymConv2D' or isinstance(module, torch.nn.Conv2d):
            padding, dilation, kern_sz, stride = module.padding, module.dilation, module.kernel_size, module.stride
            if len(dim_prev) == 0:
                Hi, Wi = dim_in[1], dim_in[2]
            elif stride[ 0 ] == 1:
                Hi, Wi = dim_prev[2], dim_prev[3]
            else:
                Hi, Wi = dim_block_first[ 2 ], dim_block_first[ 3 ]
            Ho = np.floor((Hi + 2*padding[0] - dilation[0]*(kern_sz[0]-1) - 1) / stride[0] + 1)
            Wo = np.floor((Wi + 2*padding[1] - dilation[1]*(kern_sz[1]-1) - 1) / stride[1] + 1)
            Ho, Wo = int(Ho), int(Wo)
            in_dim_cur = [batchsize, module.in_channels, Hi, Wi]
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            out_dim_cur = [batchsize, module.out_channels, Ho, Wo]
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            if module.type == 'asymConv2D':
                sgx_dnn.lyr_config.append([module.in_channels, module.out_channels, kern_sz, stride, padding])
        elif module.type == "asymReLUPooling" or isinstance(module, torch.nn.MaxPool2d) or isinstance( module, torch.nn.AvgPool2d):
            padding, kern_sz, stride = module.padding, module.kernel_size, module.stride
            out_channels, Hi, Wi = dim_prev[1], dim_prev[2], dim_prev[3]
            Ho = np.floor((Hi + 2*padding - kern_sz) / stride + 1)
            Wo = np.floor((Wi + 2*padding - kern_sz) / stride + 1)
            Ho, Wo = int(Ho), int(Wo)
            in_dim_cur = [batchsize, out_channels, Hi, Wi]
            sgx_dnn.in_memory_desc[module] = in_dim_cur
            out_dim_cur = [batchsize, out_channels, Ho, Wo]
            sgx_dnn.out_memory_desc[module] = out_dim_cur
            sgx_dnn.lyr_config.append([out_channels, kern_sz, stride, padding])

        if module.type != "asymResBlock":
            dim_prev = out_dim_cur

def init_model(model):
    for m in model:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
