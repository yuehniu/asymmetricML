"""DNN operators in SGX
Description:
    This file contains essential operators for DNN layers in SGX:
    - sgxConv2D_fwd/bwd: part of Conv2d operations (Forward/Backward);
    - sgxReLU_fwd/bwd: a full private ReLU operator in SGX (Forward/Backward);
    - sgxPooling_fwd/bwd: a full private Pooling operator in SGX (Forward/Backward);
    - sgxSVD: a light-weight SVD approximation in SGX.

Author:
    Yue (Julien) Niu

Note:
"""
from ctypes import *
from ctypes import POINTER
import torch
import numpy as np

SGXDNNLIB = "lib/enclave_bridge.so"
DNNLIB = "lib/enclave_bridge.so"

class sgxDNN(object):
    """A separate SGX execution context in parallel with untrusted PyTorch context
    """
    def __init__(self, use_sgx=True, n_enclaves = 1):
        self.useSGX = use_sgx
        if use_sgx:
            self.lib = cdll.LoadLibrary(SGXDNNLIB)
        else:
            self.lib = cdll.LoadLibrary(DNNLIB)

        self.lyr = 0 # layer index

        self.lib.init_ctx_bridge.restype = c_uint
        self.lib.init_ctx_bridge.argtypes = [c_int, c_bool, c_bool]
        self.eid = self.lib.init_ctx_bridge(0, use_sgx, False)

        self.batchsize = 0
        self.dim_in = []
        self.in_memory_desc = {} # memory description for each layer [batch, channel, H, W]
        self.out_memory_desc = {}

    def reset(self):
        self.lyr = 0


    def sgx_context(self, model, need_SGX):
        """Build SGX context including set hyperparameters, initialize memory
        :param model: transformed model
        :param need_SGX: if need SGX for each layer
        :return:
        """
        for module, need in zip(model, need_SGX):
            if need:
                if module.type == "asymReLU":
                    in_channels = self.in_memory_desc[module][1]
                    H = self.in_memory_desc[module][2]
                    W = self.in_memory_desc[module][3]
                    self.lib.add_ReLU_ctx_bridge.argtypes = [c_uint, c_int, c_int, c_int, c_int]
                    self.lib.add_ReLU_ctx_bridge(self.eid, self.batchsize, in_channels, H, W)

    # ReLU interface
    def relu_fwd(self, input):
        """ Forward op in ReLU layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        output = input.cpu().clone()
        if self.useSGX:
            pass
        else:
            input_ptr = np.ctypeslib.as_ctypes(input.cpu().detach().numpy().reshape(-1))
            output_ptr = np.ctypeslib.as_ctypes(output.numpy().reshape(-1))
            self.lib.ReLU_fwd_bridge.argtypes = [c_uint, POINTER(c_float), POINTER(c_float), c_int]
            self.lib.ReLU_fwd_bridge(self.eid, input_ptr, output_ptr, self.lyr)

        self.lyr += 1
        return output.reshape(input.size()).cuda()

    def relu_bwd(self, input, gradout):
        """ Backward op in ReLU layer
        :param input: 'public' input from previous untrusted execution
        :param gradout: gradient on output activation
        :return: gradient on input activation
        """
        self.lyr -= 1
        gradin = gradout.cpu().clone()
        if self.useSGX:
            pass
        else:
            input_ptr = np.ctypeslib.as_ctypes(input.cpu().detach().numpy().reshape(-1))
            gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
            self.lib.ReLU_bwd_bridge.argtypes = [c_uint, POINTER(c_float), POINTER(c_float), c_int]
            self.lib.ReLU_bwd_bridge(self.eid, input_ptr, gradin_ptr, self.lyr)

        return gradin.reshape(gradout.size()).cuda()

class sgxReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sgx_ctx=None):
        """
        :param ctx:
        :param input: 'public' input from previous untrusted execution; 'private' input is kept in SGX
        sgx_ctx: SGX execution context
        :return: 'public' output to untrusted execution; 'private' output is kept in SGX
        """
        ctx.save_for_backward(input)
        ctx.constant = sgx_ctx
        # return input.clamp(min=0)
        return sgx_ctx.relu_fwd(input)

    @staticmethod
    def backward(ctx, gradout):
        input,  = ctx.saved_tensors
        sgx_ctx = ctx.constant
        return sgx_ctx.relu_bwd(input, gradout), None
