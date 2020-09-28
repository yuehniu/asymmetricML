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
    def __init__(self, use_sgx=True, n_enclaves=1, batchsize=1):
        self.useSGX = use_sgx
        if use_sgx:
            self.lib = cdll.LoadLibrary(SGXDNNLIB)
        else:
            self.lib = cdll.LoadLibrary(DNNLIB)

        self.lib.init_ctx_bridge.restype = c_ulong
        self.lib.init_ctx_bridge.argtypes = [c_int, c_bool, c_int, c_bool]
        self.eid = self.lib.init_ctx_bridge(0, use_sgx, batchsize, False)

        self.batchsize = batchsize
        self.dim_in = []
        self.in_memory_desc = {} # memory description for each layer [batch, channel, H, W]
        self.out_memory_desc = {}
        self.out_memory = []
        self.lyr_config = []

    def reset(self):
        self.lyr = 0
        self.lyr_pooling = 0;


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
                    self.lib.add_ReLU_ctx_bridge.restype = c_uint
                    self.lib.add_ReLU_ctx_bridge.argtypes = [c_ulong, c_int, c_int, c_int]
                    status = self.lib.add_ReLU_ctx_bridge(self.eid, in_channels, H, W)
                    if status != 0:
                        print("[PyTorch] Add ReLU context failed with error code {}".format(hex(status)))
                        quit()

                    self.out_memory.append(self.out_memory_desc[module])
                if module.type == "asymReLUPooling":
                    #print("Add ReLUPooling context")
                    in_channels = self.in_memory_desc[module][1]
                    sz_kern = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    Hi = self.in_memory_desc[module][2]
                    Wi = self.in_memory_desc[module][3]
                    Ho = self.out_memory_desc[module][2]
                    Wo = self.out_memory_desc[module][3]
                    self.lib.add_ReLUPooling_ctx_bridge.restype = c_uint
                    self.lib.add_ReLUPooling_ctx_bridge.argtypes = [c_ulong, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
                    status = self.lib.add_ReLUPooling_ctx_bridge(self.eid, in_channels, sz_kern, stride, padding, Hi, Wi, Ho, Wo, 0)
                    if status != 0:
                        print("[PyTorch] Add ReLUPooling context failed with error code {}".format(hex(status)))
                        quit()
                    
                    self.out_memory.append(self.out_memory_desc[module])

    # ReLU interface
    def relu_fwd(self, input):
        """ Forward op in ReLU layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        output = input.cpu().clone().numpy().reshape(-1)
        if self.useSGX:
            pass
        else:
            output_ptr = np.ctypeslib.as_ctypes(output)
            self.lib.ReLU_fwd_bridge.restype = c_uint
            self.lib.ReLU_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
            status = self.lib.ReLU_fwd_bridge(self.eid, output_ptr, self.lyr)
            if status != 0:
                print("[PyTorch] Add ReLU FWD failed with error code {}".format(hex(status)))
                quit()
            #output = np.maximum(output, 0)

        self.lyr += 1
        #print("[DEBUG] input: {}".format(input[0,0]))
        #print("[DEBUG] output: {}".format(output))
        return torch.as_tensor(output).reshape(input.size()).cuda()

    def relu_bwd(self, gradout):
        """ Backward op in ReLU layer
        :param gradout: gradient on output activation
        :return: gradient on input activation
        """
        self.lyr -= 1
        gradin = gradout.cpu().clone()
        if self.useSGX:
            pass
        else:
            gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
            self.lib.ReLU_bwd_bridge.restype = c_uint
            self.lib.ReLU_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
            status = self.lib.ReLU_bwd_bridge(self.eid, gradin_ptr, self.lyr)
            if status != 0:
                print("[PyTorch] Add ReLU BWD failed with error code {}".format(hex(status)))
                quit()
            #gradin = np.multiply(gradin, input.cpu().clone()>0)

        return gradin.reshape(gradout.size()).cuda()

    # ReLUPooling interface
    def relupooling_fwd(self, input):
        """ Forward op in ReLUPooling layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        input_copy = input.cpu().clone()
        output = torch.zeros(self.out_memory[self.lyr], dtype=torch.float32).cpu()
        input_ptr = np.ctypeslib.as_ctypes(input_copy.numpy().reshape(-1))
        output_ptr = np.ctypeslib.as_ctypes(output.numpy().reshape(-1))
        self.lib.ReLUPooling_fwd_bridge.restype = c_uint
        self.lib.ReLUPooling_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int]
        status = self.lib.ReLUPooling_fwd_bridge(self.eid, input_ptr, output_ptr, self.lyr, self.lyr_pooling)
        if status != 0:
            print("[PyTorch] Add ReLUPooling FWD failed with error code {}".format(hex(status)))
            quit()

        self.lyr += 1
        self.lyr_pooling += 1
        #print("[DEBUG-PyTorch] input: {}".format(input.cpu()[127,63, 0:8, 0:8]))
        #print("[DEBUG-PyTorch] output: {}".format(output[127,63, 0:4, 0:4]))
        #quit()
        return output.cuda()

    def relupooling_bwd(self, input, gradout):
        """ Backward op in ReLUPooling layer
        :param: gradout: gradient on output activation
        :return: gradient on input activation
        """
        self.lyr -= 1
        self.lyr_pooling -= 1
        gradin = torch.zeros_like(input, dtype=torch.float32).cpu()
        gradout_ptr = np.ctypeslib.as_ctypes(gradout.cpu().detach().numpy().reshape(-1))
        gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
        self.lib.ReLUPooling_bwd_bridge.restype = c_uint
        self.lib.ReLUPooling_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int]
        status = self.lib.ReLUPooling_bwd_bridge(self.eid, gradout_ptr, gradin_ptr, self.lyr, self.lyr_pooling)
        if status != 0:
            print("[PyTorch] Add ReLUPooling BWD failed with error code {}".format(hex(status)))
            quit()
        #print("[DEBUG-PyTorch] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch] input: {}".format(gradout.cpu()[127,63, 0, 0]))
        #print("[DEBUG-PyTorch] output: {}".format(gradin[127,63, 0:2, 0:2]))

        return gradin.cuda()

class sgxReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sgx_ctx=None):
        """
        :param ctx:
        :param input: 'public' input from previous untrusted execution; 'private' input is kept in SGX
        :sgx_ctx: SGX execution context
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
        return sgx_ctx.relu_bwd(gradout), None

class sgxReLUPooling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sgx_ctx=None):
        """
        :param ctx:
        :param input: 'public' input from previous untrusted execution; 'private' input is kept in SGX
        :param sgx_ctx: SGX execution context
        :return 'public' output to untrusted execution; 'private' output is kept in SGX
        """
        ctx.save_for_backward(input)
        ctx.constant = sgx_ctx
        return sgx_ctx.relupooling_fwd(input)

    @staticmethod
    def backward(ctx, gradout):
        input, = ctx.saved_tensors
        sgx_ctx = ctx.constant
        return sgx_ctx.relupooling_bwd(input, gradout), None
