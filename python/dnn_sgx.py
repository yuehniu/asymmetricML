"""DNN operators in SGX
Description:
    This file contains essential operators for DNN layers in SGX:
    - conv_fwd/bwd: part of Conv2d operations (Forward/Backward);
    - relu_fwd/bwd: a full private ReLU operator in SGX (Forward/Backward);
    - relupooling_fwd/bwd: a full private Pooling operator in SGX (Forward/Backward);

Author:
    Yue (Julien) Niu

Note:
"""
from ctypes import *
from ctypes import POINTER
import torch
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor

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
                if module.type == "asymConv2D":
                    n_ichnls = self.in_memory_desc[ module ][ 1 ]
                    n_ochnls = self.out_memory_desc[ module ][ 1 ]
                    sz_kern = module.kernel_size[0]
                    stride = module.stride[0]
                    padding = module.padding[0]
                    Hi = self.in_memory_desc[ module ][ 2 ]
                    Wi = self.in_memory_desc[ module ][ 3 ]
                    Ho = self.out_memory_desc[ module ][ 2 ]
                    Wo = self.out_memory_desc[ module ][ 3 ]
                    r = 1 #TODO: need to be more accurate
                    self.lib.add_Conv_ctx_bridge.restype = c_uint
                    self.lib.add_Conv_ctx_bridge.argtypes = [ c_ulong, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
                    status = self.lib.add_Conv_ctx_bridge( self.eid, n_ichnls, n_ochnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, r )
                    if status != 0:
                        print("[PyTorch] Add Conv context failed with error code {}".format(hex(status)))
                        quit()
                    self.out_memory.append(None) # no need to create output memory space
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

    # Conv interface
    def conv_fwd(self, input, weight, bias):
        """Conv forward with two concurrent threas
           1. Conv on GPU
           2. Conv on SGX
           @param input: 'public' input to untrusted platform
           @param weight: conv kernels
           @bias bias: conv bias
        """
        def conv_fwd_sgx(self, weight):
            w_ptr = np.ctypeslib.as_ctypes(weight.detach().cpu().numpy().reshape(-1)) 
            self.lib.Conv_fwd_bridge.restype = c_uint
            self.lib.Conv_fwd_bridge.argtypes = [ c_ulong, POINTER(c_float), c_int]
            status = self.lib.Conv_fwd_bridge( self.eid, w_ptr, self.lyr )
            
            return status

        def conv_fwd_cuda(self, input, weight, bias):
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            m = torch.nn.functional.conv2d( input, weight, bias, stride, padding )

            return m

        #print("[CUDA] Call Conv FWD")
        executor = ThreadPoolExecutor( max_workers = 2 )
        result_sgx = executor.submit( conv_fwd_sgx,  self, weight )
        result_cuda = executor.submit( conv_fwd_cuda,  self, input, weight, bias )

        status = result_sgx.result()
        output = result_cuda.result()
        if( status != 0 ):
            print("[PyTorch] Conv FWD failed in sgx with error code {}".format( hex(status) ) )
            quit()

        self.lyr += 1

        return output


    def conv_bwd(self, input, gradout, weight, bias):
        def conv_bwd_sgx(self, gradout, weight ):
            gradw_sgx = torch.zeros_like( weight ).cpu()
            gradout_ptr = np.ctypeslib.as_ctypes( gradout.cpu().numpy().reshape( -1 ) )
            gradw_ptr = np.ctypeslib.as_ctypes( gradw_sgx.numpy().reshape( -1 ) )
            self.lib.Conv_bwd_bridge.restype = c_uint
            self.lib.Conv_bwd_bridge.argtypes = [ c_ulong, POINTER(c_float), POINTER(c_float), c_int ]
            status = self.lib.Conv_bwd_bridge( self.eid, gradout_ptr, gradw_ptr, self.lyr)

            return gradw_sgx, status

        def conv_bwd_cuda( self, gradout, input, weight ):
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            gradin = torch.nn.grad.conv2d_input( input.shape, weight, gradout, stride, padding )
            gradw = torch.nn.grad.conv2d_weight( input, weight.shape, gradout, stride, padding )
            gradb = gradout.sum( (0,2,3) )

            return gradin, gradw, gradb

        self.lyr -= 1
        executor = ThreadPoolExecutor( max_workers = 2 )
        result_sgx = executor.submit( conv_bwd_sgx, self, gradout, weight )
        gradw_sgx, status = result_sgx.result()
        result_cuda = executor.submit( conv_bwd_cuda, self, gradout, input, weight )
        gradin, gradw_cuda, gradb = result_cuda.result()

        if status != 0:
            print("[PyTorch] Conv BWD failed in sgx with error code {}".format( hex(status) ) )
            quit()

        return gradin, gradw_sgx.cuda()+gradw_cuda, gradb

    # ReLU interface
    def relu_fwd(self, input):
        """ Forward op in ReLU layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        #print("[CUDA] Call ReLU FWD")
        output = input.cpu().clone().numpy().reshape(-1)
        if not self.useSGX:
            pass
        else:
            output_ptr = np.ctypeslib.as_ctypes(output)
            self.lib.ReLU_fwd_bridge.restype = c_uint
            self.lib.ReLU_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
            status = self.lib.ReLU_fwd_bridge(self.eid, output_ptr, self.lyr)
            if status != 0:
                print("[PyTorch] ReLU FWD failed with error code {}".format(hex(status) ) )
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
        if not self.useSGX:
            pass
        else:
            gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
            self.lib.ReLU_bwd_bridge.restype = c_uint
            self.lib.ReLU_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
            status = self.lib.ReLU_bwd_bridge(self.eid, gradin_ptr, self.lyr)
            if status != 0:
                print("[PyTorch] ReLU BWD failed with error code {}".format(hex(status)))
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
            print("[PyTorch] ReLUPooling FWD failed with error code {}".format(hex(status)))
            quit()

        self.lyr += 1
        self.lyr_pooling += 1
        #print("[DEBUG-PyTorch] input: {}".format(input.cpu()[0,0, 0:8, 0:8]))
        #print("[DEBUG-PyTorch] output: {}".format(output[0,0, 0:4, 0:4]))
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
            print("[PyTorch] ReLUPooling BWD failed with error code {}".format(hex(status)))
            quit()
        #print("[DEBUG-PyTorch] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch] input: {}".format(gradout.cpu()[127,63, 0, 0]))
        #print("[DEBUG-PyTorch] output: {}".format(gradin[127,63, 0:2, 0:2]))

        return gradin.cuda()

class sgxConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sgx_ctx=None):
        """
        :param ctx:
        :param input: 'public' input from previous ReLU/Pooling layers
        :param sgx_ctx: SGX excution context
        :return: 'public' output to untruested execution
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.constant = sgx_ctx
        return sgx_ctx.conv_fwd(input, weight, bias)

    @staticmethod
    def backward(ctx, gradout):
        """
        :param ctx:
        :param gradout: gradients on outputs (as input to conv_bwd function)
        :return: gradients on inputs
        :return: gradients on weights
        :return: gradients on bias
        """
        input,weight,bias = ctx.saved_tensors
        sgx_ctx = ctx.constant
        gradin, gradw, gradb = sgx_ctx.conv_bwd(input, gradout, weight, bias)
        return gradin, gradw, gradb, None

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
        input, = ctx.saved_tensors
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
