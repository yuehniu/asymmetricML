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
import time

SGXDNNLIB = "lib/enclave_bridge.so"
DNNLIB = "lib/enclave_bridge.so"
executor = ThreadPoolExecutor( max_workers = 2 )

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
        self.out_buffer = []
        self.gradin_buffer = []
        self.gradw_buffer = []
        self.lyr_config = []
        self.n_lyrs = 0

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
                self.n_lyrs += 1
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
                    self.out_buffer.append( None )
                    self.gradin_buffer.append( torch.zeros( *self.in_memory_desc[ module ] ).cpu() )
                    self.gradw_buffer.append( torch.zeros( n_ochnls, n_ichnls, sz_kern, sz_kern ).cpu() )
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

                    self.out_buffer.append( torch.zeros( *self.in_memory_desc[ module ] ).cpu() )
                    self.gradin_buffer.append( torch.zeros( *self.in_memory_desc[ module ]).cpu() )
                    self.gradw_buffer.append( None )
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
                    
                    self.out_buffer.append( torch.zeros( *self.out_memory_desc[ module ] ).cpu() )
                    self.gradin_buffer.append( torch.zeros( *self.in_memory_desc[ module ] ).cpu() )
                    self.gradw_buffer.append( None )

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
            start = time.time()
            #print("[DEBUG-PyTorch::Conv::FWD] weight: {}".format(weight.data.cpu().numpy().reshape(-1)[0:9]))
            weight_sgx = weight.data.cpu().numpy()
            w_ptr = np.ctypeslib.as_ctypes(weight_sgx.reshape(-1)) 
            self.lib.Conv_fwd_bridge.restype = c_uint
            self.lib.Conv_fwd_bridge.argtypes = [ c_ulong, POINTER(c_float), c_int]
            status = self.lib.Conv_fwd_bridge( self.eid, w_ptr, self.lyr )
            print("[DEBUG-PyTorch::Conv::FWD] sgx FWD time: {}".format( time.time() - start ))
            
            return status

        def conv_fwd_cuda(self, input, weight, bias):
            start = time.time()
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            m = torch.nn.functional.conv2d( input.cuda(), weight, bias, stride, padding ).cpu()
            self.out_buffer[ self.lyr ] = m.detach()
            print("[DEBUG-PyTorch::Conv::FWD] torch FWD time: {}".format( time.time() - start ))

            return m

        #print("[CUDA] Call Conv FWD")
        result_sgx = executor.submit( conv_fwd_sgx,  self, weight )
        result_cuda = executor.submit( conv_fwd_cuda,  self, input, weight, bias )

        output = result_cuda.result()
        status = result_sgx.result()
        #status = conv_fwd_sgx(self, weight)
        #output = conv_fwd_cuda(self, input, weight, bias)
        if( status != 0 ):
            print("[PyTorch] Conv FWD failed in sgx with error code {}".format( hex(status) ) )
            quit()
        #print("[DEBUG-PyTorch::Conv::FWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::Conv::FWD] input: {}".format(input[0,0]))
        #print("[DEBUG-PyTorch::Conv::FWD] output: {}".format(output[0,0]))

        self.lyr += 1

        return output


    def conv_bwd(self, input, gradout, weight, bias):
        def conv_bwd_sgx(self, gradout, weight ):
            start = time.time()
            gradw_sgx = self.gradw_buffer[ self.lyr ]
            gradout_sgx = gradout.numpy()
            gradout_ptr = np.ctypeslib.as_ctypes( gradout_sgx.reshape( -1 ) )
            gradw_ptr = np.ctypeslib.as_ctypes( gradw_sgx.numpy().reshape( -1 ) )
            self.lib.Conv_bwd_bridge.restype = c_uint
            self.lib.Conv_bwd_bridge.argtypes = [ c_ulong, POINTER(c_float), POINTER(c_float), c_int ]
            status = self.lib.Conv_bwd_bridge( self.eid, gradout_ptr, gradw_ptr, self.lyr)
            print("[DEBUG-PyTorch::Conv::BWD] sgx BWD time: {}".format( time.time() - start ))

            return gradw_sgx.cuda(), status

        def conv_bwd_cuda( self, gradout, input, weight ):
            start = time.time()
            input_cuda = input.cuda()
            gradout_cuda = gradout.cuda()
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            gradin = torch.nn.grad.conv2d_input( input_cuda.shape, weight, gradout_cuda, stride, padding ).cpu()
            gradw = torch.nn.grad.conv2d_weight( input_cuda, weight.shape, gradout_cuda, stride, padding )
            gradb = gradout_cuda.sum( (0,2,3) )
            print("[DEBUG-PyTorch::Conv::BWD] torch BWD time: {}".format( time.time() - start ))

            return gradin, gradw, gradb

        self.lyr -= 1
        result_sgx = executor.submit( conv_bwd_sgx, self, gradout, weight )
        result_cuda = executor.submit( conv_bwd_cuda, self, gradout, input, weight )

        gradin, gradw_cuda, gradb = result_cuda.result()
        gradw_sgx, status = result_sgx.result()
        gradw = gradw_sgx + gradw_cuda
        #gradw_sgx, status = conv_bwd_sgx(self, gradout, weight)
        #gradin, gradw_cuda, gradb = conv_bwd_cuda(self, gradout, input, weight)
        #print("[DEBUG-PyTorch::Conv::BWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::Conv::BWD] gradw: {}".format(gradw[0,0]))

        if status != 0:
            print("[PyTorch] Conv BWD failed in sgx with error code {}".format( hex(status) ) )
            quit()

        return gradin, gradw, gradb

    # ReLU interface
    def relu_fwd(self, input):
        """ Forward op in ReLU layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        #print("[CUDA] Call ReLU FWD")
        output = input.detach()
        output_sgx = output.numpy()
        output_ptr = np.ctypeslib.as_ctypes(output_sgx.reshape(-1))
        self.lib.ReLU_fwd_bridge.restype = c_uint
        self.lib.ReLU_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
        status = self.lib.ReLU_fwd_bridge(self.eid, output_ptr, self.lyr)
        if status != 0:
            print("[PyTorch] ReLU FWD failed with error code {}".format(hex(status) ) )
            quit()
        #output = np.maximum(output, 0)

        #print("[DEBUG-PyTorch::ReLU::FWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::ReLU::FWD] input: {}".format(input[0,0]))
        #print("[DEBUG-PyTorch::ReLU::FWD] output: {}".format(output[0,0]))
        self.lyr += 1
        return output

    def relu_bwd(self, gradout):
        """ Backward op in ReLU layer
        :param gradout: gradient on output activation
        :return: gradient on input activation
        """
        self.lyr -= 1  
        gradin = gradout.detach()
        gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
        self.lib.ReLU_bwd_bridge.restype = c_uint
        self.lib.ReLU_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int]
        status = self.lib.ReLU_bwd_bridge(self.eid, gradin_ptr, self.lyr)
        if status != 0:
            print("[PyTorch] ReLU BWD failed with error code {}".format(hex(status)))
            quit()
        #gradin = np.multiply(gradin, input.cpu().clone()>0)
        #print("[DEBUG-PyTorch::ReLU::BWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::ReLU::BWD] gradout: {}".format(gradout.cpu()[0,0]))
        #print("[DEBUG-PyTorch::ReLU::BWD] gradin: {}".format(gradin[0,0]))

        return gradin

    # ReLUPooling interface
    def relupooling_fwd(self, input):
        """ Forward op in ReLUPooling layer
        :param input: 'public' input from previous untrusted execution
        :return: 'public' output to untrusted execution
        """
        if self.lyr == 0:
            input_copy = input.cpu().detach().numpy()
        else:
            input_copy = input.detach().numpy()
        output = self.out_buffer[ self.lyr ]
        input_ptr = np.ctypeslib.as_ctypes(input_copy.reshape(-1))
        output_ptr = np.ctypeslib.as_ctypes(output.detach().numpy().reshape(-1))
        self.lib.ReLUPooling_fwd_bridge.restype = c_uint
        self.lib.ReLUPooling_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int]
        status = self.lib.ReLUPooling_fwd_bridge(self.eid, input_ptr, output_ptr, self.lyr, self.lyr_pooling)
        if status != 0:
            print("[PyTorch] ReLUPooling FWD failed with error code {}".format(hex(status)))
            quit()

        #print("[DEBUG-PyTorch::ReLUPooling::FWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::ReLUPooling::FWD] input: {}".format(input.cpu()[0,0]))
        #print("[DEBUG-PyTorch::ReLUPooling::FWD] output: {}".format(output[0,0]))
        self.lyr += 1
        self.lyr_pooling += 1
        
        if self.lyr == self.n_lyrs:
            #print("[DEBUG-PyTorch::ReLUPooling::FWD] the last ReLUPooling layer.")
            return output.cuda()
        else:
            return output

    def relupooling_bwd(self, input, gradout):
        """ Backward op in ReLUPooling layer
        :param: gradout: gradient on output activation
        :return: gradient on input activation
        """
        self.lyr -= 1
        self.lyr_pooling -= 1
        gradin = torch.zeros_like(self.gradin_buffer[ self.lyr ]).cpu()
        if self.lyr == self.n_lyrs-1:
            gradout_sgx = gradout.cpu().detach().numpy()
        else:
            gradout_sgx = gradout.detach().numpy()
        gradout_ptr = np.ctypeslib.as_ctypes(gradout_sgx.reshape(-1))
        gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
        self.lib.ReLUPooling_bwd_bridge.restype = c_uint
        self.lib.ReLUPooling_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int]
        status = self.lib.ReLUPooling_bwd_bridge(self.eid, gradout_ptr, gradin_ptr, self.lyr, self.lyr_pooling)
        if status != 0:
            print("[PyTorch] ReLUPooling BWD failed with error code {}".format(hex(status)))
            quit()
        #print("[DEBUG-PyTorch::ReLUPooling::BWD] layer: {}".format(self.lyr))
        #print("[DEBUG-PyTorch::ReLUPooling::BWD] gradout: {}".format(gradout.cpu()[0,0]))
        #print("[DEBUG-PyTorch::ReLUPooling::BWD] gradin: {}".format(gradin[0,0]))

        if self.lyr == 0:
            return gradin.cuda()
        else:
            return gradin

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
