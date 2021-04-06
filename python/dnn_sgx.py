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
SGX_ONLY = False
executor = ThreadPoolExecutor( max_workers = 2 )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
        self.r = 1
        self.rr = 1

        # performance measure
        self.time_fwd = dict()
        self.time_bwd = dict()
        self.time_fwd_sgx = dict()
        self.time_bwd_sgx = dict()

    def reset(self):
        self.lyr = 0
        self.lyr_pooling = 0;

    def clear_perf(self):
        for lyr in self.time_fwd:
            self.time_fwd[lyr] = AverageMeter()
            self.time_fwd_sgx[lyr] = AverageMeter()
        for lyr in self.time_bwd:
            self.time_bwd[lyr] = AverageMeter()
            self.time_bwd_sgx[lyr] = AverageMeter()

    def sgx_context(self, model, need_SGX):
        """Build SGX context including set hyperparameters, initialize memory
        @param model: transformed model
        @param need_SGX: if need SGX for each layer
        @return:
        """
        for module, need in zip(model, need_SGX):
            if need:
                if module.type != "asymResBlock":
                    self.n_lyrs += 1
                if module.type == "asymResBlock": #TODO
                    self.sgx_context( module.children(), need )
                    self.r = min( self.r*2, 8) 
                    self.rr = self.r
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
                    self.lib.add_Conv_ctx_bridge.restype = c_uint
                    self.lib.add_Conv_ctx_bridge.argtypes = [ c_ulong, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
                    status = self.lib.add_Conv_ctx_bridge( self.eid, n_ichnls, n_ochnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, self.rr )
                    if status != 0:
                        print("[PyTorch] Add Conv context failed with error code {}".format(hex(status)))
                        quit()
                    self.out_buffer.append( torch.zeros( *self.out_memory_desc[ module ], device='cpu', pin_memory=True ) )
                    # self.gradin_buffer.append( torch.zeros( *self.in_memory_desc[ module ] ).cpu() ) 
                    self.gradin_buffer.append( None ) 
                    self.gradw_buffer.append( torch.zeros(n_ochnls, n_ichnls, sz_kern, sz_kern, device='cpu', pin_memory=True).reshape(-1) )
                    # self.r = min( self.r*2, 16) 
                    # self.rr = self.r

                    self.time_fwd['Conv'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd['Conv'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_fwd_sgx['Conv'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd_sgx['Conv'+str(self.n_lyrs-1)] = AverageMeter()

                if module.type == "asymShortCut":
                    in_channels = self.in_memory_desc[ module ][ 1 ]
                    H = self.in_memory_desc[module][2]
                    W = self.in_memory_desc[module][3]
                    self.lib.add_ShortCut_ctx_bridge.restype = c_uint
                    self.lib.add_ShortCut_ctx_bridge.argtypes = [ c_ulong, c_int, c_int, c_int ]
                    status = self.lib.add_ShortCut_ctx_bridge( self.eid, in_channels, H, W )
                    if status != 0:
                        print("[PyTorch] Add ReLU context failed with error code {}".format(hex(status)))
                        quit()
                    self.out_buffer.append( None )
                    self.gradin_buffer.append( None )
                    self.gradw_buffer.append( None )
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

                    self.out_buffer.append( None )
                    self.gradin_buffer.append( None )
                    self.gradw_buffer.append( None )

                    self.time_fwd['ReLU'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd['ReLU'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_fwd_sgx['ReLU'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd_sgx['ReLU'+str(self.n_lyrs-1)] = AverageMeter()

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
                    
                    self.out_buffer.append( torch.zeros( *self.out_memory_desc[ module ], device='cpu', pin_memory=True ) )
                    self.gradin_buffer.append( torch.zeros( *self.in_memory_desc[ module ], device='cpu', pin_memory=True ) )
                    self.gradw_buffer.append( None )

                    self.time_fwd['ReLUPooling'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd['ReLUPooling'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_fwd_sgx['ReLUPooling'+str(self.n_lyrs-1)] = AverageMeter()
                    self.time_bwd_sgx['ReLUPooling'+str(self.n_lyrs-1)] = AverageMeter()

    # Conv interface
    def conv_fwd(self, input, weight, bias, shortcut):
        """Conv forward with two concurrent threas
           1. Conv on GPU
           2. Conv on SGX
           @param input: 'public' input to untrusted platform
           @param weight: conv kernels
           @bias bias: conv bias
        """
        def conv_fwd_sgx(self, weight, shortcut):
            #print("[DEBUG-PyTorch::Conv::FWD] weight: {}".format(weight.data.cpu().numpy().reshape(-1)[0:9]))
            weight_t = torch.transpose(weight.detach(), dim0=0, dim1=1)
            weight_sgx = torch.empty_like( weight_t, device='cpu', pin_memory=True )
            weight_sgx.copy_( weight_t, non_blocking = False )
            weight_sgx = weight_sgx.numpy()
            w_ptr = np.ctypeslib.as_ctypes(weight_sgx.reshape(-1)) 
            t = np.array([0], dtype = np.int32)
            t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
            self.lib.Conv_fwd_bridge.restype = c_uint
            self.lib.Conv_fwd_bridge.argtypes = [ c_ulong, POINTER(c_float), c_int, c_bool, POINTER(c_int) ]
            status = self.lib.Conv_fwd_bridge( self.eid, w_ptr, self.lyr, shortcut, t_ptr )
            
            return status, t[0]

        def conv_fwd_cuda(self, input, weight, bias):
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            output_cuda = torch.nn.functional.conv2d( input.pin_memory().cuda( non_blocking = True ), weight, bias, stride, padding )
            output_cpu = torch.empty_like( output_cuda, device='cpu', pin_memory = True )
            output_cpu.copy_( output_cuda, non_blocking = False )
            # self.out_buffer[ self.lyr ] = output_cpu.detach()

            return output_cpu

        # print("[CUDA] Call Conv FWD")

        start = time.time()

        if not SGX_ONLY:
            result_cuda = executor.submit( conv_fwd_cuda,  self, input, weight, bias )
            result_sgx = executor.submit( conv_fwd_sgx,  self, weight, shortcut )
            
            output = result_cuda.result()
            status, time_sgx = result_sgx.result()
            if( status != 0 ):
                print("[PyTorch] Conv FWD failed in sgx with error code {}".format( hex(status) ) )
                quit()
        else:
            status = conv_fwd_sgx( self, weight, shortcut )
            output = self.out_buffer[ self.lyr ]

        self.time_fwd['Conv'+str(self.lyr)].update(time.time() - start)
        self.time_fwd_sgx['Conv'+str(self.lyr)].update(time_sgx/1000000)

        self.lyr += 1

        return output


    def conv_bwd(self, input, gradout, weight, bias):
        def conv_bwd_sgx(self, gradout, weight, sgx_only=False ):
            dim_w = weight.shape
            dim_w_t = [ dim_w[0], dim_w[2], dim_w[3], dim_w[1] ]
            gradw_sgx = self.gradw_buffer[ self.lyr ]
            gradout_sgx = gradout.numpy()
            gradout_ptr = np.ctypeslib.as_ctypes( gradout_sgx.reshape( -1 ) )
            gradw_ptr = np.ctypeslib.as_ctypes( gradw_sgx.numpy() )
            t = np.array([0], dtype = np.int32)
            t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
            self.lib.Conv_bwd_bridge.restype = c_uint
            self.lib.Conv_bwd_bridge.argtypes = [ c_ulong, POINTER(c_float), POINTER(c_float), c_int, POINTER(c_int) ]
            status = self.lib.Conv_bwd_bridge( self.eid, gradout_ptr, gradw_ptr, self.lyr, t_ptr)

            if not SGX_ONLY:
                return gradw_sgx.pin_memory().cuda( non_blocking = True ).reshape(dim_w_t).permute(0, 3, 1, 2), status, t[0]
            else:
                return gradw_sgx.pin_memory().cuda( non_blocking = True ).reshape(dim_w), status, t[0]

        def conv_bwd_cuda( self, gradout, input, weight ):
            input_cuda = input.pin_memory().cuda( non_blocking = True )
            gradout_cuda = gradout.pin_memory().cuda( non_blocking = True )
            stride = self.lyr_config[ self.lyr ][ 3 ]
            padding = self.lyr_config[ self.lyr ][ 4 ]
            gradin = torch.nn.grad.conv2d_input( input_cuda.shape, weight, gradout_cuda, stride, padding )
            gradin_cpu = torch.empty_like( gradin, device='cpu', pin_memory=True )
            gradin_cpu.copy_( gradin, non_blocking=False )
            gradw = torch.nn.grad.conv2d_weight( input_cuda, weight.shape, gradout_cuda, stride, padding )
            gradb = gradout_cuda.sum( (0,2,3) )

            return gradin_cpu, gradw, gradb

        self.lyr -= 1

        start = time.time()

        result_sgx = executor.submit( conv_bwd_sgx, self, gradout, weight, SGX_ONLY )
        result_cuda = executor.submit( conv_bwd_cuda, self, gradout, input, weight )

        gradin, gradw_cuda, gradb = result_cuda.result()
        gradw_sgx, status, time_sgx = result_sgx.result()
        if not SGX_ONLY:
            gradw = gradw_sgx + gradw_cuda
        else:
            gradw = gradw_sgx

        self.time_bwd['Conv'+str(self.lyr)].update(time.time() - start)
        self.time_bwd_sgx['Conv'+str(self.lyr)].update(time_sgx/1000000)

        if status != 0:
            print("[PyTorch] Conv BWD failed in sgx with error code {}".format( hex(status) ) )
            quit()

        return gradin, gradw, gradb

    # ShortCut interface
    def shortcut_fwd( self, input ):
        self.lib.ShortCut_fwd_bridge.restype = c_uint
        self.lib.ShortCut_fwd_bridge.argtypes = [c_ulong, c_int]
        status = self.lib.ShortCut_fwd_bridge(self.eid, self.lyr)
        if status != 0:
            print("[PyTorch] ReLU FWD failed with error code {}".format(hex(status) ) )
            quit()

        self.lyr += 1

        return input

    def shortcut_bwd( self, gradout ):
        self.lyr -= 1

        return gradout


    # ReLU interface
    def relu_fwd(self, input):
        """ Forward op in ReLU layer
        @param input: 'public' input from previous untrusted execution
        @return: 'public' output to untrusted execution
        """
        #print("[CUDA] Call ReLU FWD")
        start = time.time()

        if self.lyr == 0:
            output = torch.empty_like( input, device='cpu', pin_memory=True )
            output.copy_( input, non_blocking=False )
        else:
            output = input.detach()
        output_sgx = output.numpy() # in-place operation
        output_ptr = np.ctypeslib.as_ctypes(output_sgx.reshape(-1))
        t = np.array([0], dtype = np.int32)
        t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
        self.lib.ReLU_fwd_bridge.restype = c_uint
        self.lib.ReLU_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int, POINTER(c_int)]
        status = self.lib.ReLU_fwd_bridge(self.eid, output_ptr, self.lyr, t_ptr)
        if status != 0:
            print("[PyTorch] ReLU FWD failed with error code {}".format(hex(status) ) )
            quit()

        self.time_fwd['ReLU'+str(self.lyr)].update(time.time() - start)
        self.time_fwd_sgx['ReLU'+str(self.lyr)].update(t[0]/1000000)

        self.lyr += 1
        return output

    def relu_bwd(self, gradout):
        """ Backward op in ReLU layer
        @param gradout: gradient on output activation
        @return: gradient on input activation
        """
        self.lyr -= 1;

        start = time.time()

        gradin = gradout.detach() # in-place operation
        gradin_ptr = np.ctypeslib.as_ctypes(gradin.numpy().reshape(-1))
        t = np.array([0], dtype = np.int32)
        t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
        self.lib.ReLU_bwd_bridge.restype = c_uint
        self.lib.ReLU_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), c_int, POINTER(c_int)]
        status = self.lib.ReLU_bwd_bridge(self.eid, gradin_ptr, self.lyr, t_ptr)
        if status != 0:
            print("[PyTorch] ReLU BWD failed with error code {}".format(hex(status)))
            quit()

        self.time_bwd['ReLU'+str(self.lyr)].update(time.time() - start)
        self.time_bwd_sgx['ReLU'+str(self.lyr)].update(t[0]/1000000)

        if self.lyr == 0:
            return gradin.pin_memory().cuda( non_blocking = True )
        else:
            return gradin

    # ReLUPooling interface
    def relupooling_fwd(self, input):
        """ Forward op in ReLUPooling layer
        @param input: 'public' input from previous untrusted execution
        @return: 'public' output to untrusted execution
        """
        start = time.time()

        if self.lyr == 0:
            input_cpu = torch.empty_like( input, device='cpu', pin_memory=True )
            input_cpu.copy_( input, non_blocking=False )
            input_cpu = input.numpy()
        else:
            input_cpu = input.detach().numpy()
        output = self.out_buffer[ self.lyr ]
        input_ptr = np.ctypeslib.as_ctypes(input_cpu.reshape(-1))
        output_ptr = np.ctypeslib.as_ctypes(output.detach().numpy().reshape(-1))
        t = np.array([0], dtype = np.int32)
        t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
        self.lib.ReLUPooling_fwd_bridge.restype = c_uint
        self.lib.ReLUPooling_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int, POINTER(c_int)]
        status = self.lib.ReLUPooling_fwd_bridge(self.eid, input_ptr, output_ptr, self.lyr, self.lyr_pooling, t_ptr)
        if status != 0:
            print("[PyTorch] ReLUPooling FWD failed with error code {}".format(hex(status)))
            quit()

        self.time_fwd['ReLUPooling'+str(self.lyr)].update(time.time() - start)
        self.time_fwd_sgx['ReLUPooling'+str(self.lyr)].update(t[0]/1000000)

        self.lyr += 1
        self.lyr_pooling += 1
        
        if self.lyr == self.n_lyrs:
            return output.pin_memory().cuda( non_blocking = True )
        else:
            return output

    def relupooling_bwd(self, input, gradout):
        """ Backward op in ReLUPooling layer
        @param: gradout: gradient on output activation
        @return: gradient on input activation
        """
        self.lyr -= 1
        self.lyr_pooling -= 1

        start = time.time()

        gradin = torch.zeros_like( input, device='cpu', pin_memory=True )
        if self.lyr == self.n_lyrs-1:
            gradout_cpu = torch.empty_like( gradout, device='cpu', pin_memory=True )
            gradout_cpu.copy_( gradout, non_blocking=False )
            gradout_cpu = gradout_cpu.numpy()
        else:
            gradout_cpu = gradout.detach().numpy()
        gradout_ptr = np.ctypeslib.as_ctypes( gradout_cpu.reshape( -1 ) )
        gradin_ptr = np.ctypeslib.as_ctypes( gradin.numpy().reshape( -1 ) )
        t = np.array([0], dtype = np.int32)
        t_ptr = np.ctypeslib.as_ctypes(t.reshape(-1))
        self.lib.ReLUPooling_bwd_bridge.restype = c_uint
        self.lib.ReLUPooling_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, c_int, POINTER(c_int)]
        status = self.lib.ReLUPooling_bwd_bridge(self.eid, gradout_ptr, gradin_ptr, self.lyr, self.lyr_pooling, t_ptr)
        if status != 0:
            print("[PyTorch] ReLUPooling BWD failed with error code {}".format(hex(status)))
            quit()

        self.time_bwd['ReLUPooling'+str(self.lyr)].update(time.time() - start)
        self.time_bwd_sgx['ReLUPooling'+str(self.lyr)].update(t[0]/1000000)

        if self.lyr == 0:
            return gradin.pin_memory().cuda( non_blocking = True )
        else:
            return gradin

class sgxConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, sgx_ctx=None, shortcut=False):
        """
        @param ctx:
        @param input: 'public' input from previous ReLU/Pooling layers
        @param sgx_ctx: SGX excution context
        @return: 'public' output to untruested execution
        """
        ctx.save_for_backward(input, weight, bias)
        ctx.constant = sgx_ctx
        return sgx_ctx.conv_fwd(input, weight, bias, shortcut)

    @staticmethod
    def backward(ctx, gradout):
        """
        @param ctx:
        @param gradout: gradients on outputs (as input to conv_bwd function)
        @return: gradients on inputs
        @return: gradients on weights
        @return: gradients on bias
        """
        input,weight,bias = ctx.saved_tensors
        sgx_ctx = ctx.constant
        gradin, gradw, gradb = sgx_ctx.conv_bwd(input, gradout, weight, bias)
        return gradin, gradw, gradb, None, None

class sgxShortCut( torch.autograd.Function ):
    @staticmethod
    def forward( ctx, input, sgx_ctx=None ):
        ctx.constant = sgx_ctx
        return sgx_ctx.shortcut_fwd( input )

    @staticmethod
    def backward( ctx, gradout ):
        sgx_ctx = ctx.constant
        return sgx_ctx.shortcut_bwd( gradout ), None

class sgxReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sgx_ctx=None):
        """
        @param ctx:
        @param input: 'public' input from previous untrusted execution; 'private' input is kept in SGX
        @sgx_ctx: SGX execution context
        @return: 'public' output to untrusted execution; 'private' output is kept in SGX
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
        @param ctx:
        @param input: 'public' input from previous untrusted execution; 'private' input is kept in SGX
        @param sgx_ctx: SGX execution context
        @return 'public' output to untrusted execution; 'private' output is kept in SGX
        """
        ctx.save_for_backward(input)
        ctx.constant = sgx_ctx
        return sgx_ctx.relupooling_fwd(input)

    @staticmethod
    def backward(ctx, gradout):
        input, = ctx.saved_tensors
        sgx_ctx = ctx.constant
        return sgx_ctx.relupooling_bwd(input, gradout), None
