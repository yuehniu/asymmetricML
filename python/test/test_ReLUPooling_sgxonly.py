from ctypes import *
from ctypes import POINTER
import numpy as np
import copy
import torch

SGXDNNLIB = "lib/libenclave.signed.so"
DNNLIB = "lib/enclave_bridge.so"

# create sgx environment
lib = cdll.LoadLibrary(DNNLIB)
lib.init_ctx_bridge.restype = c_ulong
lib.init_ctx_bridge.argtypes = [c_int, c_bool, c_int, c_bool]
eid = lib.init_ctx_bridge(0, True, 1, False)

# conv parameters
batchsize = 64
n_ichnls = 16
n_ochnls = 16
sz_kern = 3
stride = 1
padding = 1
Hi = 32
Wi = 32
Ho = 16
Wo = 16
r = 4

#-> Generate samples
in_sgx = torch.randn( batchsize, n_ichnls, Hi, Wi ).cpu()
in_gpu = torch.zeros( batchsize, n_ichnls, Hi, Wi ).cpu()
in_combine = in_sgx + in_gpu
out_sgx = np.zeros( (batchsize, n_ichnls, Ho, Wo), dtype=np.float32 )
u_T = np.zeros( (batchsize, r, n_ichnls), dtype=np.float32 )
v_T = np.zeros( (batchsize, r, Ho*Wo), dtype=np.float32 )

# call ReLU fwd in sgx
in_sgx_ptr = np.ctypeslib.as_ctypes(in_sgx.numpy().reshape(-1))
in_gpu_ptr = np.ctypeslib.as_ctypes( in_gpu.numpy().reshape(-1) )
out_sgx_ptr = np.ctypeslib.as_ctypes( out_sgx.reshape(-1) )
u_T_ptr = np.ctypeslib.as_ctypes( u_T.reshape(-1) )
v_T_ptr = np.ctypeslib.as_ctypes( v_T.reshape(-1) )

lib.test_ReLUPooling_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int]
lib.test_ReLUPooling_fwd_bridge( eid, in_sgx_ptr, in_gpu_ptr, out_sgx_ptr, u_T_ptr, v_T_ptr, batchsize, n_ichnls, Hi, Wi, Ho, Wo, 1, r )

# call ReLU fwd in torch
out_relu_torch = torch.nn.functional.relu( in_combine );
out_pooling_torch = torch.nn.functional.max_pool2d( out_relu_torch, 2, 2, 0)
#print(output_torch[0,0,0])

# Verify output
print("[Test] Diff of output: {:6f}".format( torch.dist(torch.Tensor(out_sgx), out_pooling_torch ) ) )

