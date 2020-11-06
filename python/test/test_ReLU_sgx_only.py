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
Ho = 32
Wo = 32
r = 4

#-> Generate samples
in_sgx = torch.randn( batchsize, n_ochnls, Ho, Wo ).cpu()
in_gpu = torch.zeros( batchsize, n_ochnls, Ho, Wo ).cpu()
in_combine = in_sgx.clone()
u_T = np.zeros( (batchsize, r, n_ochnls), dtype=np.float32 )
v_T = np.zeros( (batchsize, r, Ho*Wo), dtype=np.float32 )

# call ReLU fwd in sgx
in_sgx_ptr = np.ctypeslib.as_ctypes(in_sgx.numpy().reshape(-1))
in_gpu_ptr = np.ctypeslib.as_ctypes( in_gpu.numpy().reshape(-1) )
u_T_ptr = np.ctypeslib.as_ctypes( u_T.reshape(-1) )
v_T_ptr = np.ctypeslib.as_ctypes( v_T.reshape(-1) )

lib.test_ReLU_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int, c_int]
lib.test_ReLU_fwd_bridge( eid, in_sgx_ptr, in_gpu_ptr, u_T_ptr, v_T_ptr, batchsize, n_ochnls, Ho, Wo, 1, r )

# call ReLU fwd in torch
output_torch = torch.nn.functional.relu( in_combine );
#print(output_torch[0,0,0])

# Verify output
print("Diff of output: {:6f}".format( np.linalg.norm( torch.dist( in_gpu, output_torch) ) ) )

