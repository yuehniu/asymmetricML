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
batchsize = 8
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
in_gpu = torch.randn( batchsize, n_ochnls, Ho, Wo ).cpu()
in_combine = in_sgx + in_gpu
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

# light-weight SVD
max_iter = 1
output_copy = output_torch.detach().clone().numpy().reshape( batchsize, n_ochnls, Ho*Wo )
u_T_py = np.ones( (batchsize, n_ochnls, r), dtype=np.float32 )
v_T_py = np.transpose(copy.deepcopy(output_copy[:,0:r,:]), (0, 2, 1) )
for b_ in range( batchsize ):
    for r_ in range( r ):
        for itr in range( max_iter ):
            u_T_py[ b_, :, r_:r_+1 ] = np.matmul(output_copy[ b_ ], v_T_py[ b_, :, r_:r_+1 ] ) / np.dot( v_T_py[ b_, :, r_ ], v_T_py[ b_, :, r_ ] )
            v_T_py[ b_, :, r_:r_+1 ] = np.matmul(output_copy[ b_ ].transpose(), u_T_py[ b_, :, r_:r_+1 ]) / np.dot( u_T_py[ b_, :, r_ ], u_T_py[ b_, :, r_ ] )
        output_copy[ b_ ] = output_copy[ b_ ] - np.matmul( u_T_py[ b_, :, r_:r_+1 ], v_T_py[ b_, :, r_:r_+1 ].transpose() )

# Verify u_T and v_T
u_T_t = np.transpose( u_T, (0,2,1))
v_T_t = np.transpose( v_T, (0,2,1))
print("Diff of u_T: {:6f}".format( np.linalg.norm(u_T_t - u_T_py) ) )
print("Diff of v_T: {:6f}".format( np.linalg.norm(v_T_t - v_T_py) ) )

# Verify output
print("Diff of output: {:6f}".format( np.linalg.norm( in_gpu.numpy().reshape(batchsize, n_ochnls, Ho*Wo) - output_copy ) ) )

