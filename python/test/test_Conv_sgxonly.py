from ctypes import *
from ctypes import POINTER
import numpy as np
import copy
import torch
import time

SGXDNNLIB = "lib/libenclave.signed.so"
DNNLIB = "lib/enclave_bridge.so"
N_THREADS = 8

# create sgx environment
lib = cdll.LoadLibrary(DNNLIB)
lib.init_ctx_bridge.restype = c_ulong
lib.init_ctx_bridge.argtypes = [c_int, c_bool, c_int, c_bool]
eid = lib.init_ctx_bridge(0, True, 1, False)

# conv parameters
batchsize = 64
n_ichnls = 64
n_ochnls = 64
sz_kern = 3
stride = 1
padding = 1
Hi = 112
Wi = 112
Ho = 112
Wo = 112
r = 2

#-> Generate samples
input_orig = torch.randn( batchsize, n_ichnls, Hi, Wi ).cpu()
input_flatten = torch.reshape( input_orig, (batchsize, n_ichnls, -1) )

#-> Test conv fwd
print("Verify Conv fwd...")
weight = torch.randn( n_ochnls, n_ichnls, sz_kern, sz_kern ).cpu()
output_sgx = np.zeros( (batchsize, n_ochnls, Ho, Wo ), dtype=np.float32 )

in_ptr = np.ctypeslib.as_ctypes(input_orig.numpy().reshape(-1))
weight_t = torch.transpose( weight, dim0=0, dim1=1 ).clone()
weight_sgx = weight_t.numpy()
weight_ptr = np.ctypeslib.as_ctypes( weight_sgx.reshape(-1) )
out_ptr = np.ctypeslib.as_ctypes( output_sgx.reshape(-1) )

# call conv fwd in sgx
start = time.time()
lib.test_Conv_fwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int ]
lib.test_Conv_fwd_bridge( eid, in_ptr, weight_ptr, out_ptr, batchsize, n_ichnls, n_ochnls, Hi, Wi, Ho, Wo, r )
print("Time needed in SGX: {}".format( time.time() - start ))

# call conv fwd in torch
start = time.time()
output_torch = torch.nn.functional.conv2d( input_orig, weight, stride=stride, padding=padding)
print("Time needed in PyTorch: {}".format( time.time() - start ))

# Verify weight
"""
b_ = 0
oc_ = 0
r_ = 1
w_sgx = torch.zeros_like(weight[0,0])
for ic in range(n_ichnls):
    w_sgx += us_sub[ b_,ic,r_ ] * weight[ oc_, ic ]
print("weight: {}".format(w_sgx))
"""


# Verify input
"""
input_sgx = torch.reshape(torch.transpose(v_sub, dim0=1, dim1=2), (batchsize, r, Hi, Wi ) )
print(input_sgx[0, r_])
"""

# Verify output
print(output_sgx[63,14,:,:])
print(output_torch[63,14,:,:])
print( "Diff between SGX and Torch execution: {:6f}".format(torch.dist(torch.Tensor(output_sgx), output_torch) ))

#-> Test conv bwd
print("Verify Conv bwd...")
gradout = torch.randn_like( output_torch ).cpu()

# call conv bwd in sgx
gradw_sgx = np.zeros( (n_ochnls, n_ichnls, sz_kern, sz_kern), dtype=np.float32 )
gradout_ptr = np.ctypeslib.as_ctypes( gradout.numpy().reshape(-1) )
gradw_ptr = np.ctypeslib.as_ctypes( gradw_sgx.reshape(-1) )
start = time.time()
lib.test_Conv_bwd_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float)]
lib.test_Conv_bwd_bridge( eid, gradout_ptr, gradw_ptr )
print("Time needed in SGX: {}".format( time.time() - start ))

# call conv bwd in torch
start = time.time()
gradw_torch = torch.nn.grad.conv2d_weight( input_orig, weight.shape, gradout, stride=stride, padding=padding)
print("Time needed in PyTorch: {}".format( time.time() - start ))

# Verify conv betwen gradout and input
"""
b_ = 2;
oc_ = 0;
r_ = 0;
v_sub_t = torch.transpose( v_sub, dim0 = 1, dim1 = 2)
in_b_r = torch.reshape(v_sub_t[b_, r_], (1, 1, Hi, Wi))
gradout_b_oc = torch.reshape(gradout[b_, oc_], (1,1,Ho, Wo))
gradw_b_oc_r_1 = torch.nn.functional.conv2d( in_b_r, gradout_b_oc, stride = stride, padding = padding)
#gradw_b_oc_r_2 = torch.nn.grad.conv2d_weight( in_b_r, [1,1,sz_kern,sz_kern], gradout_b_oc, stride = stride, padding = padding)
print("gradw_b_oc_r: {}".format(gradw_b_oc_r_1))
#print("gradw_b_oc_r_2: {}".format(gradw_b_oc_r_2))
"""

# Verify gradw
print("gradw from sgx: {}".format( gradw_sgx[2,0] ))
print("gradw from torch: {}".format( gradw_torch[2,0] ))
print("Diff between SGX and Torch excution: {:6f}".format(torch.dist( torch.Tensor(gradw_sgx), gradw_torch )))
