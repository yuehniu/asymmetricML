from ctypes import *
from ctypes import POINTER
import numpy as np
import copy

SGXDNNLIB = "lib/libenclave.signed.so"
DNNLIB = "lib/enclave_bridge.so"

# create sgx environment
lib = cdll.LoadLibrary(DNNLIB)
lib.init_ctx_bridge.restype = c_ulong
lib.init_ctx_bridge.argtypes = [c_int, c_bool, c_int, c_bool]
eid = lib.init_ctx_bridge(0, True, 1, False)

# create test samples
H = 128
W = 128
r = 16
max_iter = 1

in_sgx = np.random.randn(H, W).astype(np.float32)
u_T = np.random.randn(r, H).astype(np.float32)
v_T = np.random.rand(r, H).astype(np.float32)
in_py = copy.deepcopy(in_sgx)
uT_py = copy.deepcopy(u_T).transpose()
vT_py = copy.deepcopy(v_T).transpose()
#print("SGX in: {}".format(in_sgx))
#print("Python in: {}".format(in_py))
#print("uT in SGX: {}".format(u_T))

# Call function in SGX
in_ptr = np.ctypeslib.as_ctypes(in_sgx.reshape(-1))
uT_ptr = np.ctypeslib.as_ctypes(u_T.reshape(-1))
vT_ptr = np.ctypeslib.as_ctypes(v_T.reshape(-1))
lib.test_light_SVD_bridge.argtypes = [c_ulong, POINTER(c_float), POINTER(c_float), c_int, POINTER(c_float), c_int, c_int, c_int] 
lib.test_light_SVD_bridge(eid, in_ptr, uT_ptr, H, vT_ptr, W, r, max_iter)
#print("uT @ vT in SGX: {}".format(u_T.transpose() @ v_T))

# Call function in python
for r_ in range(0, r):
    for itr in range(0, max_iter):
        uT_py[ :, [r_] ] = np.matmul(in_py, vT_py[ :, [r_] ] ) / np.dot( vT_py[ :, r_ ], vT_py[ :, r_ ] )
        vT_py[ :, [r_] ] = np.matmul(in_py.transpose(), uT_py[ :, [r_] ]) / np.dot( uT_py[ :, r_ ], uT_py[ :, r_ ] )
    
    in_py = in_py - np.matmul( uT_py[ :, [r_] ], vT_py[ :, [r_] ].transpose() )


#print("uT @ vT in Python: {}".format(uT_py @ vT_py.transpose()))
#print("SGX out: {}".format(in_sgx))
#print("Python out: {}".format(in_py))
# Check results
threshold = 0.001
ispass = True
for val1, val2 in zip(u_T.transpose().reshape(-1), uT_py.reshape(-1)):
    if abs( val1 - val2 ) > threshold:
        print("[TEST] Test failed for u_T (SGX: {:6f}, Python: {:6f})".format(val1, val2))
        ispass = False

for val1, val2 in zip(v_T.transpose().reshape(-1), vT_py.reshape(-1)):
    if abs( val1 - val2 ) > threshold:
        print("[TEST] Test failed for v_T (SGX: {:6f}, Python: {:6f})".format(val1, val2))
        ispass = False

if ispass:
    print("[TEST] Test passed.")
