#include "dnn_sgx.h"
#include "datatype.h"

sgxContext sgx_ctx;

extern "C" {
    void init_enclave_ctx(int n_lyrs, BOOL use_sgx, BOOL verbose) {
        sgx_init_ctx(&sgx_ctx, n_lyrs, use_sgx, verbose);
    }

    void set_lyrs_enclave(int n_lyrs) {
        sgx_set_lyrs(&sgx_ctx, n_lyrs);
    }

    void set_sgx_enclave(BOOL use_sgx) {
        sgx_set_sgx(&sgx_ctx, use_sgx);
    }

    void set_verbose_enclave(BOOL verbose) {
        sgx_set_verbose(&sgx_ctx, verbose);
    }

    void add_ReLU_ctx_enclave(int batchsize, int n_chnls, int H, int W) {
        sgx_add_ReLU_ctx(&sgx_ctx, batchsize, n_chnls, H, W);
    }

    void ReLU_fwd_enclave(const float* in, float* out, int lyr) {
        sgx_ReLU_fwd(&sgx_ctx, in, out, lyr);
    }

    void ReLU_bwd_enclave(const float* in, float* gradin, int lyr) {
        sgx_ReLU_bwd(&sgx_ctx, in, gradin, lyr);
    }
}
