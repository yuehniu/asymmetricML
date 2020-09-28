#include <cassert>
#include "dnn_sgx.h"
#include "enclave_t.h"
#include "datatype.h"
#include <typeinfo>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

sgxContext sgx_ctx;

extern "C" {
    uint32_t init_enclave_ctx(int n_lyrs, BOOL use_sgx, int batchsize, BOOL verbose) {
        return sgx_init_ctx(&sgx_ctx, n_lyrs, use_sgx, batchsize, verbose);
    }

    uint32_t set_lyrs_enclave(int n_lyrs) {
        return sgx_set_lyrs(&sgx_ctx, n_lyrs);
    }

    uint32_t set_batchsize_enclave(int batchsize) {
        return sgx_set_batchsize(&sgx_ctx, batchsize);
    }

    uint32_t set_sgx_enclave(BOOL use_sgx) {
        return sgx_set_sgx(&sgx_ctx, use_sgx);
    }

    uint32_t set_verbose_enclave(BOOL verbose) {
        return sgx_set_verbose(&sgx_ctx, verbose);
    }

    uint32_t add_ReLU_ctx_enclave(int n_chnls, int H, int W) {
        char message[] = "[SGX:Trusted] ADD ReLU context\n";
        printf(message);
        return sgx_add_ReLU_ctx(&sgx_ctx, n_chnls, H, W);
    }

    uint32_t ReLU_fwd_enclave(float* out, int lyr) {
        //char message[] = "[SGX:Trusted] Call ReLU FWD\n";
        //printf(message);
        uint32_t status = sgx_ReLU_fwd(&sgx_ctx, out, lyr);

        return status;
    }

    uint32_t ReLU_bwd_enclave(float* gradin, int lyr) {
        return sgx_ReLU_bwd(&sgx_ctx, gradin, lyr);
    }

    uint32_t add_ReLUPooling_ctx_enclave(int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        char message[] = "[SGX:Trusted] ADD ReLUPooling context\n";
        printf(message);
        return sgx_add_ReLUPooling_ctx(&sgx_ctx, n_chnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, mode);
    }

    uint32_t ReLUPooling_fwd_enclave(float* in, float* out, int lyr, int lyr_pooling) {
        return sgx_ReLUPooling_fwd(&sgx_ctx, in, out, lyr, lyr_pooling);
    }

    uint32_t ReLUPooling_bwd_enclave(float* gradout, float* gradin, int lyr, int lyr_pooling) {
        return sgx_ReLUPooling_bwd(&sgx_ctx, gradout, gradin, lyr, lyr_pooling);
    }
}
