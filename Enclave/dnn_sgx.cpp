//
// Created by Yue Niu on 8/22/20.
//
#include <iostream>
#include <cassert>

#include "dnn_sgx.h"

extern "C" {
    void sgx_init_ctx(sgxContext* sgx_ctx, int n_lyrs, BOOL use_sgx, BOOL verbose) {
        sgx_ctx->nLyrs = n_lyrs;
        sgx_ctx->useSGX = use_sgx;
        sgx_ctx->verbose = verbose;
    }

    void sgx_set_lyrs(sgxContext* sgx_ctx, int n_lyrs) {
        sgx_ctx->nLyrs = n_lyrs;
    }

    void sgx_set_sgx(sgxContext* sgx_ctx, BOOL use_SGX) {
        sgx_ctx->useSGX = use_SGX;
    }

    void sgx_set_verbose(sgxContext* sgx_ctx, BOOL verbose) {
        sgx_ctx->verbose = verbose;
    }

    void sgx_add_ReLU_ctx(sgxContext* sgx_ctx, int batchsize, int n_chnls, int H, int W) {
        //if (sgx_ctx->verbose)
        //    std::cout << "[SGX] add ReLU context." << std::endl;

        int size = batchsize * n_chnls * H * W;
        auto in_ptr = (float*) malloc(sizeof(float) * size);
        auto out_ptr = (float*) malloc(sizeof(float) * size);
        assert(in_ptr != nullptr); assert(out_ptr != nullptr);
        sgx_ctx->bottom.push_back(in_ptr);
        sgx_ctx->top.push_back(out_ptr);
        sgx_ctx->sz_bottom.push_back(size);
        sgx_ctx->sz_top.push_back(size);
    }

    void sgx_ReLU_fwd(sgxContext* sgx_ctx, const float *in, float *out, int lyr) {
        //if (sgx_ctx->verbose)
        //    std::cout << "[SGX] Perform ReLU Forward at layer: " << lyr <<"." << std::endl;

        int size = sgx_ctx->sz_bottom.at(lyr);
        for (int i = 0; i < size; i++) {
            if (*(in+i) < 0.0) *(out+i) = 0.0;
        }
    }

    void sgx_ReLU_bwd(sgxContext* sgx_ctx, const float *in, float *gradin, int lyr) {
        //if (sgx_ctx->verbose)
        //    std::cout << "[SGX] Perform ReLU Backward at layer: " << lyr << "." << std::endl;

        int size = sgx_ctx->sz_bottom.at(lyr);
        for (int i = 0; i < size; i++) {
            if (*(in + i) < 0.0 ) *(gradin + i) = 0.0;
        }
    }
}
