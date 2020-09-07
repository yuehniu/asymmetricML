//
// Created by Yue Niu on 8/22/20.
//

#ifndef _DNN_SGX_H_
#define _DNN_SGX_H_

#include <vector>
#include "datatype.h"

struct sgxContext {
    int nLyrs;
    BOOL useSGX;
    BOOL verbose;

    std::vector<int> sz_bottom = {};
    std::vector<int> sz_top = {};
    std::vector<float *> bottom = {};
    std::vector<float *> top = {};
};

extern "C" {
    void sgx_init_ctx(sgxContext* sgx_ctx, int n_lyrs, BOOL use_sgx, BOOL verbose);
    void sgx_set_lyrs(sgxContext* sgx_ctx, int n_lyrs);
    void sgx_set_sgx(sgxContext* sgx_ctx, BOOL use_sgx);
    void sgx_set_verbose(sgxContext* sgx_ctx, BOOL verbose);

    //ReLU interface
    void sgx_add_ReLU_ctx(sgxContext* sgx_ctx, int batchsize, int n_chanls, int H, int W);

    void sgx_ReLU_fwd(sgxContext* sgx_ctx, float* in, float* out, int lyr);

    void sgx_ReLU_bwd(sgxContext* sgx_ctx, float* gradin, int lyr);
}

#endif //_DNN_SGX_H_
