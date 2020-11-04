//
// Created by Yue Niu on 8/22/20.
//

#ifndef _DNN_SGX_H_
#define _DNN_SGX_H_

#include <vector>
#include "datatype.h"
#include "error_codes.h"

struct sgxContext {
    int nLyrs;
    int batchsize;
    BOOL useSGX;
    BOOL verbose;

    std::vector<lyrConfig*> config = {};
    std::vector<int> sz_bottom = {};
    std::vector<int> sz_top = {};
    std::vector<float *> bottom = {};
    std::vector<float *> top = {};
    std::vector<int *> max_index = {};
    std::vector<float *> w_T = {};
    std::vector<int> n_pchanls = {};
};

extern "C" {
    ATTESTATION_STATUS sgx_init_ctx(sgxContext* sgx_ctx, int n_lyrs, BOOL use_sgx, int batchsize, BOOL verbose);
    ATTESTATION_STATUS sgx_set_lyrs(sgxContext* sgx_ctx, int n_lyrs);
    ATTESTATION_STATUS sgx_set_batchsize(sgxContext* sgx_ctx, int batchsize);
    ATTESTATION_STATUS sgx_set_sgx(sgxContext* sgx_ctx, BOOL use_sgx);
    ATTESTATION_STATUS sgx_set_verbose(sgxContext* sgx_ctx, BOOL verbose);

    //Conv interface
    ATTESTATION_STATUS sgx_add_Conv_ctx(sgxContext* sgx_ctx, int n_ichnls, int n_ochnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int r);
    ATTESTATION_STATUS sgx_Conv_fwd(sgxContext* sgx_ctx, float* w, int lyr, int b_beg, int b_end);
    ATTESTATION_STATUS sgx_Conv_bwd(sgxContext* sgx_ctx, float* gradout, float* gradw, int lyr, int c_beg, int c_end);

    //ReLU interface
    ATTESTATION_STATUS sgx_add_ReLU_ctx(sgxContext* sgx_ctx, int n_chnls, int H, int W);

    ATTESTATION_STATUS sgx_ReLU_fwd(sgxContext* sgx_ctx, float* out, int lyr, int b_beg, int b_end);

    ATTESTATION_STATUS sgx_ReLU_bwd(sgxContext* sgx_ctx, float* gradin, int lyr, int b_beg, int b_end);

    //ReLUPooling interface
    ATTESTATION_STATUS sgx_add_ReLUPooling_ctx(sgxContext* sgx_ctx, int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode);

    ATTESTATION_STATUS sgx_ReLUPooling_fwd(sgxContext* sgx_ctx, float* in, float* out, int lyr, int lyr_pooling, int b_beg, int b_end);
    
    ATTESTATION_STATUS sgx_ReLUPooling_bwd(sgxContext* sgx_ctx, float*gradout, float* gradin, int lyr, int lyr_pooling, int b_beg, int b_end);

    //SVD
    void sgx_light_SVD(float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter);

    int printf(const char* fmt, ...);
}

#endif //_DNN_SGX_H_
