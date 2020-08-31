//
// Created by Yue Niu on 8/22/20.
//

#ifndef ASYMMETRICML_APP_H
#define ASYMMETRICML_APP_H

#endif //ASYMMETRICML_APP_H

#include <iostream>
#include <vector>

class sgxContext {
public:
    sgxContext () {nLyrs = 1; useSGX = false;}
    sgxContext (int n_Lyrs, bool use_SGX);

    void set_lyrs(int n_Lyrs);
    void set_SGX(bool use_SGX);
    void enable_verbose(bool verbose);

    void add_ReLU_Ctx(int batchsize, int n_Chnls, int H, int W);

    void relu_fwd(const float* in, float *out, int lyr);
    void relu_bwd(const float *in, float *gradin, int lyr);

private:
    int nLyrs;
    bool useSGX;
    bool verbose;

    std::vector<int> sz_bottom = {};
    std::vector<int> sz_top = {};
    std::vector<float *> bottom = {};
    std::vector<float *> top = {};
};

extern "C" {
    sgxContext* sgxContext_new(int n_Lyrs, bool use_SGX) {
        auto sgxCtx = new sgxContext(n_Lyrs, use_SGX);
        //std::cout << "[DEBUG]: " << sgxCtx << std::endl;
        return sgxCtx;
    }
    void sgxContext_set_lyrs(sgxContext* sgx_Ctx, int n_Lyrs) {
        sgx_Ctx->set_lyrs(n_Lyrs);
    }

    void sgxContext_enable_verbose(sgxContext* sgx_Ctx, bool verbose) {
        sgx_Ctx->enable_verbose(verbose);
    }

    //ReLU interface
    void sgxContext_add_ReLU_Ctx(sgxContext* sgx_Ctx, int batchsize, int n_Chanl, int H, int W){
        sgx_Ctx->add_ReLU_Ctx(batchsize, n_Chanl, H, W);
    }

    void sgxContext_relu_fwd(sgxContext* sgx_Ctx, const float* in, float* out, int lyr) {
        //std::cout << "[DEBUG]: " << sgxCtx << std::endl;
        sgx_Ctx->relu_fwd(in, out, lyr);
    }

    void sgxContext_relu_bwd(sgxContext* sgx_Ctx, const float* in, float* gradin, int lyr) {
        sgx_Ctx->relu_bwd(in, gradin, lyr);
    }
}
