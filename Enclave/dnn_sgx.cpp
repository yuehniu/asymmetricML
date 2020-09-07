//
// Created by Yue Niu on 8/22/20.
//
#include <iostream>
#include <cassert>
#include <typeinfo>

#include "dnn_sgx.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#define USE_EIGEN_TENSOR

extern "C" {
    void sgx_init_ctx(sgxContext* sgx_ctx, int n_lyrs, BOOL use_sgx, BOOL verbose) {
        sgx_ctx->nLyrs = n_lyrs;
        sgx_ctx->useSGX = use_sgx;
        sgx_ctx->verbose = verbose;

	Eigen::initParallel();
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
        int size = batchsize * n_chnls * H * W;
        auto in_ptr = (float*) malloc(sizeof(float) * size);
        auto out_ptr = (float*) malloc(sizeof(float) * size);
        assert(in_ptr != nullptr); assert(out_ptr != nullptr);
        sgx_ctx->bottom.push_back(in_ptr);
        sgx_ctx->top.push_back(out_ptr);
        sgx_ctx->sz_bottom.push_back(size);
        sgx_ctx->sz_top.push_back(size);
    }

    void sgx_ReLU_fwd(sgxContext* sgx_ctx, float *in, float *out, int lyr) {
        int size = sgx_ctx->sz_bottom.at(lyr);
	// Merge input
	float* in_sgx = sgx_ctx->bottom.at(lyr);
	for (int i = 0; i < size; i++)
	    *(in_sgx+i) = *(in + i);

        auto in_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(in, size);
        //for (int i = 0; i < size; i++) {
        //     if (*(in+i) < 0.0) *(out+i) = 0.0;

        //}
	Eigen::Tensor<float, 1> out_map = in_map.cwiseMax(static_cast<float>(0));
        out = out_map.data();
    }

    void sgx_ReLU_bwd(sgxContext* sgx_ctx, float *gradin, int lyr) {
	float* in = sgx_ctx->bottom.at(lyr);

        int size = sgx_ctx->sz_bottom.at(lyr);
        //for (int i = 0; i < size; i++) {
        //    if (*(in + i) < 0.0 ) *(gradin + i) = 0.0;
        //}
	auto in_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(in, size);
	auto gradin_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(gradin, size);
	auto in_sign = in_map > static_cast<float>(0);
	gradin_map *= in_sign;
	gradin = gradin_map.data();
    }
}
