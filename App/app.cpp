//
// Created by Yue Niu on 8/22/20.
//
#include <cassert>

#include "app.h"

#include <iostream>

sgxContext::sgxContext(int n_Lyrs, bool use_SGX) {
    nLyrs = n_Lyrs;
    useSGX = use_SGX;
}

void sgxContext::set_lyrs(int n_Lyrs) {
    nLyrs = n_Lyrs;
}

void sgxContext::set_SGX(bool use_SGX) {
    useSGX = use_SGX;
}

void sgxContext::enable_verbose(bool verbose) {
    this->verbose = verbose;
}

void sgxContext::add_ReLU_Ctx(int batchsize, int n_Chnls, int H, int W) {
    if (this->verbose)
        std::cout << "[SGX] add ReLU context." << std::endl;

    int size = batchsize * n_Chnls * H * W;
    auto in_ptr = (float*) malloc(sizeof(float) * size);
    auto out_ptr = (float*) malloc(sizeof(float) * size);
    assert(in_ptr != nullptr); assert(out_ptr != nullptr);
    bottom.push_back(in_ptr);
    top.push_back(out_ptr);
    sz_bottom.push_back(size);
    sz_top.push_back(size);
}

void sgxContext::relu_fwd(const float *in, float *out, int lyr) {
    if (this->verbose)
        std::cout << "[SGX] Perform ReLU Forward at layer: " << lyr <<"." << std::endl;

    int size = sz_bottom.at(lyr);
    for (int i = 0; i < size; i++) {
        if (*(in+i) < 0.0) *(out+i) = 0.0;
    }
}

void sgxContext::relu_bwd(const float *in, float *gradin, int lyr) {
    if (this->verbose)
        std::cout << "[SGX] Perform ReLU Backward at layer: " << lyr << "." << std::endl;

    int size = sz_bottom.at(lyr);
    for (int i = 0; i < size; i++) {
        if (*(in + i) < 0.0 ) *(gradin+i) = 0.0;
    }
}