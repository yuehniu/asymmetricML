//
// Created by Yue Niu on 8/22/20.
//
#include <cassert>
#include <typeinfo>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "omp.h"

#include "dnn_sgx.h"
#include "enclave_t.h"
#include "error_codes.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#define USE_EIGEN_TENSOR

extern "C" {
    ATTESTATION_STATUS sgx_init_ctx(sgxContext* sgx_ctx, int n_lyrs, BOOL use_sgx, int batchsize, BOOL verbose) {
        sgx_ctx->nLyrs = n_lyrs;
        sgx_ctx->useSGX = use_sgx;
	sgx_ctx->batchsize = batchsize;
        sgx_ctx->verbose = verbose;

	//Eigen::initParallel();
        //omp_set_num_threads(NUM_THREADS);

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_set_lyrs(sgxContext* sgx_ctx, int n_lyrs) {
        sgx_ctx->nLyrs = n_lyrs;

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_set_batchsize(sgxContext* sgx_ctx, int batchsize) {
        sgx_ctx->batchsize = batchsize;

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_set_sgx(sgxContext* sgx_ctx, BOOL use_SGX) {
        sgx_ctx->useSGX = use_SGX;
        
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_set_verbose(sgxContext* sgx_ctx, BOOL verbose) {
        sgx_ctx->verbose = verbose;

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_add_ReLU_ctx(sgxContext* sgx_ctx, int n_chnls, int H, int W) {
        relu_Config* relu_conf = ( relu_Config* )malloc( sizeof( relu_Config ) );
	relu_conf->n_chnls = n_chnls;
        lyrConfig* lyr_conf = ( lyrConfig* )malloc( sizeof( lyrConfig ) );
        lyr_conf->relu_conf = relu_conf;
	sgx_ctx->config.push_back(lyr_conf);
        int batchsize = sgx_ctx->batchsize;

        int size = batchsize * n_chnls * H * W;
        auto in_ptr = (float*) malloc(sizeof(float) * size);
        auto out_ptr = (float*) malloc(sizeof(float) * size);
        if (!in_ptr || !out_ptr) {
            return MALLOC_ERROR;
        }
        sgx_ctx->bottom.push_back(in_ptr);
        sgx_ctx->top.push_back(out_ptr);
        sgx_ctx->sz_bottom.push_back(size);
        sgx_ctx->sz_top.push_back(size);

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_ReLU_fwd(sgxContext* sgx_ctx, float *out, int lyr) {
        int size = sgx_ctx->sz_bottom.at(lyr);
        //std::string s = std::to_string(size);
        //s = s + "\n";
        //const char* message = s.c_str();
        //printf(message);
	// Merge input
	float* in_sgx = sgx_ctx->bottom.at(lyr);
	for (int i = 0; i < size; i++){
	    *(in_sgx+i) = *(out + i);
	}

        //auto in_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(in_sgx, size);
        //auto out_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(out, size);
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
             if (*(in_sgx+i) < 0.0) *(out+i) = 0.0;
        }
	//Eigen::Tensor<float, 1> out_map = in_map.cwiseMax(static_cast<float>(0));
        //out = out_map.data();
        //char message[] = "[SGX:Trusted+] Call ReLU FWD\n";
        //printf(message);
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_ReLU_bwd(sgxContext* sgx_ctx, float *gradin, int lyr) {
	float* in = sgx_ctx->bottom.at(lyr);

        int size = sgx_ctx->sz_bottom.at(lyr);
        #pragma omp parallel
        for (int i = 0; i < size; i++) {
            if (*(in + i) < 0.0 ) *(gradin + i) = 0.0;
        }
	//auto in_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(in, size);
	//auto gradin_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(gradin, size);
	//auto in_sign = in_map > static_cast<float>(0);
	//gradin_map *= in_sign;
	//gradin = gradin_map.data();

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_add_ReLUPooling_ctx(sgxContext* sgx_ctx, int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        relupooling_Config* relupooling_conf = (relupooling_Config*)malloc(sizeof(relupooling_Config));
        lyrConfig* lyr_config = ( lyrConfig* )malloc( sizeof( lyrConfig ) );
	relupooling_conf->n_chnls = n_chnls;
	relupooling_conf->sz_kern = sz_kern;
	relupooling_conf->stride = stride;
	relupooling_conf->padding = padding;
	relupooling_conf->Hi = Hi; relupooling_conf->Wi = Wi;
	relupooling_conf->Ho = Ho; relupooling_conf->Wo = Wo;
        relupooling_conf->mode = mode;
        lyr_config->relupooling_conf = relupooling_conf;
	sgx_ctx->config.push_back(lyr_config);

	int sz_in = sgx_ctx->batchsize * n_chnls * Hi * Wi;
	auto in_ptr = (float*)malloc(sizeof(float) * sz_in);
	int sz_out = sgx_ctx->batchsize * n_chnls * Ho * Wo;
	auto out_ptr = (float*)malloc(sizeof(float) * sz_out);
        auto max_indx_ptr = ( int* )malloc( sizeof(int)*sz_out );
        if ( !in_ptr || !out_ptr || ! max_indx_ptr ) {
            return MALLOC_ERROR;
        }
	sgx_ctx->bottom.push_back(in_ptr); sgx_ctx->top.push_back(out_ptr);
	sgx_ctx->sz_bottom.push_back(sz_in); sgx_ctx->sz_top.push_back(sz_out);
	sgx_ctx->max_index.push_back(max_indx_ptr);

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_ReLUPooling_fwd(sgxContext* sgx_ctx, float *in, float *out, int lyr, int lyr_pooling) {
        if (!out || !in) {
            return ERROR_UNEXPECTED;
        }

        int sz_in = sgx_ctx->sz_bottom.at(lyr);
	lyrConfig* lyr_config = sgx_ctx->config.at(lyr);
        relupooling_Config* relupooling_conf = lyr_config->relupooling_conf;
	int n_chnls = relupooling_conf->n_chnls;
	int Wi = relupooling_conf->Wi;
	int Hi = relupooling_conf->Hi;
	int Wo = relupooling_conf->Wo;
	int Ho = relupooling_conf->Ho;
        int sz_kern = relupooling_conf->sz_kern;
        int stride = relupooling_conf->stride;
        int mode = relupooling_conf->mode;
        // Merge input (no input from trusted execution yet)
	float* in_sgx = sgx_ctx->bottom.at(lyr);
	int* max_indx_ptr = sgx_ctx->max_index.at(lyr_pooling);
	for ( int i = 0; i < sz_in; i++ )
            *(in_sgx + i) = *(in + i);
        float* out_ptr = out;
        int batchsize = sgx_ctx->batchsize;
	
        //int i = 0;
        #pragma omp parallel
        for ( int b_ = 0; b_ < batchsize; b_++ ){
	    for ( int c_ = 0; c_ < n_chnls; c_++ ) {
	        int offset1 = ( b_*n_chnls + c_ ) * Wi * Hi;
		for ( int ho_=0; ho_<Ho; ho_++ ) {
		    for ( int wo_=0; wo_<Wo; wo_++ ) {
                        int pos_left = wo_ * stride;
                        int pos_right = pos_left + sz_kern;
                        int pos_top = ho_ * stride;
                        int pos_bottom = pos_top + sz_kern;
                        float val_reduced = 0.0;
                        int offset_o = ( b_*n_chnls + c_ ) * Wo * Ho + ho_*Wo + wo_;
                        int max_indx = -1;
                        for ( int hi_=pos_top; hi_<pos_bottom; hi_++ ){
                            for ( int wi_=pos_left; wi_<pos_right; wi_++ ){
                                int offset2 = ( hi_*Wi ) + wi_;
                                float val = *(in_sgx + offset1 + offset2);
                                if (val_reduced < val ) {
                                    val_reduced = val;
                                    max_indx = offset2;
                                }
                            }
                        }
                        *( out+offset_o ) = val_reduced;
			*( max_indx_ptr+offset_o ) = max_indx;
                        //std::string s = std::to_string(i);
                        //s = s + "\n";
                        //const char* message = s.c_str();
                        //printf(message);
                        //i = i+1;
		    } } } }

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_ReLUPooling_bwd(sgxContext* sgx_ctx, float* gradout, float* gradin, int lyr, int lyr_pooling) {
        lyrConfig* lyr_config = sgx_ctx->config.at(lyr);
        relupooling_Config* relupooling_conf = lyr_config->relupooling_conf;
        int n_chnls = relupooling_conf->n_chnls;
        int Wi = relupooling_conf->Wi;
        int Hi = relupooling_conf->Hi;
        int Wo = relupooling_conf->Wo;
        int Ho = relupooling_conf->Ho;
        int mode = relupooling_conf->mode;
        int batchsize = sgx_ctx->batchsize;
        int* max_indx_ptr = sgx_ctx->max_index.at(lyr_pooling);
        #pragma omp parallel
        for ( int b_=0; b_<batchsize; b_++ ) {
            for ( int c_=0; c_<n_chnls; c_++ ) {
                int offset1 = ( b_*n_chnls + c_ ) * Wi * Hi;
                int offset2 = ( b_*n_chnls + c_ ) * Wo * Ho;
                for( int ho_=0; ho_<Ho; ho_++ ) {
                    for ( int wo_=0; wo_<Wo; wo_++) {
                       int offset3 = offset2 + ho_*Wo + wo_;
                       int cur_indx = *( max_indx_ptr+offset3 );
                       //*( max_indx_ptr+offset3 ) = -1;
			   if( cur_indx != -1)
                               *(gradin + offset1 + cur_indx) = *(gradout + offset3); 
                       } } } }
        return SUCCESS;
    }

    int printf(const char* fmt, ...) {
        char buf[BUFSIZ] = {'\0'};
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, BUFSIZ, fmt, ap);
        va_end(ap);
        ocall_print_string(buf);
        return (int)strnlen(buf, BUFSIZ-1) + 1;
    }
}
