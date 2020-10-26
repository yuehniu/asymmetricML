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

#define MAX(a, b) ( (a)>(b) ? (a) : (b) )
#define MIN(a, b) ( (a)<(b) ? (a) : (b) )

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

    ATTESTATION_STATUS sgx_add_Conv_ctx(sgxContext* sgx_ctx, int n_ichnls, int n_ochnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int r){
        conv_Config* conv_conf = ( conv_Config* )malloc( sizeof( conv_Config) );
        conv_conf->n_ichnls = n_ichnls;
        conv_conf->n_ochnls = n_ochnls;
        conv_conf->sz_kern = sz_kern;
        conv_conf->stride = stride;
        conv_conf->padding = padding;
        conv_conf->Hi = Hi; conv_conf->Wi = Wi;
        conv_conf->Ho = Ho; conv_conf->Wo = Wo;
        conv_conf->r = r;
        lyrConfig* lyr_conf = ( lyrConfig* )malloc( sizeof( lyrConfig ) );
        lyr_conf->conv_conf = conv_conf;
        sgx_ctx->config.push_back(lyr_conf);

        int batchsize = sgx_ctx->batchsize;
        int sz_u_T = n_ichnls; int sz_v_T = Hi * Wi;
        int sz_in = batchsize * r * ( sz_u_T + sz_v_T );
        int sz_out = batchsize * n_ochnls * Ho * Wo;
        auto in_ptr = ( float* )malloc( sizeof(float) * sz_in );
        auto out_ptr = ( float* )malloc( sizeof(float) * sz_out );
        if (!in_ptr || !out_ptr) {
            return MALLOC_ERROR;
        }
        sgx_ctx->bottom.push_back( in_ptr ); sgx_ctx->top.push_back( out_ptr );
        sgx_ctx->sz_bottom.push_back( sz_in ); sgx_ctx->sz_top.push_back( sz_out );

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_Conv_fwd(sgxContext* sgx_ctx, float* w, int lyr, int b_beg, int b_end){
        lyrConfig* lyr_conf = sgx_ctx->config.at(lyr);
        conv_Config* conv_conf = lyr_conf->conv_conf;
        int batchsize = sgx_ctx->batchsize;
        int n_ichnls = conv_conf->n_ichnls; int n_ochnls = conv_conf->n_ochnls;
        int sz_kern = conv_conf->sz_kern;
        int sz_kern2 = sz_kern * sz_kern;
        int r = conv_conf->r;
        int sz_w = n_ochnls * r * sz_kern * sz_kern;

        //re-arrange kernels
        int b_stride = b_end - b_beg;
        float* w_T = ( float* )malloc( sizeof(float) * b_stride * sz_w );
        //std::memset( w_T, 0, sizeof(float) * batchsize * sz_w );
        float* u_T = sgx_ctx->bottom.at(lyr);
        float* w_T_oc = w_T;
        float* w_oc = w;
        for ( int i = 0; i < b_stride*sz_w; i++ ) {
            *( w_T+i ) = 0.0;
        }
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            w_oc = w;
            float* u_T_ptr = u_T + ( b_ * r * n_ichnls );
            for( int oc_ = 0; oc_ < n_ochnls; oc_++ ){
                for ( int r_ = 0; r_ < r; r_++ ) {
                    float* w_T_r = w_T_oc + ( r_ * sz_kern2 ); 
                    float* u_T_r = u_T_ptr + ( r_ * n_ichnls );
                    for ( int ic_ = 0; ic_ < n_ichnls; ic_++ ) {
                        float* w_r = w_oc + ( ic_ * sz_kern2 );
                        float u_ij = *( u_T_r + ic_ );
                        for ( int k = 0; k < sz_kern2; k++ ) {
                            *( w_T_r+k ) += u_ij * ( *(w_r+k) );
                            //if (lyr == 1 && b_ == 0 && oc_ == 0){
                            //    std::string s = std::to_string(*(w_r+k)) + "\n";
                            //    printf(s.c_str());
                            //}
                        } } } 
                w_T_oc += ( r * sz_kern2 );
                w_oc += ( n_ichnls * sz_kern2 );
            } 
        }

        //->Debug
        /*if ( lyr == 1 && b_beg == 0 ){
            for ( int b_ = b_beg; b_ < b_end; b_++ ) {
                for ( int oc_ = 0; oc_ < 1; oc_++ ) {
                    for ( int r_ = 0; r_ < 1; r_++ ) {
                        std::string s0 = std::to_string(b_) + "-" + std::to_string(oc_) + "-" + std::to_string(r_) + "\n";
                        printf(s0.c_str());
                        for ( int k1 = 0; k1 < sz_kern; k1++ ) {
                            for ( int k2 = 0; k2 < sz_kern; k2++ ) {
                                int i = ( b_*n_ochnls*r + oc_*r + r_)*sz_kern2 + k1* sz_kern + k2;
                                std::string s = "[DEBUG-SGX::Conv::FWD] " + std::to_string( *( w_T+i ) ) + "\t";
                                printf(s.c_str());
                            }
                            printf("\n");
                        }
                        printf("\n");
                    }
                }
            }
        }*/
        //->End Debug

        //convolution forward
        float* out = sgx_ctx->top.at( lyr );
        float* in = sgx_ctx->bottom.at( lyr ) + ( batchsize * r * n_ichnls );
        int Wo = conv_conf->Wo; int Ho = conv_conf->Ho;
        int Wi = conv_conf->Wi; int Hi = conv_conf->Hi;
        int stride = conv_conf->stride;
        int padding = conv_conf->padding;
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            int bi = b_ - b_beg;
            for ( int oc_ = 0; oc_ < n_ochnls; oc_++ ) {
                float* w_b = w_T + ( bi*n_ochnls*r + oc_*r ) * sz_kern2;
                float* out_oc = out + ( b_*n_ochnls + oc_ ) * Wo * Ho;
                for ( int ho_ = 0; ho_ < Ho; ho_++ ) {
                    for ( int wo_ = 0; wo_ < Wo; wo_++ ) {
                        float out_ij = 0.0;
                        int in_left = wo_ * stride - padding;
                        int w_left_beg = MAX( 0, -in_left );
                        int in_right = in_left+sz_kern;
                        in_left = MAX( 0, in_left );
                        in_right = MIN( Wi, in_right );
                        int in_top = ho_ * stride - padding;
                        int w_top_beg = MAX( 0, -in_top );
                        int in_bottom = in_top+sz_kern;
                        in_top = MAX( 0, in_top );
                        in_bottom = MIN( Hi, in_bottom );
                        for ( int r_ = 0; r_ < r; r_++ ) {
                            float* in_r = in + ( b_*r + r_ ) * Hi * Wi;
                            float* w_r = w_b + ( r_ * sz_kern2 );
                            int w_top = w_top_beg;
                            int w_left = w_left_beg;
                            for ( int hi_ = in_top; hi_ < in_bottom; hi_++ ) {
                                for ( int wi_ = in_left; wi_ < in_right; wi_++ ) {
                                    float w_ij = *( w_r + w_top*sz_kern + w_left );
                                    float in_ij = *( in_r + hi_*Wi + wi_ ); 
                                    out_ij += ( w_ij * in_ij );

                                    w_left++;
                                    //if ( b_ == 0 && oc_ == 0 && ho_==0 && wo_==0 && r_==0 ) {
                                    //    std::string s = std::to_string(w_ij) + ":" + std::to_string(in_ij);
                                    //    s = s + "\n";
                                    //    const char* message = s.c_str();
                                    //    printf(message);
                                    //}
                                } 
                                w_left = w_left_beg;
                                w_top++;
                                } }
                        *( out_oc + ho_*Wo + wo_ ) = out_ij;
                        //if(lyr == 1 && b_ == 0 && oc_ == 0 && ho_ == 0 ){
                        //    std::string s = "[DEBUG-SGX::Conv::FWD] out: " + std::to_string(out_ij) + "\n";
                        //    printf(s.c_str());
                        //}
                        //std::string s = std::to_string(b_)+ "-" + 
                        //                std::to_string(oc_) + "-" + 
                        //                std::to_string(ho_) + "-" + 
                        //                std::to_string(wo_);
                    } } } }
        free( w_T );
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_Conv_bwd(sgxContext* sgx_ctx, float* gradout, float* gradw, int lyr, int c_beg, int c_end ){
        lyrConfig* lyr_conf = sgx_ctx->config.at(lyr);
        conv_Config* conv_conf = lyr_conf->conv_conf;
        int batchsize = sgx_ctx->batchsize;
        int r = conv_conf->r;
        int sz_kern = conv_conf->sz_kern;
        int n_ochnls = conv_conf->n_ochnls;
        int n_ichnls = conv_conf->n_ichnls;
        int Hi = conv_conf->Hi; int Wi = conv_conf->Wi;
        int Ho = conv_conf->Ho; int Wo = conv_conf->Wo;
        int stride = conv_conf->stride; int padding = conv_conf->padding;
        float* u_T = sgx_ctx->bottom.at( lyr );
        float* in = sgx_ctx->bottom.at( lyr ) + ( batchsize * r * n_ichnls );
        int c_stride = c_end - c_beg;
        int sz_w = c_stride * r * sz_kern * sz_kern;
        float* dw_T = ( float* )malloc( sizeof(float) * batchsize * sz_w );

        //convolution backward
        for ( int b_ = 0; b_ < batchsize; b_++ ) {
            for ( int oc_ = c_beg; oc_ < c_end; oc_++ ) {
                int ci = oc_ - c_beg;
                for ( int r_ = 0; r_ < r; r_++ ) {
                    float* dw_T_r = dw_T + ( b_*c_stride*r + ci*r + r_ ) * sz_kern * sz_kern;
                    for ( int i = 0; i < sz_kern; i++ ) {
                        for ( int j = 0; j < sz_kern; j++ ) {
                            float dw_ij = 0.0;

                            int in_left = j * stride - padding;
                            int out_left_beg = MAX( 0, -in_left);
                            int in_right = in_left + Wi;
                            in_left = MAX( 0, in_left );
                            in_right = MIN( Wi, in_right );
                            
                            int in_top = i * stride - padding;
                            int out_top_beg = MAX( 0, -in_top );
                            int in_bottom = in_top + Hi;
                            in_top = MAX( 0, in_top );
                            in_bottom = MIN( Hi, in_bottom );

                            float* dout_b = gradout + ( b_*n_ochnls + oc_ ) * Ho * Wo;
                            float* in_b = in + ( b_*r + r_ ) * Hi * Wi;
                            int out_left = out_left_beg; int out_top = out_top_beg;
                            for ( int hi_ = in_top; hi_ < in_bottom; hi_++ ) {
                                for ( int wi_ = in_left; wi_ < in_right; wi_++ ) {
                                    float dout_ij = *( dout_b + out_top*Wo + out_left );
                                    float in_ij = *( in_b + hi_*Wi + wi_);
                                    dw_ij += ( in_ij * dout_ij );

                                    out_left++;
                                }
                                out_left = out_left_beg;
                                out_top++;
                            } 

                            *( dw_T_r + i*sz_kern + j ) = dw_ij;
                        } } } } }
        //for( int k1 = 0; k1 < sz_kern; k1++ ) {
        //    for ( int k2 = 0; k2 < sz_kern; k2++ ) {
        //        int b_ = 2; int oc_ = 0; int r_ = 0;
        //        float* dw_T_r = dw_T + ( b_*n_ochnls*r + oc_*r + r_ ) * sz_kern * sz_kern;
        //        std::string s = std::to_string( *(dw_T_r + k1*sz_kern + k2) ) + "\t";
        //        printf(s.c_str());
        //    }
        //    printf("\n");
        //}

        //re-arrange gradients
        for ( int oc_ = c_beg; oc_ < c_end; oc_++ ) {
            int ci = oc_ - c_beg;
            for( int ic_ = 0; ic_ < n_ichnls; ic_++ ) {
                for ( int i = 0; i < sz_kern; i++ ) {
                    for ( int j = 0; j < sz_kern; j++ ) {
                        float dw_ij = 0.0;
                        for ( int b_ = 0; b_ < batchsize; b_++ ) {
                            for ( int r_ = 0; r_ < r; r_++ ) {
                                int dw_pos = ( b_*c_stride*r + ci*r + r_ ) * sz_kern * sz_kern + i * sz_kern + j;
                                int u_pos = ( b_*n_ichnls*r + r_*n_ichnls + ic_ );
                                float u_ij = *( u_T + u_pos );
                                float dw_T_ij = *( dw_T + dw_pos );
                                dw_ij += ( u_ij * dw_T_ij );

                                //if (b_ == 2 && oc_ == 0 && ic_ == 0 && i == 0 && j == 0) {
                                //    std::string s = std::to_string(u_ij) + " - " + std::to_string(dw_T_ij) + "\n"; 
                                //    printf(s.c_str());
                                //}
                            }
                        }
                        int dw_pos = ( oc_*n_ichnls + ic_ ) * sz_kern * sz_kern + i*sz_kern + j;
                        *( gradw + dw_pos ) = dw_ij;
                    }
                }
            }
        }

        free( dw_T );
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_add_ReLU_ctx(sgxContext* sgx_ctx, int n_chnls, int H, int W) {
        relu_Config* relu_conf = ( relu_Config* )malloc( sizeof( relu_Config ) );
	relu_conf->n_chnls = n_chnls;
        relu_conf->H = H; relu_conf->W = W;
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

    /*
    * ReLU FWD in sgx (only subset of batches, full FWD need call multi-thread of this function)
    * @param sgx_ctx: SGX running context
    * @param out: output pointer
    * @param lyr: current layer index
    * @param b_beg: mini-batch begin
    * @param b_end: mini-batch end
    */
    ATTESTATION_STATUS sgx_ReLU_fwd(sgxContext* sgx_ctx, float *out, int lyr, int b_beg, int b_end) {
        int size = sgx_ctx->sz_bottom.at(lyr);
        //std::string s = std::to_string(size);
        //s = s + "\n";
        //const char* message = s.c_str();
        //printf(message);

	//-> Merge input
        lyrConfig* lyr_conf = sgx_ctx->config.at( lyr );
        int n_chnls = lyr_conf->relu_conf->n_chnls;
        int W = lyr_conf->relu_conf->W; int H = lyr_conf->relu_conf->H;
	float* in_sgx = sgx_ctx->bottom.at( lyr );
        float* out_sgx = NULL;
        if ( lyr > 0) {
            out_sgx = sgx_ctx->top.at( lyr-1 );
        }
        int beg = b_beg * n_chnls * W * H;
        int end = b_end * n_chnls * W * H;
	for (int i = beg; i < end; i++){
            //if( lyr > 0 && i < 4 ) {
            //    std::string s = "[DEBUG-SGX::ReLU::FWD] prev: " + std::to_string( *(out+i) ) + "\n";
            //    printf(s.c_str());
            //}
            if ( lyr > 0 )
	        *(in_sgx+i) = *(out + i) + *( out_sgx + i );
            else
	        *(in_sgx+i) = *(out + i);
	}

        //-> Apply ReLU op
        //auto in_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(in_sgx, size);
        //auto out_map = Eigen::TensorMap<Eigen::Tensor<float, 1>>(out, size);
        //#pragma omp parallel
        for (int i = beg; i < end; i++) {
             if (*(in_sgx+i) < 0.0) 
                 *(out+i) = 0.0;
             else 
                 *(out+i) = *(in_sgx+i);
        }

	//Eigen::Tensor<float, 1> out_map = in_map.cwiseMax(static_cast<float>(0));
        //out = out_map.data();
        //char message[] = "[SGX:Trusted+] Call ReLU FWD\n";
        //printf(message);

        //-> Resplit data using light-weight SVD
        int batchsize = sgx_ctx->batchsize;
        lyr_conf = sgx_ctx->config.at( lyr+1 );
        n_chnls = lyr_conf->conv_conf->n_ichnls;
        //number of principle components in current layer
        int r = lyr_conf->conv_conf->r;
        int sz_u_T = lyr_conf->conv_conf->n_ichnls;
        float* u_T = sgx_ctx->bottom.at( lyr+1 );
        int sz_v_T = lyr_conf->conv_conf->Hi * lyr_conf->conv_conf->Wi;
        float* v_T = sgx_ctx->bottom.at( lyr+1 ) + ( batchsize * r * sz_u_T );
        //Initialize u_T and v_T
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            float* out_b  = out + ( b_ * n_chnls * sz_v_T);
            for ( int r_ = 0; r_ < r; r_++ ) {
                float* u_T_r = u_T + ( (b_*r + r_ ) * sz_u_T );
                for ( int iu = 0; iu < sz_u_T; iu++ ) {
                    *( u_T_r + iu ) = 1.0;
                }
                float* v_T_r = v_T + ( (b_*r + r_) * sz_v_T );
                for ( int iv = 0; iv < sz_v_T; iv++ ) {
                    *( v_T_r + iv ) = *( out_b + (r_ * sz_v_T) + iv );
                }
            }
        }

        lyr_conf = sgx_ctx->config.at( lyr );
        lyr_conf->relu_conf->r = r;
        for( int b_ = b_beg; b_ < b_end; b_++ ) {
            float* u_ptr = u_T + ( b_ * r * sz_u_T );
            float* v_ptr = v_T + ( b_ * r * sz_v_T );
            float* out_ptr = out + ( b_ * n_chnls * H * W );
            sgx_light_SVD(out_ptr, u_ptr, sz_u_T, v_ptr, sz_v_T, r, 1);
            
            //if ( b_ == 1 ) {
            //    for ( int i = 0; i < 32; i++ ) {
            //        std::string s = std::to_string( *(out_ptr+(0*sz_v_T)+i)) + "\n";
            //        printf(s.c_str());
            //    }
            //}
        }

        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_ReLU_bwd(sgxContext* sgx_ctx, float *gradin, int lyr, int b_beg, int b_end) {
	float* in = sgx_ctx->bottom.at(lyr);

        //int size = sgx_ctx->sz_bottom.at(lyr);
        lyrConfig* lyr_conf = sgx_ctx->config.at( lyr );
        int n_chnls = lyr_conf->relu_conf->n_chnls;
        int H = lyr_conf->relu_conf->H; int W = lyr_conf->relu_conf->W;
        int beg = b_beg * n_chnls * H * W;
        int end = b_end * n_chnls * H * W;
        for (int i = beg; i < end; i++) {
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
    ATTESTATION_STATUS sgx_ReLUPooling_fwd(sgxContext* sgx_ctx, float *in, float *out, int lyr, int lyr_pooling, int b_beg, int b_end) {
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

        //-> Merge input 
	float* in_sgx = sgx_ctx->bottom.at(lyr);
        float* out_prev_sgx = NULL;
        if ( lyr > 0 ) {
            out_prev_sgx = sgx_ctx->top.at( lyr-1 );
        }
	int* max_indx_ptr = sgx_ctx->max_index.at(lyr_pooling);
        int beg = b_beg * n_chnls * Hi * Wi;
        int end = b_end * n_chnls * Hi * Wi;
	for ( int i = beg; i < end; i++ ) {
           //if (lyr > 0 && i < 4){
           //    std::string s = "[DEBUG-SGX::ReLUPooling::FWD] prev top: " + std::to_string(*(out_prev_sgx+i)) + "\n";
           //    printf(s.c_str());
           //}
            if ( lyr > 0 )
                *(in_sgx + i) = *(in + i) + *(out_prev_sgx + i);
            else
                *(in_sgx + i) = *(in + i);
        }
        int batchsize = sgx_ctx->batchsize;

        //int i = 0;
        for ( int b_ = b_beg; b_ < b_end; b_++ ){
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
                        //i = i+1;
		    } } } }

        //Resplit data using light-weight SVD
        if ( lyr != sgx_ctx->config.size()-1 ){
            lyrConfig* lyr_conf = sgx_ctx->config.at( lyr+1 );
            int r = lyr_conf->conv_conf->r;
            int sz_u_T = lyr_conf->conv_conf->n_ichnls;
            float* u_T = sgx_ctx->bottom.at( lyr+1 );
            int sz_v_T = lyr_conf->conv_conf->Hi * lyr_conf->conv_conf->Wi;
            float* v_T = u_T + ( batchsize * r * sz_u_T );
            float* out_ptr = out;
            //TODO: Initialize u_T and v_T
            for ( int b_ = b_beg; b_ < b_end; b_++ ) {
                float* out_b  = out_ptr + ( b_ * n_chnls * sz_v_T);
                for ( int r_ = 0; r_ < r; r_++ ) {
                    float* u_T_r = u_T + ( (b_*r + r_ ) * sz_u_T );
                    for ( int iu = 0; iu < sz_u_T; iu++ ) {
                        *( u_T_r + iu ) = 1.0;
                    }
                    float* v_T_r = v_T + ( (b_*r + r_) * sz_v_T );
                    for ( int iv = 0; iv < sz_v_T; iv++ ) {
                        *( v_T_r + iv ) = *( out_b + (r_ * sz_v_T) + iv );
                    }
                }
            }

            lyr_conf = sgx_ctx->config.at( lyr );
            lyr_conf->relupooling_conf->r = r;
            float* u_ptr = u_T;
            float* v_ptr = v_T;
            for( int b_ = b_beg; b_ < b_end; b_++ ) {
                u_ptr = u_T + ( b_ * r * sz_u_T );
                v_ptr = v_T + ( b_ * r * sz_v_T );
                out_ptr = out + ( b_ * n_chnls * Ho * Wo );

                sgx_light_SVD(out_ptr, u_ptr, sz_u_T, v_ptr, sz_v_T, r, 1);
            }
        }

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_ReLUPooling_bwd(sgxContext* sgx_ctx, float* gradout, float* gradin, int lyr, int lyr_pooling, int b_beg, int b_end) {
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
        for ( int b_=b_beg; b_<b_end; b_++ ) {
            for ( int c_=0; c_<n_chnls; c_++ ) {
                int offset1 = ( b_*n_chnls + c_ ) * Wi * Hi;
                int offset2 = ( b_*n_chnls + c_ ) * Wo * Ho;
                for( int ho_=0; ho_<Ho; ho_++ ) {
                    for ( int wo_=0; wo_<Wo; wo_++) {
                       int offset3 = offset2 + ho_*Wo + wo_;
                       int cur_indx = *( max_indx_ptr+offset3 );
                       //if( b_ == 0 && c_ == 0 & ho_ == 0 && wo_ == 0 ) {
                       //    std::string s = "[DEBUG-SGX::ReLUPooling::BWD] indx: " + std::to_string(cur_indx) + "\n";
                       //    printf(s.c_str());
                       //}
                       //*( max_indx_ptr+offset3 ) = -1;
			   if( cur_indx != -1)
                               *(gradin + offset1 + cur_indx) = *(gradout + offset3); 
                       } } } }
        return SUCCESS;
    }

    /*
     * Light-weight SVD to extract trusted/untrusted part from input
     * 
     * @param in Pointer to input tensor
     * @param u_T List of pointers to trusted 'u'
     * @param u_len Length of each vector 'u'
     * @param v_T List of pointer to trusted 'v'
     * @param v_len Length of each vector 'v'
     * @param r Number of principle chananels in trusted platform
     * @param max_iter Max number of iterations for alternating optimization
    */
    void sgx_light_SVD(float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter)
    {
        float eps = 0.000001;
        for ( int r_=0; r_<r; r_++ ) {
            float* u = u_T + ( r_ * u_len );
            float* v = v_T + ( r_ * v_len );
            //Alternating optimization
            for ( int iter=0; iter<max_iter; iter++ ) {
                // compute u
                float v_norm = 0.0; 
                for ( int i=0; i<v_len; i++) {
                    float v_i = *( v+i );
                    v_norm += ( v_i * v_i);
                }
                for ( int i = 0; i < u_len; i++ ) {
                    float u_i = 0.0;
                    for ( int j = 0; j < v_len; j++ ) {
                        float X_ij = *( in + i*v_len + j);
                        float v_j = *( v + j );
                        u_i += ( X_ij * v_j );
                    }
                    *( u+i ) = u_i / (v_norm + eps);
                }

                // computer v
                float u_norm = 0.0; 
                for ( int i = 0; i < u_len; i++ ) {
                    float u_i = *( u+i );
                    u_norm += ( u_i * u_i );
                }
                for ( int j = 0; j < v_len; j++) {
                    float v_j = 0.0;
                    for ( int i = 0; i < u_len; i++ ) {
                        float X_ij = *( in + i*v_len + j );
                        float u_i = *( u+i );
                        v_j += ( X_ij * u_i );
                    }
                    *( v+j ) = v_j / (u_norm + eps);
                }
            }

            // remove r-th principle channel from input
            for ( int i = 0; i < u_len; i++ ) {
                for ( int j = 0; j < v_len; j++ ) {
                    float X_ij = *( in + i*v_len + j );
                    X_ij -= ( *(u+i) * *(v+j) );
                    *( in + i*v_len +j ) = X_ij;
                }
            }
        }
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
