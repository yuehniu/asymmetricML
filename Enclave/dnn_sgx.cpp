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
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "eigen_spatial_convolutions-inl.h"
#include "eigen_backward_spatial_convolutions.h"


#define MAX(a, b) ( (a)>(b) ? (a) : (b) )
#define MIN(a, b) ( (a)<(b) ? (a) : (b) )

using namespace Eigen;

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
#ifndef SGX_ONLY
        int sz_u_T = n_ichnls; int sz_v_T = Hi * Wi;
        int sz_in = batchsize * r * ( sz_u_T + sz_v_T );
        int sz_out = batchsize * n_ochnls * Ho * Wo;
        auto in_ptr = ( float* )malloc( sizeof(float) * sz_in );
        auto out_ptr = ( float* )malloc( sizeof(float) * sz_out );
        auto w_T_ptr = ( float* )malloc( sizeof(float) * batchsize * n_ochnls * r * sz_kern * sz_kern );
        if (!in_ptr || !out_ptr || !w_T_ptr) {
            return MALLOC_ERROR;
        }
        sgx_ctx->bottom.push_back( in_ptr ); sgx_ctx->top.push_back( out_ptr );
        sgx_ctx->sz_bottom.push_back( sz_in ); sgx_ctx->sz_top.push_back( sz_out );
        sgx_ctx->w_T.push_back( w_T_ptr );
#else
        int sz_in = batchsize * n_ichnls * Hi * Wi;
        int sz_out = batchsize * n_ochnls * Ho * Wo;
        auto in_ptr = ( float* ) malloc( sizeof(float) * sz_in );
        auto out_ptr = ( float* ) malloc( sizeof(float) * sz_out );
        if( !in_ptr || !out_ptr ) return MALLOC_ERROR;
        sgx_ctx->bottom.push_back( in_ptr ); sgx_ctx->top.push_back( out_ptr );
        sgx_ctx->sz_bottom.push_back( sz_in ); sgx_ctx->sz_top.push_back( sz_out );
#endif

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_Conv_fwd(sgxContext* sgx_ctx, float* w, int lyr, int b_beg, int b_end, BOOL shortcut){
        lyrConfig* lyr_conf = sgx_ctx->config.at(lyr);
        conv_Config* conv_conf = lyr_conf->conv_conf;
        int batchsize = sgx_ctx->batchsize;
        int n_ichnls = conv_conf->n_ichnls; int n_ochnls = conv_conf->n_ochnls;
        int sz_kern = conv_conf->sz_kern;
        int sz_kern2 = sz_kern * sz_kern;
        int r = conv_conf->r;
        int sz_w = n_ochnls * r * sz_kern2;
        int Wo = conv_conf->Wo; int Ho = conv_conf->Ho;
        int Wi = conv_conf->Wi; int Hi = conv_conf->Hi;
        int stride = conv_conf->stride;
        int padding = conv_conf->padding;
        float* out = sgx_ctx->top.at( lyr );

        //re-arrange kernels
        int b_stride = b_end - b_beg;
        float* in = sgx_ctx->bottom.at( lyr );
        if ( shortcut == 1) {
            float* in_prev = sgx_ctx->bottom.at( lyr-3 );
            int beg = b_beg * r * n_ichnls;
            int size = b_stride * r * n_ichnls;
            int end = beg + size;
            for( int i = beg; i < end; i++ ) {
                *( in + i ) = *( in_prev + i );
            }
            beg = batchsize * r * n_ichnls + b_beg * r * Hi * Wi;
            size = b_stride * r * Hi * Wi;
            end = beg + size;
            for( int i = beg; i < end; i++ ) {
                *( in + i ) = *( in_prev + i );
            }
        }
#ifndef SGX_ONLY
        //float* w_T = ( float* )malloc( sizeof(float) * b_stride * sz_w );
        float* w_T = sgx_ctx->w_T.at( lyr ) + b_beg * sz_w;
        //std::memset( w_T, 0, sizeof(float) * batchsize * sz_w );
        float* u_T = sgx_ctx->bottom.at(lyr);
        float* w_T_oc = w_T;
        float* w_oc = w;
        //for ( int i = 0; i < b_stride*sz_w; i++ ) {
        //    *( w_T+i ) = 0.0;
        //}
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_w( w, n_ichnls, n_ochnls*sz_kern2 );
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            int bi = b_ - b_beg;
            float* w_T_b = w_T + ( bi * n_ochnls * r * sz_kern2 );
            float* u_T_b = u_T + ( b_ * r * n_ichnls ); 
            Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_w_T( w_T_b, r, n_ochnls*sz_kern2 );
            Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_u_T( u_T_b, r, n_ichnls );
            mat_w_T = mat_u_T * mat_w;
        }

        //->Debug
        /*if ( lyr == 0 && b_beg == 0 ){
            for ( int b_ = 0; b_ < 1; b_++ ) {
                for ( int oc_ = 0; oc_ < 1; oc_++ ) {
                    for ( int r_ = 0; r_ < 1; r_++ ) {
                        std::string s0 = std::to_string(b_) + "-" + std::to_string(oc_) + "-" + std::to_string(r_) + "\n";
                        printf(s0.c_str());
                        for ( int k1 = 0; k1 < sz_kern; k1++ ) {
                            for ( int k2 = 0; k2 < sz_kern; k2++ ) {
                                int i = ( b_*n_ochnls*r + oc_ + r_*n_ochnls)*sz_kern2 + k1* sz_kern + k2;
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
        // Eigen-based implementation
        in = sgx_ctx->bottom.at( lyr ) + ( batchsize * r * n_ichnls );
        for (int b_ = b_beg; b_ < b_end; b_++) {
            int bi = b_ - b_beg;
            float* w_T_ptr = w_T + ( bi*n_ochnls*r ) * sz_kern2;
            TensorMap< Tensor<float, 4, RowMajor> > tensor_w_T(w_T_ptr, r, n_ochnls, sz_kern, sz_kern);
            array<ptrdiff_t, 4> shuffles;
            shuffles[0] = 2; shuffles[1] = 3; shuffles[2] = 0; shuffles[3] = 1;
            Tensor<float, 4, RowMajor> tensor_w = tensor_w_T.shuffle(shuffles);
            float* in_ptr = in + ( b_ * r ) * Hi * Wi; 
            TensorMap< Tensor<float, 3, RowMajor> > tensor_in_b( in_ptr, r, Hi, Wi );
            array<ptrdiff_t, 3> shuffles2;
            shuffles2[0] = 1; shuffles2[1] = 2; shuffles2[2] = 0;
            Tensor<float, 3, RowMajor> tensor_in = tensor_in_b.shuffle(shuffles2);
            float* out_ptr = out + ( b_ * n_ochnls ) * Ho * Wo;
            TensorMap< Tensor<float, 3, RowMajor> > tensor_out_b( out_ptr, n_ochnls, Wo, Ho );
            //Tensor<float, 3, RowMajor> tensor_out = tensor_out_b.shuffle(shuffles2);
            tensor_out_b.shuffle(shuffles2) = SpatialConvolution( tensor_in, tensor_w, stride, stride, PADDING_SAME );
        }
        if ( shortcut == 1) {
            float* out_prev = sgx_ctx->top.at( lyr-1 );
            int beg = b_beg * n_ochnls * Ho * Wo;
            int size = b_stride * n_ochnls * Ho * Wo;
            Map< Matrix<float, 1, Dynamic> > mat_out( out+beg, size );
            Map< Matrix<float, 1, Dynamic> > mat_out_prev( out_prev+beg, size );
            mat_out += mat_out_prev;
        }

        /* Naive implementation
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            int bi = b_ - b_beg;
            for ( int oc_ = 0; oc_ < n_ochnls; oc_++ ) {
                //float* w_b = w_T + ( bi*n_ochnls*r + oc_*r ) * sz_kern2;
                float* w_b = w_T + ( bi*n_ochnls*r + oc_ ) * sz_kern2;
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
                            //float* w_r = w_b + ( r_ * sz_kern2 );
                            float* w_r = w_b + ( r_* n_ochnls * sz_kern2 );
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
                        //if(lyr == 0 && b_ == 41 && oc_ == 15 && ho_ == 0 ){
                        //    std::string s = "[DEBUG-SGX::Conv::FWD] out: " + std::to_string(*(out_oc+ho_*Wo+wo_)) + "\n";
                        //    printf(s.c_str());
                        //}
                        //std::string s = std::to_string(b_)+ "-" + 
                        //                std::to_string(oc_) + "-" + 
                        //                std::to_string(ho_) + "-" + 
                        //                std::to_string(wo_);
                    } } } }
        */
        //free( w_T );
#else
        //convolution forward
        in = sgx_ctx->bottom.at( lyr );
        array<ptrdiff_t, 4> shuffles;
        shuffles[0] = 2; shuffles[1] = 3; shuffles[2] = 0; shuffles[3] = 1;
        array<ptrdiff_t, 3> shuffles2;
        shuffles2[0] = 1; shuffles2[1] = 2; shuffles2[2] = 0;
        int ic_stride = 4;

        // Eigen-based implementation
        for (int b_ = b_beg; b_ < b_end; b_++) {
            float* out_ptr = out + ( b_ * n_ochnls ) * Ho * Wo;
            TensorMap< Tensor<float, 3, RowMajor> > tensor_out_b( out_ptr, n_ochnls, Wo, Ho );
            tensor_out_b.setZero();
            for( int ic_ = 0; ic_ < n_ichnls; ic_+=ic_stride ) {
                float* w_ptr = w + ic_ * n_ochnls * sz_kern * sz_kern;
                TensorMap< Tensor<float, 4, RowMajor> > tensor_w_T(w_ptr, ic_stride, n_ochnls, sz_kern, sz_kern);
                Tensor<float, 4, RowMajor> tensor_w = tensor_w_T.shuffle(shuffles);

                float* in_ptr = in + ( b_ * n_ichnls + ic_ ) * Hi * Wi; 
                TensorMap< Tensor<float, 3, RowMajor> > tensor_in_b( in_ptr, ic_stride, Hi, Wi );
                Tensor<float, 3, RowMajor> tensor_in = tensor_in_b.shuffle(shuffles2);
                tensor_out_b.shuffle(shuffles2) += SpatialConvolution( tensor_in, tensor_w, stride, stride, PADDING_SAME );
            }
        }

#endif
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_Conv_bwd(sgxContext* sgx_ctx, float* gradout, float* gradw, int lyr, int c_beg, int c_end ){
        lyrConfig* lyr_conf = sgx_ctx->config.at(lyr);
        conv_Config* conv_conf = lyr_conf->conv_conf;
        int batchsize = sgx_ctx->batchsize;
        int r = conv_conf->r;
        int sz_kern = conv_conf->sz_kern;
        int sz_kern2 = sz_kern*sz_kern;
        int n_ochnls = conv_conf->n_ochnls;
        int n_ichnls = conv_conf->n_ichnls;
        int Hi = conv_conf->Hi; int Wi = conv_conf->Wi;
        int Ho = conv_conf->Ho; int Wo = conv_conf->Wo;
        int stride = conv_conf->stride; int padding = conv_conf->padding;
        int c_stride = c_end - c_beg;
        int sz_w = r * sz_kern * sz_kern;
#ifndef SGX_ONLY
        float* u_T = sgx_ctx->bottom.at( lyr );
        float* in = sgx_ctx->bottom.at( lyr ) + ( batchsize * r * n_ichnls );
        //float* dw_T = ( float* )malloc( sizeof(float) * batchsize * c_stride * sz_w );
        float* dw_T = sgx_ctx->w_T.at( lyr ) + c_beg * batchsize * sz_w;

        //convolution backward (Eigen-based implementation)
        /*array<ptrdiff_t, 4> shuffles_w;
        shuffles_w[0] = 2; shuffles_w[1] = 3; shuffles_w[2] = 0; shuffles_w[3] = 1;
        array<ptrdiff_t, 4> shuffles_i;
        shuffles_i[0] = 0; shuffles_i[1] = 2; shuffles_i[2] = 3; shuffles_i[3] = 1;
        array<ptrdiff_t, 4> shuffles_o;
        shuffles_o[0] = 0; shuffles_o[1] = 2; shuffles_o[2] = 3; shuffles_o[3] = 1;
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            int bi = b_ - b_beg;

            float* dw_ptr = dw_T + ( bi*n_ochnls*sz_w);
            TensorMap< Tensor< float, 4, RowMajor > > tensor_dw( dw_ptr, r, n_ochnls, sz_kern, sz_kern );
            //TensorMap< Tensor< float, 4, RowMajor > > tensor_dw( dw_ptr, sz_kern, sz_kern, r, n_ochnls );

            float* in_ptr = in + ( b_*r ) * Hi * Wi;
            TensorMap< Tensor< float, 4, RowMajor > > tensor_in( in_ptr, 1, r, Hi, Wi );
            Tensor< float, 4, RowMajor > tensor_in_t = tensor_in.shuffle(shuffles_i);
            //TensorMap< Tensor< float, 4, RowMajor > > tensor_in( in_ptr, 1, Hi, Wi, r );

            float* gradout_ptr = gradout + ( b_ * n_ochnls * Ho * Wo );
            TensorMap< Tensor< float, 4, RowMajor > > tensor_gradout( gradout_ptr, 1, n_ochnls, Ho, Wo);
            Tensor< float, 4, RowMajor > tensor_gradout_t = tensor_gradout.shuffle( shuffles_o );
            //TensorMap< Tensor< float, 4, RowMajor > > tensor_gradout( gradout_ptr, 1, Ho, Wo, n_ochnls);

            tensor_dw.shuffle( shuffles_w ) = SpatialConvolutionBackwardKernel( tensor_in_t, tensor_gradout_t, sz_kern, sz_kern, stride, stride );
            //tensor_dw = SpatialConvolutionBackwardKernel( tensor_in, tensor_gradout, sz_kern, sz_kern, stride, stride );
        }
        //re-arrange gradients
        int thread_i = b_beg / b_stride;
        Map< Matrix<float, Dynamic, Dynamic, ColMajor> > mat_dw_T( dw_T, n_ochnls*sz_kern2, b_stride*r );
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_u_T( u_T+b_beg*r*n_ichnls, b_stride*r, n_ichnls);
        float* gradw_ptr = gradw + (thread_i * n_ochnls *n_ichnls*sz_kern2);
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_gradw( gradw_ptr, n_ochnls*sz_kern2, n_ichnls );
        mat_gradw = mat_dw_T * mat_u_T; */

        
        //convolution backward (Naive implementation)
        for ( int b_ = 0; b_ < batchsize; b_++ ) {
            for ( int oc_ = c_beg; oc_ < c_end; oc_++ ) {
                int ci = oc_ - c_beg;
                for ( int r_ = 0; r_ < r; r_++ ) {
                    //float* dw_T_r = dw_T + ( b_*c_stride*r + ci*r + r_ ) * sz_kern * sz_kern;
                    float* dw_T_r = dw_T + ci*batchsize*sz_w + b_*r + r_;
                    for ( int i = 0; i < sz_kern; i++ ) {
                        for ( int j = 0; j < sz_kern; j++ ) {
                            float dw_ij = 0.0f;

                            int in_left = j * stride - padding;
                            int out_left_beg = MAX( 0, -in_left);
                            int in_right = in_left + Wo;
                            in_left = MAX( 0, in_left );
                            in_right = MIN( Wi, in_right );
                            
                            int in_top = i * stride - padding;
                            int out_top_beg = MAX( 0, -in_top );
                            int in_bottom = in_top + Ho;
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

                            // *( dw_T_r + i*sz_kern + j ) = dw_ij;
                            *( dw_T_r + (i*sz_kern + j)*batchsize*r ) = dw_ij;
                        } } } } }
        //re-arrange gradients
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_dw_T( dw_T, c_stride*sz_kern2, batchsize*r );
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_u_T( u_T, batchsize*r, n_ichnls);
        float* gradw_ptr = gradw + (c_beg *n_ichnls*sz_kern2);
        Map< Matrix<float, Dynamic, Dynamic, RowMajor> > mat_gradw( gradw_ptr, c_stride*sz_kern2, n_ichnls );
        mat_gradw = mat_dw_T * mat_u_T;


        /*
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
        */

        //free( dw_T );
#else
        //convolution backward (Naive implementation)
        float* in = sgx_ctx->bottom.at( lyr );
        for ( int oc_ = c_beg; oc_ < c_end; oc_++ ) {
            for ( int ic_ = 0; ic_ < n_ichnls; ic_++ ) {
                float* dw = gradw + ( oc_*n_ichnls + ic_ )*sz_kern2;
                for ( int i = 0; i < sz_kern; i++ ) {
                    for ( int j = 0; j < sz_kern; j++ ) {
                        float dw_ij = 0.0f;

                        int in_left = j * stride - padding;
                        int out_left_beg = MAX( 0, -in_left);
                        int in_right = in_left + Wo;
                        in_left = MAX( 0, in_left );
                        in_right = MIN( Wi, in_right );
                            
                        int in_top = i * stride - padding;
                        int out_top_beg = MAX( 0, -in_top );
                        int in_bottom = in_top + Ho;
                        in_top = MAX( 0, in_top );
                        in_bottom = MIN( Hi, in_bottom );

                        for ( int b_ = 0; b_ < batchsize; b_++ ) {
                            float* dout_b = gradout + ( b_*n_ochnls + oc_ ) * Ho * Wo;
                            float* in_b = in + ( b_*n_ichnls + ic_ ) * Hi * Wi;
                            int out_top = out_top_beg;
                            for ( int hi_ = in_top; hi_ < in_bottom; hi_++ ) {
                                int out_left = out_left_beg;
                                for ( int wi_ = in_left; wi_ < in_right; wi_++ ) {
                                    float dout_ij = *( dout_b + out_top*Wo + out_left );
                                    float in_ij = *( in_b + hi_*Wi + wi_);
                                    dw_ij += ( in_ij * dout_ij );

                                    out_left++;
                                }
                                out_top++;
                            } 
                        }

                        // *( dw_T_r + i*sz_kern + j ) = dw_ij;
                        *( dw + (i*sz_kern + j) ) = dw_ij;
                        } } } }
#endif
        return SUCCESS;
    }

    ATTESTATION_STATUS sgx_add_ShortCut_ctx( sgxContext* sgx_ctx, int n_chnls, int H, int W ) {
        shortcut_Config* sc_conf = ( shortcut_Config* ) malloc( sizeof( shortcut_Config) );
        sc_conf->n_chnls = n_chnls;
        sc_conf->H = H; sc_conf->W = W;
        lyrConfig* lyr_conf = ( lyrConfig* )malloc( sizeof(lyrConfig) );
        lyr_conf->shortcut_conf = sc_conf;
	sgx_ctx->config.push_back(lyr_conf);

        if( !sc_conf || !lyr_conf ) {
            return MALLOC_ERROR;
        }
        int batchsize = sgx_ctx->batchsize;
        int size = batchsize * n_chnls * H * W;
        auto out_ptr = ( float* ) malloc( sizeof(float*) * size );
        if ( !out_ptr ) {
            return MALLOC_ERROR;
        }

        sgx_ctx->top.push_back( out_ptr );
        sgx_ctx->bottom.push_back( NULL ); // no need to allocate memory
        sgx_ctx->w_T.push_back( NULL );

        sgx_ctx->sz_bottom.push_back( size );
        sgx_ctx->sz_top.push_back( size );


        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_ShortCut_fwd( sgxContext* sgx_ctx, int lyr, int b_beg, int b_end ) {
        lyrConfig* lyr_conf = sgx_ctx->config.at( lyr );
        shortcut_Config* sc_conf = lyr_conf->shortcut_conf;
        int n_chnls = sc_conf->n_chnls;
        int W = sc_conf->W; int H = sc_conf->H;
        int beg = b_beg * n_chnls * H * W;
        int end = b_end * n_chnls * H * W;
        int size = end - beg;
 
        float* out_sgx_1 = sgx_ctx->top.at( lyr - 1 );
        float* out_sgx_2 = sgx_ctx->top.at( lyr - 3 );
        float* out_sgx = sgx_ctx->top.at( lyr );
        Map< Matrix<float, 1, Dynamic> > mat_out_1 (out_sgx_1+beg, size );
        Map< Matrix<float, 1, Dynamic> > mat_out_2 (out_sgx_2+beg, size );
        Map< Matrix<float, 1, Dynamic> > mat_out (out_sgx+beg, size );

        mat_out = mat_out_1 + mat_out_2;

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
        sgx_ctx->w_T.push_back( NULL );

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
	//-> Merge input
        lyrConfig* lyr_conf = sgx_ctx->config.at( lyr );
        int n_chnls = lyr_conf->relu_conf->n_chnls;
        int W = lyr_conf->relu_conf->W; int H = lyr_conf->relu_conf->H;
	float* in_sgx = sgx_ctx->bottom.at( lyr );
        float* out_sgx = NULL;
        int beg = b_beg * n_chnls * W * H;
        int end = b_end * n_chnls * W * H;
        int size = end - beg;
        Map< Matrix<float, 1, Dynamic > > mat_merge( in_sgx+beg, size );
        Map< Matrix<float, 1, Dynamic > > mat_untrust( out+beg, size );
#ifndef SGX_ONLY
        if ( lyr > 0 ) {
            out_sgx = sgx_ctx->top.at( lyr-1 );
            Map< Matrix<float, 1, Dynamic > > mat_sgx( out_sgx+beg, size );
            mat_merge = mat_untrust + mat_sgx;
        }
        else {
            mat_merge = mat_untrust;
        }
#else
        if( lyr > 0) { 
            out_sgx = sgx_ctx->top.at( lyr-1 );
            Map< Matrix<float, 1, Dynamic > > mat_sgx( out_sgx+beg, size );
            mat_merge = mat_sgx;
        }
        else {
            mat_merge = mat_untrust;
        }
#endif

        //-> Apply ReLU op
        mat_untrust = mat_merge.cwiseMax(0.0);

#ifndef SGX_ONLY
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
                    //*( v_T_r + iv ) = 1.0;
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
        }
#else
    float* in_next_lyr = sgx_ctx->bottom.at( lyr + 1);
    for ( int b_ = b_beg; b_ < b_end; b_++ ) {
        int sz_in = n_chnls * H * W;
        float* in_ptr = in_next_lyr + b_ * sz_in;
        float* out_ptr = out + b_ * sz_in;
        for( int i = 0; i < sz_in; i++ ) {
           *( in_ptr + i )  = *( out_ptr + i );
           *( out_ptr + i ) = 0.0f;
        }
    }
#endif

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
        sgx_ctx->w_T.push_back( NULL );

        return SUCCESS;
    }
    ATTESTATION_STATUS sgx_ReLUPooling_fwd(sgxContext* sgx_ctx, float *in, float *out, int lyr, int lyr_pooling, int b_beg, int b_end) {
        if (!out || !in) {
            return ERROR_UNEXPECTED;
        }

        int batchsize = sgx_ctx->batchsize;
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
	int* max_indx_ptr = sgx_ctx->max_index.at(lyr_pooling);
        int beg = b_beg * n_chnls * Hi * Wi;
        int end = b_end * n_chnls * Hi * Wi;
        int size = end - beg;
        Map< Matrix<float, 1, Dynamic > > mat_merge( in_sgx+beg, size );
        Map< Matrix<float, 1, Dynamic > > mat_untrust( in+beg, size );
#ifndef SGX_ONLY
        if ( lyr > 0 ) {
            out_prev_sgx = sgx_ctx->top.at( lyr-1 );
            Map< Matrix<float, 1, Dynamic > > mat_sgx( out_prev_sgx+beg, size );
            mat_merge = mat_untrust + mat_sgx;
        }
        else {
            mat_merge = mat_untrust;
        }
#else
        if ( lyr > 0 ) { 
            out_prev_sgx = sgx_ctx->top.at( lyr-1 );
            Map< Matrix<float, 1, Dynamic > > mat_sgx( out_prev_sgx+beg, size );
            mat_merge = mat_sgx;
        }
        else {
            mat_merge = mat_untrust;
        }
#endif

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

#ifndef SGX_ONLY
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
#else
    if ( lyr != sgx_ctx->config.size()-1 ){
        float* in_next_lyr = sgx_ctx->bottom.at( lyr + 1);
        for ( int b_ = b_beg; b_ < b_end; b_++ ) {
            int sz_out = n_chnls * Ho * Wo;
            float* in_ptr = in_next_lyr + b_ * sz_out;
            float* out_ptr = out + b_ * sz_out;
            for( int i = 0; i < sz_out; i++ ) {
               *( in_ptr + i )  = *( out_ptr + i );
               *( out_ptr + i ) = 0.0f;
            }
        }
    }

#endif

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
        Map< Matrix< float, Dynamic, Dynamic, RowMajor > > X( in, u_len, v_len );
        for ( int r_=0; r_<r; r_++ ) {
            float* u = u_T + ( r_ * u_len );
            float* v = v_T + ( r_ * v_len );
            Map< Matrix< float, Dynamic, 1 > > u_vec( u, u_len );
            Map< Matrix< float, Dynamic, 1 > > v_vec( v, v_len );
            //Alternating optimization
            for ( int iter=0; iter<max_iter; iter++ ) {
                // compute u
                float v_norm = v_vec.squaredNorm(); 
                u_vec = X * v_vec / ( v_norm + eps );

                // computer v
                float u_norm = u_vec.squaredNorm();
                v_vec = X.transpose() * u_vec / ( u_norm + eps );
            }

            // remove r-th principle channel from input
            X -= ( u_vec * v_vec.transpose() );
        }
        /*Map< Matrix< float, Dynamic, Dynamic > > u_vec( u_T, u_len, r );
        Map< Matrix< float, Dynamic, Dynamic > > v_vec( v_T, v_len, r );
        for ( int iter = 0; iter < max_iter; iter++ ) {
            Matrix< float, Dynamic, Dynamic > Xv = X * v_vec;
            Xv.eval();
            for ( int r_ = 0; r_ < r; r_++ ) {
                float v_norm = v_vec.col( r_ ).squaredNorm() + eps;
                u_vec.col( r_ ) = Xv.col( r_ ) / v_norm;
            }
            Matrix< float, Dynamic, Dynamic > Xu = X.transpose() * u_vec;
            Xu.eval();
            for ( int r_ = 0; r_ < r; r_++ ) {
                float u_norm = u_vec.col( r_ ).squaredNorm() + eps;
                v_vec.col( r_ ) = Xu.col( r_ ) / u_norm;
            }
        }
        X -= ( u_vec * v_vec.transpose() );*/
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
