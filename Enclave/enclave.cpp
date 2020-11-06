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

    uint32_t add_Conv_ctx_enclave(int n_ichnls, int n_ochnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int r) {
        char message[] = "[SGX:Trusted] ADD Conv context\n";
        printf(message);
        return sgx_add_Conv_ctx(&sgx_ctx, n_ichnls, n_ochnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, r);
    }

    uint32_t Conv_fwd_enclave(float *w, int lyr, int b_beg, int b_end) {
        uint32_t status = sgx_Conv_fwd( &sgx_ctx, w, lyr, b_beg, b_end );

        //char message[] = "[SGX:Trusted] Call Conv FWD\n";
        //printf(message);
        return status;
    }
    
    uint32_t Conv_bwd_enclave( float* gradout, float* gradw, int lyr, int c_beg, int c_end ) {
        uint32_t status = sgx_Conv_bwd( &sgx_ctx, gradout, gradw, lyr, c_beg, c_end );

        return status;
    }

    uint32_t add_ReLU_ctx_enclave(int n_chnls, int H, int W) {
        char message[] = "[SGX:Trusted] ADD ReLU context\n";
        printf(message);
        return sgx_add_ReLU_ctx(&sgx_ctx, n_chnls, H, W);
    }

    uint32_t ReLU_fwd_enclave(float* out, int lyr, int b_beg, int b_end) {
        //char message[] = "[SGX:Trusted] Call ReLU FWD\n";
        //printf(message);
        uint32_t status = sgx_ReLU_fwd(&sgx_ctx, out, lyr, b_beg, b_end);

        return status;
    }

    uint32_t ReLU_bwd_enclave(float* gradin, int lyr, int b_beg, int b_end) {
        return sgx_ReLU_bwd(&sgx_ctx, gradin, lyr, b_beg, b_end);
    }

    uint32_t add_ReLUPooling_ctx_enclave(int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        char message[] = "[SGX:Trusted] ADD ReLUPooling context\n";
        printf(message);
        return sgx_add_ReLUPooling_ctx(&sgx_ctx, n_chnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, mode);
    }

    uint32_t ReLUPooling_fwd_enclave(float* in, float* out, int lyr, int lyr_pooling, int b_beg, int b_end ) {
        //char message[] = "[SGX:Trusted] Call ReLUPooling FWD\n";
        //printf(message);
        return sgx_ReLUPooling_fwd(&sgx_ctx, in, out, lyr, lyr_pooling, b_beg, b_end);
    }

    uint32_t ReLUPooling_bwd_enclave(float* gradout, float* gradin, int lyr, int lyr_pooling, int b_beg, int b_end) {
        return sgx_ReLUPooling_bwd(&sgx_ctx, gradout, gradin, lyr, lyr_pooling, b_beg, b_end);
    }

    void test_light_SVD_enclave(float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter)
    {
        sgx_light_SVD(in, u_T, u_len, v_T, v_len, r, max_iter);
    }

    void test_Conv_fwd_enclave(float* in, float* w, float* out, int b_beg, int b_end ) {
        float* in_sgx = sgx_ctx.bottom.at(0);
        //int sz_in = sgx_ctx.sz_bottom.at(0);
        lyrConfig* lyr_conf = sgx_ctx.config.at( 0 );
        int r = lyr_conf->conv_conf->r;
        int n_ichnls = lyr_conf->conv_conf->n_ichnls;
        int Hi = lyr_conf->conv_conf->Hi; int Wi = lyr_conf->conv_conf->Wi;
#ifndef SGX_ONLY
        int beg = b_beg * r * ( n_ichnls + Hi*Wi );
        int end = b_end * r * ( n_ichnls + Hi*Wi );
#else
        int beg = b_beg * n_ichnls * Hi * Wi;
        int end = b_end * n_ichnls * Hi * Wi;
#endif
        for( int i = beg; i < end; i++ ) {
            *( in_sgx+i ) = *( in+i );
        }

        sgx_Conv_fwd( &sgx_ctx, w, 0, b_beg, b_end );

        float* out_sgx = sgx_ctx.top.at( 0 );
        int sz_out = sgx_ctx.sz_top.at( 0 );
        int n_ochnls = lyr_conf->conv_conf->n_ochnls;
        int Ho = lyr_conf->conv_conf->Ho; int Wo = lyr_conf->conv_conf->Wo;
        beg = b_beg * n_ochnls * Ho * Wo; end = b_end * n_ochnls * Ho * Wo;
        for( int i = beg; i < end; i++ ) {
            *( out+i ) = *( out_sgx+i );
        }
    }

    void test_Conv_bwd_enclave(float* gradout, float* gradw, int c_beg, int c_end ) {
        int lyr = 0;

        sgx_Conv_bwd( &sgx_ctx, gradout, gradw, lyr, c_beg, c_end );
    }

    void test_ReLU_fwd_enclave(float *in_sgx, float* in_gpu, float* u_T, float* v_T, int b_beg, int b_end) {
        //-> init in_sgx
        int sz_top_prev = sgx_ctx.sz_top.at( 0 );
        float* top_prev = sgx_ctx.top.at( 0 );
        lyrConfig*  lyr_conf = sgx_ctx.config.at( 0 );
        int n_chnls = lyr_conf->conv_conf->n_ochnls;
        int H = lyr_conf->conv_conf->Ho; int W = lyr_conf->conv_conf->Wo;
        int beg = b_beg * n_chnls * H * W;
        int end = b_end * n_chnls * H * W;
        for ( int i = beg; i < end; i++ ) {
            *( top_prev + i ) = *( in_sgx + i );
        }

        //-> ReLU FWD
        sgx_ReLU_fwd( &sgx_ctx, in_gpu, 1, b_beg, b_end);

        lyr_conf = sgx_ctx.config.at( 1 );
        int r = lyr_conf->relu_conf->r;
        int sz_u_T = sgx_ctx.batchsize * r * n_chnls;
        float* u_sgx = sgx_ctx.bottom.at( 2 );
        float* v_sgx = sgx_ctx.bottom.at( 2 ) + (sz_u_T);
        beg = b_beg * r * n_chnls;
        end = b_end * r * n_chnls;
        for ( int i = beg; i < end; i++ ) {
            *( u_T+i ) = *( u_sgx + i ); 
        }
        beg = b_beg * r * H * W;
        end = b_end * r * H * W;
        for ( int i = beg; i < end; i++ ) {
            *( v_T+i ) = *( v_sgx+i );
        }
    }

    void test_ReLUPooling_fwd_enclave(float *in_sgx, float* in_gpu, float* out, float* u_T, float* v_T, int b_beg, int b_end) {
        //-> init in_sgx
        int sz_top_prev = sgx_ctx.sz_top.at( 0 );
        float* top_prev = sgx_ctx.top.at( 0 );
        lyrConfig* lyr_conf = sgx_ctx.config.at( 1 );
        int n_chnls = lyr_conf->relupooling_conf->n_chnls;
        int Hi = lyr_conf->relupooling_conf->Hi; int Wi = lyr_conf->relupooling_conf->Wi;
        int beg = b_beg * n_chnls * Hi * Wi;
        int end = b_end * n_chnls * Hi * Wi;
        for ( int i = beg; i < end; i++ ) {
            *( top_prev + i ) = *( in_sgx + i );
        }

        //-> ReLUPooling FWD
        sgx_ReLUPooling_fwd( &sgx_ctx, in_gpu, out, 1, 0, b_beg, b_end);

        //-> Extract u_T and v_T
        int b_stride = b_end - b_beg;
        int r = lyr_conf->relupooling_conf->r;
        int sz_u_T = sgx_ctx.batchsize * r * n_chnls;
        float* u_sgx = sgx_ctx.bottom.at( 2 );
        float* v_sgx = u_sgx + ( sz_u_T );
        beg = b_beg * r * n_chnls;
        end = b_end * r * n_chnls;
        for ( int i = beg; i < end; i++ ) {
            *(u_T + i) = *( u_sgx + i );
        }
        int Ho = lyr_conf->relupooling_conf->Ho; int Wo = lyr_conf->relupooling_conf->Wo;
        beg = b_beg * r * Ho * Wo;
        end = b_end * r * Ho * Wo;
        for ( int i = beg; i < end; i++ ) {
            *( v_T + i ) = *( v_sgx + i );
        }
    }
}
