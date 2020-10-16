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

    uint32_t Conv_fwd_enclave(float *w, int lyr) {
        uint32_t status = sgx_Conv_fwd( &sgx_ctx, w, lyr );

        //char message[] = "[SGX:Trusted] Call Conv FWD\n";
        //printf(message);
        return status;
    }
    
    uint32_t Conv_bwd_enclave( float* gradout, float* gradw, int lyr ) {
        uint32_t status = sgx_Conv_bwd( &sgx_ctx, gradout, gradw, lyr );

        return status;
    }

    uint32_t add_ReLU_ctx_enclave(int n_chnls, int H, int W) {
        char message[] = "[SGX:Trusted] ADD ReLU context\n";
        printf(message);
        return sgx_add_ReLU_ctx(&sgx_ctx, n_chnls, H, W);
    }

    uint32_t ReLU_fwd_enclave(float* out, int lyr) {
        //char message[] = "[SGX:Trusted] Call ReLU FWD\n";
        //printf(message);
        uint32_t status = sgx_ReLU_fwd(&sgx_ctx, out, lyr);

        return status;
    }

    uint32_t ReLU_bwd_enclave(float* gradin, int lyr) {
        return sgx_ReLU_bwd(&sgx_ctx, gradin, lyr);
    }

    uint32_t add_ReLUPooling_ctx_enclave(int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        char message[] = "[SGX:Trusted] ADD ReLUPooling context\n";
        printf(message);
        return sgx_add_ReLUPooling_ctx(&sgx_ctx, n_chnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, mode);
    }

    uint32_t ReLUPooling_fwd_enclave(float* in, float* out, int lyr, int lyr_pooling) {
        //char message[] = "[SGX:Trusted] Call ReLUPooling FWD\n";
        //printf(message);
        return sgx_ReLUPooling_fwd(&sgx_ctx, in, out, lyr, lyr_pooling);
    }

    uint32_t ReLUPooling_bwd_enclave(float* gradout, float* gradin, int lyr, int lyr_pooling) {
        return sgx_ReLUPooling_bwd(&sgx_ctx, gradout, gradin, lyr, lyr_pooling);
    }

    void test_light_SVD_enclave(float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter)
    {
        sgx_light_SVD(in, u_T, u_len, v_T, v_len, r, max_iter);
    }

    void test_Conv_fwd_enclave(float* in, float* w, float* out) {
        // init sgx context
        sgx_ctx.batchsize = 8;
        sgx_ctx.useSGX = 1;
        sgx_ctx.verbose = 1;
        int n_ichnls = 16;
        int n_ochnls = 16;
        int sz_kern = 3;
        int stride = 1;
        int padding = 1;
        int Hi = 32; int Wi = 32;
        int Ho = 32; int Wo = 32;
        int r = 4;

        sgx_add_Conv_ctx(&sgx_ctx, n_ichnls, n_ochnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, r);

        float* in_sgx = sgx_ctx.bottom.at(0);
        int sz_in = sgx_ctx.sz_bottom.at(0);
        for( int i = 0; i < sz_in; i++ ) {
            *( in_sgx+i ) = *( in+i );
        }

        sgx_Conv_fwd( &sgx_ctx, w, 0 );

        float* out_sgx = sgx_ctx.top.at( 0 );
        int sz_out = sgx_ctx.sz_top.at( 0 );
        for( int i = 0; i < sz_out; i++ ) {
            *( out+i ) = *( out_sgx+i );
        }
    }

    void test_Conv_bwd_enclave(float* gradout, float* gradw) {
        int lyr = 0;

        sgx_Conv_bwd( &sgx_ctx, gradout, gradw, lyr );
    }

    void test_ReLU_fwd_enclave(float *in_sgx, float* in_gpu, float* u_T, float* v_T, int batchsize, int n_chnls, int H, int W, int lyr, int r) {
        //-> init sgx context
        sgx_ctx.batchsize = batchsize;
        sgx_ctx.useSGX = 1; sgx_ctx.verbose = 1;
        // first add a pseudo conv context
        sgx_add_Conv_ctx( &sgx_ctx, n_chnls, n_chnls, 3, 1, 1, H, W, H, W, r);

        sgx_add_ReLU_ctx( &sgx_ctx, n_chnls, H, W );
        
        // last add a pseudo conv context
        sgx_add_Conv_ctx( &sgx_ctx, n_chnls, n_chnls, 3, 1, 1, H, W, H, W, r);

        //-> init in_sgx
        int sz_top_prev = sgx_ctx.sz_top.at( lyr-1 );
        float* top_prev = sgx_ctx.top.at( lyr-1 );
        for ( int i = 0; i < sz_top_prev; i++ ) {
            *( top_prev + i ) = *( in_sgx + i );
        }

        //-> ReLU FWD
        sgx_ReLU_fwd( &sgx_ctx, in_gpu, lyr);

        int sz_u_T = batchsize * r * n_chnls;
        int sz_v_T = batchsize * r * H * W;
        float* u_sgx = sgx_ctx.bottom.at( lyr+1 );
        float* v_sgx = sgx_ctx.bottom.at( lyr+1 ) + (sz_u_T);
        for ( int i = 0; i < sz_u_T; i++ ) {
            *( u_T+i ) = *( u_sgx + i ); 
        }
        for ( int i = 0; i < sz_v_T; i++ ) {
            *( v_T+i ) = *( v_sgx+i );
        }
    }

    void test_ReLUPooling_fwd_enclave(float *in_sgx, float* in_gpu, float* out, float* u_T, float* v_T, int batchsize, int n_chnls, int Hi, int Wi, int Ho, int Wo, int lyr, int r ) {
        //-> init sgx context
        sgx_ctx.batchsize = batchsize;
        sgx_ctx.useSGX = 1; sgx_ctx.verbose = 1;
        // fist add a pseudo conv context
        sgx_add_Conv_ctx( &sgx_ctx, n_chnls, n_chnls, 3, 1, 1, Hi, Wi, Hi, Wi, r);

        sgx_add_ReLUPooling_ctx( &sgx_ctx, n_chnls, 2, 2, 0, Hi, Wi, Ho, Wo, 0);

        // last add a pseudo conv context
        sgx_add_Conv_ctx( &sgx_ctx, n_chnls, n_chnls, 3, 1, 1, Ho, Wo, Ho, Wo, r);

        //-> init in_sgx
        int sz_top_prev = sgx_ctx.sz_top.at( lyr-1 );
        float* top_prev = sgx_ctx.top.at( lyr-1 );
        for ( int i = 0; i < sz_top_prev; i++ ) {
            *( top_prev + i ) = *( in_sgx + i );
        }

        //-> ReLUPooling FWD
        sgx_ReLUPooling_fwd( &sgx_ctx, in_gpu, out, lyr, 0);

        //-> Extract u_T and v_T
        int sz_u_T = batchsize * r * n_chnls;
        int sz_v_T = batchsize * r * Ho * Wo;
        float* u_sgx = sgx_ctx.bottom.at( lyr+1 );
        float* v_sgx = u_sgx + ( sz_u_T );
        for ( int i = 0; i < sz_u_T; i++ ) {
            *(u_T + i) = *( u_sgx + i );
        }
        for ( int i = 0; i < sz_v_T; i++ ) {
            *( v_T + i ) = *( v_sgx + i );
        }
    }
}
