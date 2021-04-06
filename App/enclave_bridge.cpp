#include "sgx_urts.h"
#include "enclave_u.h"
#include "utils.h"
#include "datatype.h"

#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>
#include <chrono>

#define ENCLAVE_FILE_NAME "lib/libenclave.signed.so"
#define N_THREADS 32

using namespace std::chrono;

struct sgxContext_public {
    int batchsize;
    std::vector<int> n_ichnls = {};
    std::vector<int> n_ochnls = {};
}; 

sgxContext_public sgx_ctx_pub;

extern "C"
{
    unsigned long int init_ctx_bridge(int n_lyrs, BOOL use_sgx, int batchsize, BOOL verbose) {
        std::cout << "[Public] Initializing Enclave..." << std::endl; 
        sgx_enclave_id_t eid = 0;
        sgx_launch_token_t token = {0};
        sgx_status_t ret = SGX_ERROR_UNEXPECTED;
        int updated = 0;

        ret = sgx_create_enclave(ENCLAVE_FILE_NAME, SGX_DEBUG_FLAG, &token, &updated, &eid, NULL);
        if (ret != SGX_SUCCESS) {
            print_error_message(ret);
            throw ret;
        }
        std::cout << "[Public] Enclave id: " << eid << std::endl;

        std::cout << "[Public] Initializing SGX running context..." << std::endl; 
        uint32_t status = 0;
        init_enclave_ctx(eid, &status, n_lyrs, use_sgx, batchsize, verbose);

        sgx_ctx_pub.batchsize = batchsize;

        return eid;
    }

    void destroy_ctx_bridge(sgx_enclave_id_t eid) {
        sgx_destroy_enclave(eid);
    }

    uint32_t set_lyrs_bridge(sgx_enclave_id_t eid, int n_lyrs) {
        uint32_t status = 0;
        set_lyrs_enclave(eid, &status, n_lyrs);

        return status;
    } 

    uint32_t set_batchsize_bridge(sgx_enclave_id_t eid, int batchsize) {
        uint32_t status = 0;
        set_batchsize_enclave(eid, &status, batchsize);

        return status;
    }

    uint32_t set_sgx_bridge(sgx_enclave_id_t eid, BOOL use_sgx) {
        uint32_t status = 0;
        set_sgx_enclave(eid, &status, use_sgx);

        return status;
    }

    uint32_t set_verbose_bridge(sgx_enclave_id_t eid, BOOL verbose) {
        uint32_t status = 0;
        set_verbose_enclave(eid, &status, verbose);

        return status;
    }

    uint32_t add_Conv_ctx_bridge(sgx_enclave_id_t eid, int n_ichnls, int n_ochnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int r) {
        uint32_t status = 0;
        add_Conv_ctx_enclave(eid, &status, n_ichnls, n_ochnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, r);

        sgx_ctx_pub.n_ichnls.push_back( n_ichnls );
        sgx_ctx_pub.n_ochnls.push_back( n_ochnls );

        return status;
    }

    uint32_t Conv_fwd_bridge(sgx_enclave_id_t eid, float *w, int lyr, BOOL shortcut, int *t ) {
        //std::cout << "[DEBUG-SGX-Bridge::Conv::FWD] " << *w << std::endl;

        auto start = high_resolution_clock::now();

        uint32_t status[ N_THREADS ] = {0};
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ){
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( Conv_fwd_enclave, eid, status+i, w, lyr, b_beg, b_end, shortcut );
        }
        for( int i = 0; i < N_THREADS; i++ ){
            trd[ i ].join();
        }

        //Conv_fwd_enclave(eid, &status, w, lyr);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();

        return status[ 0 ];
    }

    uint32_t Conv_bwd_bridge( sgx_enclave_id_t eid, float* gradout, float* gradw, int lyr, int *t ) {
        auto start = high_resolution_clock::now();

        uint32_t status[ N_THREADS ] = {0};
        int n_ochnls = sgx_ctx_pub.n_ochnls.at( lyr );
        int batchsize = sgx_ctx_pub.batchsize;
        int c_stride = n_ochnls / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ){
            int c_beg = i * c_stride;
            int c_end = c_beg + c_stride;
            trd[ i ] = std::thread( Conv_bwd_enclave, eid, status+i, gradout, gradw, lyr, c_beg, c_end );
        }
        for( int i =0; i < N_THREADS; i++ ){
            trd[ i ].join();
        }

        //Conv_bwd_enclave( eid, &status, gradout, gradw, lyr );

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();

        return status[ 0 ];
    }

    uint32_t add_ShortCut_ctx_bridge( sgx_enclave_id_t eid, int n_chnls, int H, int W ) {
        uint32_t status = 0;
        add_ShortCut_ctx_enclave( eid, &status, n_chnls, H, W );

        sgx_ctx_pub.n_ichnls.push_back( n_chnls );
        sgx_ctx_pub.n_ochnls.push_back( n_chnls );

        return status;
    }

    uint32_t ShortCut_fwd_bridge( sgx_enclave_id_t eid, int lyr ) {
        uint32_t status[ N_THREADS ] = { 0 };
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( ShortCut_fwd_enclave, eid, status+i, lyr, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }

        return status[ 0 ];
    }


    uint32_t add_ReLU_ctx_bridge(sgx_enclave_id_t eid, int n_chnls, int H, int W) {
        uint32_t status = 0;
        add_ReLU_ctx_enclave(eid, &status, n_chnls, H, W);

        sgx_ctx_pub.n_ichnls.push_back( n_chnls );
        sgx_ctx_pub.n_ochnls.push_back( n_chnls );

        return status;
    }

    uint32_t ReLU_fwd_bridge(sgx_enclave_id_t eid, float* out, int lyr, int *t) {
        auto start = high_resolution_clock::now();

        uint32_t status[N_THREADS] = {0};
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[N_THREADS];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( ReLU_fwd_enclave, eid, status+i, out, lyr, b_beg, b_end);
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //ReLU_fwd_enclave(eid, status, out, lyr, 0, batchsize);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();
        
        return status[ 0 ];
    }

    uint32_t ReLU_bwd_bridge(sgx_enclave_id_t eid, float* gradin, int lyr, int *t) {
        auto start = high_resolution_clock::now();

        uint32_t status[N_THREADS] = {0};
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( ReLU_bwd_enclave, eid, status+i, gradin, lyr, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //ReLU_bwd_enclave(eid, status, gradin, lyr, 0, batchsize);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();

        return status[ 0 ];
    }

    uint32_t add_ReLUPooling_ctx_bridge(sgx_enclave_id_t eid, int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        uint32_t status = 0;
        add_ReLUPooling_ctx_enclave(eid, &status, n_chnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, mode);

        sgx_ctx_pub.n_ichnls.push_back( n_chnls );
        sgx_ctx_pub.n_ochnls.push_back( n_chnls );

        return status;
    }

    uint32_t ReLUPooling_fwd_bridge(sgx_enclave_id_t eid, float* in, float* out, int lyr, int lyr_pooling, int *t) {
        auto start = high_resolution_clock::now();

        uint32_t status[N_THREADS] = {0};
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread(ReLUPooling_fwd_enclave, eid, status+i, in, out, lyr, lyr_pooling, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //ReLUPooling_fwd_enclave(eid, &status, in, out, lyr, lyr_pooling);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();
        
        return status[ 0 ];
    }

    uint32_t ReLUPooling_bwd_bridge(sgx_enclave_id_t eid, float* gradout, float* gradin, int lyr, int lyr_pooling, int *t) {
        auto start = high_resolution_clock::now();

        //std::cout << "[DEBUG-SGX-Bridge::ReLUPooling::BWD] " << *gradout << std::endl;
        uint32_t status[N_THREADS] = {0};
        int batchsize = sgx_ctx_pub.batchsize;
        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread(ReLUPooling_bwd_enclave, eid, status+i, gradout, gradin, lyr, lyr_pooling, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //ReLUPooling_bwd_enclave(eid, &status, gradout, gradin, lyr, lyr_pooling);

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        *t = duration.count();

        return status[ 0 ];
    }

    void test_light_SVD_bridge(sgx_enclave_id_t eid, float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter)
    {
        test_light_SVD_enclave(eid, in, u_T, u_len, v_T, v_len, r, max_iter);
    }

    void test_Conv_fwd_bridge( sgx_enclave_id_t eid, float* in, float* w, float* out,
                               int batchsize, int n_ichnls, int n_ochnls, int Hi, int Wi, int Ho, int Wo, int r ){
        //->Construct context
        uint32_t status;
        set_batchsize_enclave( eid, &status, batchsize );
        set_sgx_enclave( eid, &status, 1 );
        add_Conv_ctx_enclave( eid, &status, n_ichnls, n_ochnls, 3, 1, 1, Hi, Wi, Ho, Wo, r );
        sgx_ctx_pub.n_ichnls.push_back( n_ichnls );
        sgx_ctx_pub.n_ochnls.push_back( n_ochnls );
        sgx_ctx_pub.batchsize = batchsize;

        int b_stride = batchsize / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( test_Conv_fwd_enclave, eid, in, w, out, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ){
            trd[ i ].join();
        }
        //test_Conv_fwd_enclave( eid, in, w, out );
    }

    void test_Conv_bwd_bridge( sgx_enclave_id_t eid, float* gradout, float* gradw) {
        int n_ochnls = sgx_ctx_pub.n_ochnls.at( 0 );
        int batchsize = sgx_ctx_pub.batchsize;
        int c_stride = n_ochnls / N_THREADS;
        std::thread trd[ N_THREADS ];
        for( int i = 0; i < N_THREADS; i++ ){
            int c_beg = i * c_stride;
            int c_end = c_beg + c_stride;
            trd[ i ] = std::thread( test_Conv_bwd_enclave, eid, gradout, gradw, c_beg, c_end );
        }
        for( int i = 0; i < N_THREADS; i++ ){
            trd[ i ].join();
        }
        //test_Conv_bwd_enclave( eid, gradout, gradw );
    }

    void test_ReLU_fwd_thread( sgx_enclave_id_t eid, float* in_sgx, float* in_gpu, float* u_T, float* v_T, int b_beg, int b_end ) {
        size_t thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        std::cout << "thread id: " << thread_id << std::endl;

        test_ReLU_fwd_enclave( eid, in_sgx, in_gpu, u_T, v_T, b_beg, b_end );
    }
    void test_ReLU_fwd_bridge( sgx_enclave_id_t eid, float* in_sgx, float* in_gpu, float* u_T, float* v_T,
                               int batchsize, int n_chnls, int H, int W, int lyr, int r){
        //->Construct context
        uint32_t status;
        set_batchsize_enclave( eid, &status, batchsize );
        set_sgx_enclave( eid, &status, 1 );
        add_Conv_ctx_enclave( eid, &status, n_chnls, n_chnls, 3, 1, 1, H, W, H, W, r);
        add_ReLU_ctx_enclave( eid, &status, n_chnls, H, W);
        add_Conv_ctx_enclave( eid, &status, n_chnls, n_chnls, 3, 1, 1, H, W, H, W, r);

        int b_stride = batchsize / N_THREADS;
        std::thread trd[N_THREADS];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( test_ReLU_fwd_thread, eid, in_sgx, in_gpu, u_T, v_T, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //test_ReLU_fwd_enclave( eid, in_sgx, in_gpu, u_T, v_T, batchsize, n_chnls, H, W, lyr, r);
    }

    void test_ReLUPooling_fwd_bridge( sgx_enclave_id_t eid, float* in_sgx, float* in_gpu, float* out, float* u_T, float* v_T,
                                      int batchsize, int n_chnls, int Hi, int Wi, int Ho, int Wo, int lyr, int r ){
        //->Construct context
        uint32_t status;
        set_batchsize_enclave( eid, &status, batchsize );
        set_sgx_enclave( eid, &status, 1 );
        add_Conv_ctx_enclave( eid, &status, n_chnls, n_chnls, 3, 1, 1, Hi, Wi, Hi, Wi, r );
        add_ReLUPooling_ctx_enclave( eid, &status, n_chnls, 2, 2, 0, Hi, Wi, Ho, Wo, 0);
        add_Conv_ctx_enclave( eid, &status, n_chnls, n_chnls, 3, 1, 1, Ho, Wo, Ho, Wo, r );
        int b_stride = batchsize / N_THREADS;
        std::thread trd[N_THREADS];
        for( int i = 0; i < N_THREADS; i++ ) {
            int b_beg = i * b_stride;
            int b_end = b_beg + b_stride;
            trd[ i ] = std::thread( test_ReLUPooling_fwd_enclave, eid, in_sgx, in_gpu, out, u_T, v_T, b_beg, b_end );
        }
        for( int i = 0; i < N_THREADS; i++ ) {
            trd[ i ].join();
        }
        //test_ReLUPooling_fwd_enclave( eid, in_sgx, in_gpu, out, u_T, v_T, batchsize, n_chnls, Hi, Wi, Ho, Wo, lyr, r );
    }

    void ocall_print_string(const char *str) {
        printf("%s", str);
    }
}
