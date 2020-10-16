#include "sgx_urts.h"
#include "enclave_u.h"
#include "utils.h"
#include "datatype.h"

#include <iostream>
#include <stdio.h>

#define ENCLAVE_FILE_NAME "lib/libenclave.signed.so"

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

        return status;
    }

    uint32_t Conv_fwd_bridge(sgx_enclave_id_t eid, float *w, int lyr ) {
        uint32_t status = 0;

        Conv_fwd_enclave(eid, &status, w, lyr);

        return status;
    }

    uint32_t Conv_bwd_bridge( sgx_enclave_id_t eid, float* gradout, float* gradw, int lyr ) {
        uint32_t status = 0;

        Conv_bwd_enclave( eid, &status, gradout, gradw, lyr );

        return status;
    }


    uint32_t add_ReLU_ctx_bridge(sgx_enclave_id_t eid, int n_chnls, int H, int W) {
        uint32_t status = 0;
        add_ReLU_ctx_enclave(eid, &status, n_chnls, H, W);

        return status;
    }

    uint32_t ReLU_fwd_bridge(sgx_enclave_id_t eid, float* out, int lyr) {
        uint32_t status = 0;
        ReLU_fwd_enclave(eid, &status, out, lyr);
        
        return status;
    }

    uint32_t ReLU_bwd_bridge(sgx_enclave_id_t eid, float* gradin, int lyr) {
        uint32_t status = 0;
        ReLU_bwd_enclave(eid, &status, gradin, lyr);

        return status;
    }

    uint32_t add_ReLUPooling_ctx_bridge(sgx_enclave_id_t eid, int n_chnls, int sz_kern, int stride, int padding, int Hi, int Wi, int Ho, int Wo, int mode) {
        uint32_t status = 0;
        add_ReLUPooling_ctx_enclave(eid, &status, n_chnls, sz_kern, stride, padding, Hi, Wi, Ho, Wo, mode);

        return status;
    }

    uint32_t ReLUPooling_fwd_bridge(sgx_enclave_id_t eid, float* in, float* out, int lyr, int lyr_pooling) {
        uint32_t status = 0;
        ReLUPooling_fwd_enclave(eid, &status, in, out, lyr, lyr_pooling);
        
        return status;
    }

    uint32_t ReLUPooling_bwd_bridge(sgx_enclave_id_t eid, float* gradout, float* gradin, int lyr, int lyr_pooling) {
        uint32_t status = 0;
        ReLUPooling_bwd_enclave(eid, &status, gradout, gradin, lyr, lyr_pooling);

        return status;
    }

    void test_light_SVD_bridge(sgx_enclave_id_t eid, float* in, 
                       float* u_T, int u_len, 
                       float* v_T, int v_len, 
                       int r, int max_iter)
    {
        test_light_SVD_enclave(eid, in, u_T, u_len, v_T, v_len, r, max_iter);
    }

    void test_Conv_fwd_bridge( sgx_enclave_id_t eid, float* in, float* w, float* out) {
        test_Conv_fwd_enclave( eid, in, w, out );
    }

    void test_Conv_bwd_bridge( sgx_enclave_id_t eid, float* gradout, float* gradw) {
        test_Conv_bwd_enclave( eid, gradout, gradw );
    }

    void test_ReLU_fwd_bridge( sgx_enclave_id_t eid, float* in_sgx, float* in_gpu, float* u_T, float* v_T,
                               int batchsize, int n_chnls, int H, int W, int lyr, int r){
        test_ReLU_fwd_enclave( eid, in_sgx, in_gpu, u_T, v_T, batchsize, n_chnls, H, W, lyr, r);
    }

    void test_ReLUPooling_fwd_bridge( sgx_enclave_id_t eid, float* in_sgx, float* in_gpu, float* out, float* u_T, float* v_T,
                                      int batchsize, int n_chnls, int Hi, int Wi, int Ho, int Wo, int lyr, int r ){
        test_ReLUPooling_fwd_enclave( eid, in_sgx, in_gpu, out, u_T, v_T, batchsize, n_chnls, Hi, Wi, Ho, Wo, lyr, r );
    }

    void ocall_print_string(const char *str) {
        printf("%s", str);
    }
}
