#include "sgx_urts.h"
#include "enclave_u.h"
#include "utils.h"
#include "datatype.h"

#include <iostream>

#define ENCLAVE_FILE_NAME "lib/libenclave.signed.so"

extern "C"
{
    unsigned long int init_ctx_bridge(int n_lyrs, BOOL use_sgx, BOOL verbose) {
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
        init_enclave_ctx(eid, n_lyrs, use_sgx, verbose);

        return eid;
    }

    void set_lyrs_bridge(sgx_enclave_id_t eid, int n_lyrs) {
        set_lyrs_enclave(eid, n_lyrs);
    } 

    void set_sgx_bridge(sgx_enclave_id_t eid, BOOL use_sgx) {
        set_sgx_enclave(eid, use_sgx);
    }

    void set_verbose_bridge(sgx_enclave_id_t eid, BOOL verbose) {
        set_verbose_enclave(eid, verbose);
    }


    void add_ReLU_ctx_bridge(sgx_enclave_id_t eid, int batchsize, int n_chnls, int H, int W) {
        add_ReLU_ctx_enclave(eid, batchsize, n_chnls, H, W);
    }

    void ReLU_fwd_bridge(sgx_enclave_id_t eid, float* in, float* out, int lyr) {
        ReLU_fwd_enclave(eid, in, out, lyr);
    }

    void ReLU_bwd_bridge(sgx_enclave_id_t eid, float* gradin, int lyr) {
        ReLU_bwd_enclave(eid, gradin, lyr);
    }
}
