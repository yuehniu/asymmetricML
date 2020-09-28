#ifndef ERROR_CODES_H_
#define ERROR_CODES_H_

typedef uint32_t ATTESTATION_STATUS;

#define SUCCESS                          0x00
#define INVALID_PARAMETER                0xE1
#define VALID_SESSION                    0xE2
#define INVALID_SESSION                  0xE3
#define ATTESTATION_ERROR                0xE4
#define ATTESTATION_SE_ERROR             0xE5
#define IPP_ERROR                        0xE6
#define NO_AVAILABLE_SESSION_ERROR       0xE7
#define MALLOC_ERROR                     0xE8
#define ERROR_TAG_MISMATCH               0xE9
#define OUT_BUFFER_LENGTH_ERROR          0xEA
#define INVALID_REQUEST_TYPE_ERROR       0xEB
#define INVALID_PARAMETER_ERROR          0xEC
#define ENCLAVE_TRUST_ERROR              0xED
#define ENCRYPT_DECRYPT_ERROR            0xEE
#define DUPLICATE_SESSION                0xEF
#define ERROR_OUT_OF_MEMORY              0xF0
#define ERROR_UNEXPECTED		 0xF1
#endif
