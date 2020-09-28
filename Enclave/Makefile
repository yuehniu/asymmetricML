include ../buildenv.mk

ifneq ($(SGX_MODE), HW)
		Trts_Library_Name := sgx_trts_sim
        Service_Library_Name := sgx_tservice_sim
else
		Trts_Library_Name := sgx_trts
        Service_Library_Name := sgx_tservice
endif
Crypto_Library_Name := sgx_tcrypto

ENCLAVE_NAME := libenclave.so
SIGNED_ENCLAVE_NAME := libenclave.signed.so

$(SIGNED_ENCLAVE_NAME): $(ENCLAVE_NAME)
	@$(SGX_ENCLAVE_SIGNER) sign -key enclave_private_test.pem -enclave $(ENCLAVE_NAME) -out $@ -config enclave.config.xml
	@cp $(SIGNED_ENCLAVE_NAME) $(TOPDIR)/lib/
	@echo "SIGN => $@"

Enclave_Security_Link_Flags := -Wl,-z,relro,-z,now,-z,noexecstack

ENCLAVE_LINK_FLAGS := $(Enclave_Security_Link_Flags) \
     -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
             -Wl,--whole-archive -l$(Trts_Library_Name) -Wl,--no-whole-archive \
             -Wl,--start-group -lsgx_tstdc -lsgx_tcxx -l$(Crypto_Library_Name) -l$(Service_Library_Name) -Wl,--end-group \
             -Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
             -Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
             -Wl,--defsym,__ImageBase=0 -Wl,--gc-sections 

SGX_COMMON_FLAGS += -Wall -Wextra -Winit-self -Wpointer-arith -Wreturn-type \
                    -Waddress -Wsequence-point -Wformat-security \
                    -Wmissing-include-dirs -Wfloat-equal -Wundef -Wshadow \
                    -Wcast-align -Wconversion -Wredundant-decls 

SGX_COMMON_CFLAGS := $(SGX_COMMON_FLAGS) -Wjump-misses-init -Wstrict-prototypes -Wunsuffixed-float-constants
SGX_COMMON_CXXFLAGS := $(SGX_COMMON_FLAGS) -Wnon-virtual-dtor -std=c++11

Enclave_Include_Paths := -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -I../Include -I/usr/lib/gcc/x86_64-linux-gnu/7.5.0/include/ -I../Include/eigen3_sgx/

Enclave_C_Flags := $(Enclave_Include_Paths) -nostdinc -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -march=native
APP_C_Flags := -fPIC -Wno-attributes -IApp -IInclude -I$(SGX_SDJ)/include
# Enclave_C_Flags := $(Enclave_Include_Paths) -fvisibility=hidden -fpie -ffunction-sections -fdata-sections -march=native
CC_BELOW_4_9 := $(shell expr "`$(CC) -dumpversion`" \< "4.9")
ifeq ($(CC_BELOW_4_9), 1)
        Enclave_C_Flags += -fstack-protector
else
        Enclave_C_Flags += -fstack-protector-strong
endif
Enclave_Cpp_Flags := $(Enclave_C_Flags) -nostdinc++
# Enclave_Cpp_Flags := $(Enclave_C_Flags)

ifeq ($(LAv2), 1)
	Enclave_C_Flags += -DSGX_USE_LAv2
	Enclave_Cpp_Flags += -DSGX_USE_LAv2
endif

Enclave_Cpp_Files := $(wildcard *.cpp)
ENCLAVE_CPP_OBJECTS := $(Enclave_Cpp_Files:.cpp=.o)

.PHONY = all clean target

target = $(SIGNED_ENCLAVE_NAME)

all:
	@make target

clean:
	@rm -f  $(ENCLAVE_NAME) $(SIGNED_ENCLAVE_NAME) *.o *_t.c *_t.h *_u.c *_u.h

$(ENCLAVE_NAME): enclave_t.o $(ENCLAVE_CPP_OBJECTS)
	@$(CXX) $^ -o $@ $(ENCLAVE_LINK_FLAGS) -pthread -fopenmp
	@echo "LINK => $@"

enclave_u.h: $(SGX_EDGER8R) enclave.edl
	@$(SGX_EDGER8R) --trusted enclave.edl --search-path $(SGX_SDK)/include

enclave_u.c: enclave_u.h

enclave_u.o: enclave_u.c dnn_sgx.o
	@$(CC) $(SGX_COMMON_CFLAGS) $(APP_C_Flags) -c $< -o $@
	@echo "CC <= $<"

enclave_t.h: $(SGX_EDGER8R) enclave.edl
	@$(SGX_EDGER8R) --trusted enclave.edl --search-path $(SGX_SDK)/include

enclave_t.c: enclave_t.h

enclave_t.o: enclave_t.c dnn_sgx.o
	@$(CC) $(SGX_COMMON_CFLAGS) $(Enclave_C_Flags) -c $< -o $@
	@echo "CC <= $<"

%.o: %.cpp enclave_t.h
	@$(CXX) $(SGX_COMMON_CXXFLAGS) $(Enclave_Cpp_Flags) -c $< -o $@
	@echo "CXX <= $<"
