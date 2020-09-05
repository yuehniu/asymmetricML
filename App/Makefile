include ../buildenv.mk

TARGET = enclave_bridge.so

RM = rm -f

ifneq ($(SGX_MODE), HW)
	URTS_LIB_NAME := sgx_urts_sim
    UAE_SERVICE_LIB := sgx_uae_service_sim
else
	URTS_LIB_NAME := sgx_urts
    UAE_SERVICE_LIB := sgx_uae_service
endif

INC := -I$(SGX_SDK)/include -I./ -I../Include
LIB := -l$(URTS_LIB_NAME) -l$(UAE_SERVICE_LIB) -L$(SGX_SDK)/lib64 -lpthread
CXXFLAGS += $(INC) $(LIB) -fPIC
CFLAGS += $(INC) $(LIB)

ifeq ($(SGX_DEBUG), 1)
	CXXFLAGS += -DDEBUG -UNDEBUG -UEDEBUG
	CFLAGS += -DDEBUG -UNDEBUG -UEDEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	CXXFLAGS += -DEDEBUG -DNDEBUG -UDEBUG
	CFLAGS += -DEDEBUG -DNDEBUG -UDEBUG
else
	CXXFLAGS += -DNDEBUG -UEDEBUG -UDEBUG
	CFLAGS += -DNDEBUG -UEDEBUG -UDEBUG
endif

SRC_CPP = $(wildcard *.cpp)
SRC_C = $(wildcard *.c)

SRC_OBJ += $(SRC_CPP:.cpp=.o)
SRC_OBJ += $(SRC_C:.c=.o)

.PHONY = all clean

all: $(TARGET)

SGX_INC := -I$(SGX_SDK)/include

enclave_bridge.o: enclave_bridge.cpp utils.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "CXX <= $<"

utils.o: utils.cpp
	@$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "CXX <= $<"

enclave_bridge_u.o: enclave_bridge_u.c
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "CC <= $<"

enclave_bridge_u.c:
	@$(SGX_EDGER8R) --untrusted ../Enclave/enclave.edl --search-path $(SGX_SDK)/include

enclave_u.c:
	@$(SGX_EDGER8R) --untrusted ../Enclave/enclave.edl --search-path $(SGX_SDK)/include

$(TARGET):enclave_u.o $(SRC_OBJ)
	@$(CXX) $^ $(CXXFLAGS) -shared -o $@
	@mv $@ $(TOPDIR)/lib/
	@echo "GEN => $@"

clean:
	@$(RM) $(TARGET) *.o *_u.c *_u.h
