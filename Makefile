include buildenv.mk

ifeq ($(SGX_MODE), HW)
ifeq ($(SGX_DEBUG), 1)
	Build_Mode = HW_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
	Build_Mode = HW_PRERELEASE
else
	Build_Mode = HW_RELEASE
endif
endif

ifeq ($(SGX_MODE), SIM)
ifeq ($(SGX_DEBUG), 1)
		Build_Mode = SIM_DEBUG
else ifeq ($(SGX_PRERELEASE), 1)
		Build_Mode = SIM_PRERELEASE
else
		Build_Mode = SIM_RELEASE
endif
endif

SUB_DIR := Enclave App

.PHONY: all clean

all:
	for dir in $(SUB_DIR); do \
		$(MAKE) -C $$dir; \
	done

ifeq ($(Build_Mode), HW_DEBUG)
		@echo "The project has been build in hardware debug mode."
else ifeq ($(Build_Mode), HW_RELEASE)
		@echo "The project has been build in hardware release mode."
else ifeq ($(Build_Mode), HW_PRERELEASE)
		@echo "The project has been build in hardware pre-release mode."
else ifeq ($(Build_Mode), SIM_DEBUG)
		@echo "The project has been build in simulation debug mode."
else ifeq ($(Build_Mode), SIM_RELEASE)
		@echo "The project has been build in simulation release mode."
else ifeq ($(Build_Mode), SIM_PRERELEASE)
		@echo "The project has been build in simulation pre-release mode."
endif

clean:
	for dir in $(SUB_DIR); do \
			$(MAKE) -C $$dir clean; \
	done
	rm -rf lib/*.so
