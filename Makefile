SRC_DIR := src/
INCLUDE_DIRS := $(SRC_DIR) \
	$(SRC_DIR)/kernels \
	$(SRC_DIR)/kernels/activation_funcs \
PPFLAGS := -DUSE_PWL_ACTIVATION_FNS=1
LFLAGS := -lm
WORK_DIR := build
REPORT_DIR := reports
LOG_DIR := logs
PLATFORM := $(XILINX_VITIS)/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm

SOURCES := $(wildcard $(SRC_DIR)/*.cpp) \
	   $(wildcard $(SRC_DIR)/*.hpp) \
	   $(wildcard $(SRC_DIR)/kernels/*.cpp) \
	   $(wildcard $(SRC_DIR)/kernels/*.hpp) \
	   $(wildcard $(SRC_DIR)/kernels/activation_funcs/*.cpp) \
	   $(wildcard $(SRC_DIR)/kernels/activation_funcs/*.hpp) \

AIECC := v++ \
      --compile \
      --mode aie \
      --aie.debug \
      --aie.verbose \
      --aie.adf-api-log-level=1 \
      --aie.log-level=1 \
      --platform $(PLATFORM) \
      --include $(XILINX_VITIS)/aietools/include \
      $(foreach x,$(INCLUDE_DIRS),--include $x) \
      $(foreach x,$(PPFLAGS),--aie.Xpreproc=$x) \
      $(foreach x,$(LFLAGS),--aie.Xpslinker=$x) \
      --report_dir $(REPORT_DIR) \
      --work_dir $(WORK_DIR) \
      --log_dir $(LOG_DIR)

.PHONY: build-hw \
	build-x86sim \
	build-aiesim \
	run-x86sim \
	run-aiesim \
	clean

build-hw: $(SOURCES)
	$(AIECC) --target hw src/main.cpp

build-x86sim: $(SOURCES)
	$(AIECC) --target x86sim src/main.cpp

run-x86sim:
	x86simulator --pkg-dir=$(WORK_DIR) --trace

run-aiesim:
	aiesimulator --pkg-dir=$(WORK_DIR)

clean:
	rm -r $(WORK_DIR)/*
	rm -r $(REPORT_DIR)/*
	rm -r $(LOG_DIR)/*
