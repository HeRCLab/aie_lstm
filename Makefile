SRC_DIR:=src/
INCLUDE_DIRS:=$(SRC_DIR) \
	$(SRC_DIR)/kernels \
	$(SRC_DIR)/kernels/activation_funcs

# 0 : No kernel optimizations.
#
# 1 : Computation of heap size, generation of guidance based on LLVM
#     IR analysis, and insertion of loop pragmas.
#
# 2 : Optimizations of level 1 and in addition loop peeling for
#     unrolled loops and automatic inlining.
OPT_LVL:=0

STACK_SIZE:=4096

# PREPROC_FLAGS:=
# PREPROC_FLAGS:=-DCHESS_CXX_ATTRIBUTES -DUSE_PWL_ACTIVATION_FNS
# PREPROC_FLAGS:=-DSTRICT_COL_ACCUM_ORDER
PREPROC_FLAGS:=-DNO_DISK_IO -DSTRICT_COL_ACCUM_ORDER
LINKER_FLAGS:=

WORK_DIR:=build
REPORT_DIR:=reports
LOG_DIR:=logs

PLATFORM:=$(XILINX_VITIS)/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm

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
      --aie.xlopt=$(OPT_LVL) \
      --aie.heapsize=1024 \
      --aie.stacksize=$(STACK_SIZE) \
      --platform $(PLATFORM) \
      --include $(XILINX_VITIS)/aietools/include \
      $(foreach x,$(INCLUDE_DIRS),--include $x) \
      $(foreach x,$(PREPROC_FLAGS),--aie.Xpreproc=$x) \
      $(foreach x,$(LINKER_FLAGS),--aie.Xpslinker=$x) \
      --report_dir $(REPORT_DIR) \
      --work_dir $(WORK_DIR) \
      --log_dir $(LOG_DIR)

.PHONY: build-hw \
	build-x86sim \
	build-aiesim \
	run-x86sim \
	debug-x86sim \
	run-aiesim \
	clean

build-hw: $(SOURCES)
	$(AIECC) --target hw src/main.cpp

build-x86sim: $(SOURCES)
	$(AIECC) --target x86sim src/main.cpp

run-x86sim:
	x86simulator --pkg-dir=$(WORK_DIR) --trace

debug-x86sim:
	x86simulator --pkg-dir=$(WORK_DIR) --trace --gdb

run-aiesim:
	aiesimulator --pkg-dir=$(WORK_DIR) --profile

clean:
	-rm -rf $(WORK_DIR)/*
	-rm -rf $(REPORT_DIR)/*
	-rm -rf $(LOG_DIR)/*
	-rm -rf aiesimulator_output/
	-rm -rf x86simulator_output/
	-rm -f *.log *.jou *.db libadf.a Map_Report.csv
