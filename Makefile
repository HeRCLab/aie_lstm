SRC_DIR:=src/
INCLUDE_DIRS:=$(SRC_DIR) \
	$(SRC_DIR)/kernels \

# Type of implementation variations:
#
# 4X1_BUFF: 4 gates, buffer based comm.
#
# 4x2_BUFF: 8 half-gates, each split vertically,
#           buffer based comm.
#
# 4x2_STRM: 8 half-gates, each split vertically,
#           stream based comm.
VARIANT:=4X2_BUFF

GATE_HALF_LOOP:=V1

# 0 : No kernel optimizations.
#
# 1 : Computation of heap size, generation of guidance based on LLVM
#     IR analysis, and insertion of loop pragmas.
#
# 2 : Optimizations of level 1 and in addition loop peeling for
#     unrolled loops and automatic inlining.
OPT_LVL:=0

STACK_SIZE:=4096

# Pre-processor directives:
# ------------------------------------------------------------------,
# USE_PWL_ACTIVATION_FNS : Use piecewise linear activation funcs.
#
# USE_APPROX_ACTIVATION_FNS : Use activation functions implemented
#   with various recursive polynomial approximation techniques.
#
# STRICT_COL_ACCUM_ORDER : Use strict column accumulation order to
#   generate identical output to ref implementation. This code path is
#   SLOW!
#
# NO_DISK_IO : Instead of reading input out data from disk, embedded
#   them in header files.
#
# DEBUG: Print debug statements.
# ------------------------------------------------------------------'
PREPROC_FLAGS:=-DNO_DISK_IO -DUSE_PWL_ACTIVATION_FNS # -DDEBUG
LINKER_FLAGS:=

WORK_DIR_X86:=build_x86
WORK_DIR_HW:=build_hw
REPORT_DIR:=reports
LOG_DIR:=logs

ADF_API_LOG_LVL:=2
AIE_LOG_LVL:=2

PLATFORM:=$(XILINX_VITIS)/base_platforms/xilinx_vck190_base_202420_1/xilinx_vck190_base_202420_1.xpfm

SOURCES := $(wildcard $(SRC_DIR)/*.cpp) \
	   $(wildcard $(SRC_DIR)/*.hpp) \
	   $(wildcard $(SRC_DIR)/kernels/*.cpp) \
	   $(wildcard $(SRC_DIR)/kernels/*.hpp) \
	   $(wildcard $(SRC_DIR)/kernels/activation_funcs/*.cpp) \
	   $(wildcard $(SRC_DIR)/kernels/activation_funcs/*.hpp) \

PREPROC_FLAGS += -DVARIANT_$(VARIANT)
PREPROC_FLAGS += -DGATE_HALF_LOOP_$(GATE_HALF_LOOP)


AIECC := v++ \
      --compile \
      --mode aie \
      --aie.debug \
      --aie.verbose \
      --aie.adf-api-log-level=$(ADF_API_LOG_LVL) \
      --aie.log-level=$(AIE_LOG_LVL) \
      --aie.xlopt=$(OPT_LVL) \
      --aie.stacksize=$(STACK_SIZE) \
      --aie.fastmath \
      --aie.fast-floats \
      --aie.fast-nonlinearfloats \
      --aie.Xchess="gate_half_buff:backend.mist2.maxfoldk=512" \
      --aie.Xchess="gate_half_buff:backend.mist2.xargs=-ggraph" \
      --platform $(PLATFORM) \
      --include $(XILINX_VITIS)/aietools/include \
      $(foreach x,$(INCLUDE_DIRS),--include $x) \
      $(foreach x,$(PREPROC_FLAGS),--aie.Xpreproc=$x) \
      $(foreach x,$(LINKER_FLAGS),--aie.Xpslinker=$x) \
      --report_dir $(REPORT_DIR) \
      --log_dir $(LOG_DIR) \


.PHONY: build-hw \
	run-aiesim \
	build-x86sim \
	run-x86sim \
	debug-x86sim \
	clean-x86 \
	clean-hw \
	clean-comm \
	clean

build-x86sim: $(SOURCES)
	$(AIECC) --target x86sim --work_dir $(WORK_DIR_X86) src/main.cpp

run-x86sim:
	x86simulator --pkg-dir=$(WORK_DIR_X86) --trace

debug-x86sim:
	x86simulator --pkg-dir=$(WORK_DIR_X86) --trace --gdb

build-hw: $(SOURCES)
	$(AIECC) --target hw --work_dir $(WORK_DIR_HW) src/main.cpp

run-aiesim:
	aiesimulator \
		--profile \
		--pkg-dir=$(WORK_DIR_HW) \
		--options-file=aiesim-options.txt \
		--dump-vcd aiesim

clean-comm:
	-rm -f *.log *.jou *.db libadf.a Map_Report.csv
	-rm -rf $(REPORT_DIR)/*
	-rm -rf $(LOG_DIR)/*

clean-x86:
	-rm -rf $(WORK_DIR_X86)/*
	-rm -rf x86simulator_output/

clean-hw:
	-rm -rf $(WORK_DIR_HW)/*
	-rm -rf aiesimulator_output/

clean: clean-comm clean-hw clean-x86
