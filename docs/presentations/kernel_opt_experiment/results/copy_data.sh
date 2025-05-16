#!/usr/bin/bash
set -x
set -e
BDIR=../build_hw
REPORT_DIR=$1
TARGET_TILE=24_4
TARGET_FUNC=gate_half_buff

cp -r $BDIR/reports $REPORT_DIR/
cp $BDIR/aie/$TARGET_TILE/$TARGET_TILE.log $REPORT_DIR/
cp $BDIR/aie/$TARGET_TILE/Release/$TARGET_TILE.o.lst $REPORT_DIR/
cp $BDIR/aie/$TARGET_TILE/Release/$TARGET_FUNC.o.lst $REPORT_DIR/
cp $BDIR/aie/$TARGET_TILE/Release/chesswork/$TARGET_FUNC-*_info/sfg.rpt $REPORT_DIR/
cp $BDIR/../AIESimulator.log $REPORT_DIR/
