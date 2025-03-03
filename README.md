# AMD AI Engine LSTM Implementation

This repository aims to provide an implementation of dropbear LSTM
for AMD Versal ACAP AI Engine architecture, in C++ using AIE API.

## Compiler Quirks

### Function Inlining

Shared utility functions like activation functions defined outside the kernel must be
declared `inline`. Otherwise, x86 simulator build will fail with some re-definition
error. If inlining is not desired for cases where program memory is a concern,
use `inline __attribute__((noinline))`.

