#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#define VECTOR_LANES 16

// Actual number of LSTM units is 100. 128 is chosen so that when the
// recurrent weights are split into two matrices of Nx(M/2), M is
// divisible by native vector lane size of 16.
#define N_LSTM_UNITS_REAL 100
#define N_LSTM_UNITS 128

// Actual input dim is 20. 32 is chosen to be even divisible by vector
// lane size of 16. The last 12 bytes will be filled by zero by the
// buffer feeder.
#define INPUT_DIM_REAL 20
#define INPUT_DIM 32

#endif
