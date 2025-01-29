#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#define VECTOR_LANES 16

// Actual number of LSTM units is 100. 128 is chosen so that when the
// recurrent weights are split into two matrices of Nx(M/2), M is
// divisible by native vector lane size of 16.
#define LSTM_UNITS_NON_EMPTY 100
#define LSTM_UNITS 128

// Actual input dim is 20. 32 is chosen to be even divisible by vector
// lane size of 16. The last 12 bytes will be filled by zero by the
// buffer feeder.
#define INPUT_DIM_NON_EMPTY 20
#define INPUT_DIM 32

#define SPLIT_W_STATE_COLS (LSTM_UNITS / 2)
#define SPLIT_W_STATE_SIZE (LSTM_UNITS * SPLIT_W_STATE_COLS)
#define W_INPUT_SIZE (LSTM_UNITS * INPUT_DIM)

#endif
