#ifndef COMMON_DEFS_H
#define COMMON_DEFS_H

#define VECTOR_LANES 16

// Actual number of LSTM units is 100. 128 is chosen so that when the
// recurrent weights are split into two matrices of Nx(M/2), M is
// divisible by native vector lane size of 16.
#define LSTM_UNITS 128
#define LSTM_UNITS_NON_EMPTY 100
#define LSTMP_UNITS_EMPTY (LSTM_UNITS - LSTM_UNITS_NON_EMPTY)

// Actual input dim is 20. 32 is chosen to be even divisible by vector
// lane size of 16. The last 12 bytes will be filled by zero by the
// buffer feeder.
#define INPUT_DIM_NON_EMPTY 20
#define INPUT_DIM 32

#define W_STATE_COLS LSTM_UNITS
#define W_STATE_SIZE (LSTM_UNITS * SPLIT_W_STATE_COLS)
#define W_INPUT_SIZE (LSTM_UNITS * INPUT_DIM)

#define N_GATES 4
#define GMIO_BURST_LENGTH 64  /* Options: 64, 128, 256 */
#define GMIO_BANDWIDTH 1024  /* Unit: MB/s */

#define INPUT_DATA_PATH "data/dropbear.dat" 
#define WEIGHT_DATA_PATH "data/weights.dat" 

#endif
