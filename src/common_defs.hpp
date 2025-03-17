#pragma once

#define VECTOR_LANES 16

// Actual number of LSTM units is 100. 128 is chosen so that when the
// recurrent weights are split into two matrices of Nx(M/2), M is
// divisible by native vector lane size of 16.
#define LSTM_UNITS 128
#define LSTM_UNITS_REAL 100
#define LSTM_UNITS_EMPTY (LSTM_UNITS - LSTM_UNITS_REAL)

// Actual input dim is 20. 32 is chosen to be even divisible by vector
// lane size of 16. The last 12 bytes will be filled by zero by the
// buffer feeder.
#define INPUT_DIM_REAL 20
#define INPUT_DIM 32

#if defined(VARIANT_4X2_STRM)
#define OUTPUT_BLOCK_SIZE 1
#else
#define OUTPUT_BLOCK_SIZE 4 /* Minimum buffer size of 16bytes */
#endif

#define W_STATE_COLS LSTM_UNITS
#define W_STATE_SIZE (LSTM_UNITS * W_STATE_COLS)
#define W_INPUT_SIZE (LSTM_UNITS * INPUT_DIM)

#define W_STATE_COLS_REAL LSTM_UNITS_REAL
#define W_STATE_SIZE_REAL (LSTM_UNITS_REAL * W_STATE_COLS_REAL) 
#define W_INPUT_SIZE_REAL (LSTM_UNITS_REAL * INPUT_DIM_REAL) 

#define N_GATES 4
#define N_HALF_GATES (N_GATES * 2)
#define GMIO_BURST_LENGTH 64  /* Options: 64, 128, 256 */
#define GMIO_BANDWIDTH 1024  /* Unit: MB/s */

#define DROPBEAR_DATA_PATH "data/dropbear.dat" 
#define WEIGHT_DATA_PATH "data/weights.dat" 

#define OUTPUT_SAMPLES 3
#define OUTPUT_FILE_PATH "aie_lstm_output.txt"
