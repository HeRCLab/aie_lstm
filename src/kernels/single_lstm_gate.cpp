#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

#define SPLIT_W_STATE_COLS (N_LSTM_UNITS / 2)
#define SPLIT_W_STATE_SIZE (SPLIT_W_STATE_COLS * N_LSTM_UNITS)
#define W_INPUT_SIZE (N_LSTM_UNITS * INPUT_DIM)

void single_lstm_gate(adf::input_buffer<float> &__restrict input,
                      adf::input_buffer<float> &__restrict state,
                      adf::output_buffer<float> &__restrict output,
                      // Run Time Parameters.
                      const float (&w_input)[W_INPUT_SIZE],
                      // State weight matrix is split horizontally
                      // into two because each runtime parameter needs
                      // to fit within a single memory module of 32KB
                      // (roughly).
                      const float (&w_state_left)[SPLIT_W_STATE_SIZE],
                      const float (&w_state_right)[SPLIT_W_STATE_SIZE],
                      const float (&bias)[N_LSTM_UNITS]) {
  // Input and state iterators are circular, since they will be
  // repeatedly used to multiply with each row of the input and state
  // weight matrix, respectively.
  auto inpIter = aie::cbegin_vector_circular<VECTOR_LANES, INPUT_DIM>(input);
  auto stateIter =
      aie::cbegin_vector_circular<VECTOR_LANES, N_LSTM_UNITS>(state);

  auto wInpIter = aie::cbegin_vector<VECTOR_LANES, INPUT_DIM>(weight_input);
  auto wStateLIter =
      aie::cbegin_vector<VECTOR_LANES, SPLIT_W_STATE_SIZE>(weight_state_left);
  auto wStateRIter =
      aie::cbegin_vector<VECTOR_LANES, SPLIT_W_STATE_SIZE>(weight_state_right);

  auto bIter = aie::cbegin<N_LSTM_UNITS>(bias);
  auto outIter = aie::begin_vector<VECTOR_LANES, N_LSTM_UNITS>(output);

  aie::accumulator<accfloat, VECTOR_LANES> acc1, acc2;
  aie::vector<float, VECTOR_LANES> va, vb;
  float sum1, sum2;

  [[chess::prepare_for_pipelining]] {

    for (int row = 0; row < N_LSTM_UNITS; row++) {
      // NOTE: The reason for using two accumulators is to fill the
      // pipeline during the two idle cycles of float MACS, as
      // described in UG1079(v2024.1), Ch 3, Sec "Floating Point
      // Operations", page 54.
      acc1 = aie::zeros<accfloat, VECTOR_LANES>();
      acc2 = aie::zeros<accfloat, VECTOR_LANES>();

      // Multiply and accumulate with left half of state weight matrix.
      for (int i = 0; i < SPLIT_W_STATE_COLS / VECTOR_LANES / 2; i++) {
        va = *wStateLIter++;
        vb = *recIter++;
        acc1 = aie::mac(acc1, va, vb);
        va = *wStateLIter++;
        vb = *recIter++;
        acc2 = aie::mac(acc2, va, vb);
      }

      // Multiply and accumulate with right half of state weight matrix.
      for (int i = 0; i < SPLIT_W_STATE_COLS / VECTOR_LANES / 2; i++) {
        va = *wStateRIter++;
        vb = *recIter++;
        acc1 = aie::mac(acc1, va, vb);
        va = *wStateRIter++;
        vb = *recIter++;
        acc2 = aie::mac(acc2, va, vb);
      }

      // Mutiply and accumulate with input-weight matrix.
      for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++) {
        acc1 = aie::mac(acc1, *wInpIter++, *inpIter++);
      }

      // Sum all of the accumulator lanes and the bias.
      sum1 = aie::reduce_add(acc1.to_vector<float>(0));
      sum2 = aie::reduce_add(acc2.to_vector<float>(0));
      sum1 = aie::add(sum1, sum2);

      // Save output.
      *outIter++ = aie::add(sum1, *bIter++);
    }
  }
}
