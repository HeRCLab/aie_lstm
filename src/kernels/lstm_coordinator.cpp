#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

// LSTM recurrent/persistent state.
static float state[N_LSTM_UNITS] = {0};

void lstm_coordinator(adf::input_buffer<float> &__restrict input_gate_rslt,
                      adf::input_buffer<float> &__restrict forget_gate_rslt,
                      adf::input_buffer<float> &__restrict output_gate_rslt,
                      adf::input_buffer<float> &__restrict modulation_gate_rslt,
                      adf::output_buffer<float> &__restrict state_next,
                      adf::output_buffer<float> &__restrict output,
                      // Run Time Parameters.
                      const float (&w_dense)[N_LSTM_UNITS]) {
  auto iGIter = aie::cbegin_vector<VECTOR_LANES, N_LSTM_UNITS>(input_gate_rslt);
  auto fGIter =
      aie::cbegin_vector<VECTOR_LANES, N_LSTM_UNITS>(forget_gate_rslt);
  auto oGIter =
      aie::cbegin_vector<VECTOR_LANES, N_LSTM_UNITS>(output_gate_rslt);
  auto mGIter =
      aie::cbegin_vector<VECTOR_LANES, N_LSTM_UNITS>(modulation_gate_rslt);
  auto wDenseIter =
      aie::cbegin_restrict_vector<VECTOR_LANES, N_LSTM_UNITS>(w_dense);

  auto currStateIter =
      aie::cbegin_restrict_vector<VECTOR_LANES, N_LSTM_UNITS>(state);
  auto nextStateIter =
      aie::begin_vector<VECTOR_LANES, N_LSTM_UNITS>(state_next);

  aie::vector<float, VECTOR_LANES> v1, v2, v3, out_v;
  next_s aie::accumulator<accfloat, VECTOR_LANES> acc;
  aie::accumulator<accfloat, VECTOR_LANES> dense_acc =
      aie::zeros<float>(VECTOR_LANES);

  // FIXME: This code block has scope for further optimisation.
  [[chess:prepare_for_pipelining]] for (int i = 0; i < N_LSTM_UNITS; i++) {
    // Calculate next state.
    v1 = sigm(*fGIter++);
    acc = aie::mul(v1, *currStateIter++);
    v2 = sigm(*iGIter++);
    v3 = tanh(*mGIter++);
    acc = aie::mac(acc, v2, v3);
    next_s = acc.to_vector<float>(0);
    *nextStateIter++ = next_s;

    // Calculate partial dense layer output.
    v1 = sigm(*oGIter++);
    v2 = sigm(next_s);
    out_v = aie::add(v1, v2);
    dense_acc = aie::mac(dense_acc, out_v, *wDenseIter++);
  }

  // Set output.
  *output = aie::reduce_add(dense_acc);
}
