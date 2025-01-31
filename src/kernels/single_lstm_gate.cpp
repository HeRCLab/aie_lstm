#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

void single_lstm_gate(
    // Input buffers.
    adf::input_circular_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_circular_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,

    // Run Time Parameters.
    // State weight matrix is split horizontally into two because each
    // runtime parameter needs to fit within a single memory module of
    // 32KB (roughly).
    const float (&w_state_left)[W_STATE_SIZE/2],
    const float (&w_state_right)[W_STATE_SIZE/2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS],

    // Output buffers.
    adf::output_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output) {

    // Input and state iterators are circular, since they will be
    // repeatedly used to multiply with each row of the input and state
    // weight matrix, respectively.
    auto inp_iter = cbegin_vector_circular<VECTOR_LANES, INPUT_DIM>(input);
    auto state_iter = cbegin_vector_circular<VECTOR_LANES, LSTM_UNITS>(state);

    // Setup iterators for weights and bias run time parameters.
    auto w_inp_iter = cbegin_vector<VECTOR_LANES>(w_input);
    auto w_state_left_iter = cbegin_vector<VECTOR_LANES>(w_state_left);
    auto w_state_right_iter = cbegin_vector<VECTOR_LANES>(w_state_right);
    auto b_iter = cbegin<LSTM_UNITS>(bias);

    auto out_iter = begin_vector<VECTOR_LANES>(output);

    accumulator<accfloat, VECTOR_LANES> acc1, acc2;
    vector<float, VECTOR_LANES> va, vb;
    float sum1, sum2, rslt;

    constexpr int partial_block_id = LSTM_UNITS_NON_EMPTY / VECTOR_LANES;
    constexpr int partial_block_size = LSTM_UNITS_NON_EMPTY % VECTOR_LANES;

    [[chess::prepare_for_pipelining]] {
      for (int i = 0; row < LSTM_UNITS / VECTOR_LANES; i++) {
            // NOTE: The reason for using two accumulators is to fill the
            // pipeline during the two idle cycles of float MACS, as
            // described in UG1079(v2024.1), Ch 3, Sec "Floating Point
            // Operations", page 54.
            acc1 = zeros<accfloat, VECTOR_LANES>();
            acc2 = zeros<accfloat, VECTOR_LANES>();

            // Multiply and accumulate with left half of state weight matrix.
            for (int i = 0; i < W_STATE_COLS / VECTOR_LANES / 2; i++) {
                va = *w_state_left_iter++;
                vb = *state_iter++;
                acc1 = mac(acc1, va, vb);
                va = *w_state_left_iter++;
                vb = *state_iter++;
                acc2 = mac(acc2, va, vb);
            }

            // Multiply and accumulate with right half of state weight matrix.
            for (int i = 0; i < W_STATE_COLS / VECTOR_LANES / 2; i++) {
                va = *w_state_right_iter++;
                vb = *state_iter++;
                acc1 = mac(acc1, va, vb);
                va = *w_state_right_iter++;
                vb = *state_iter++;
                acc2 = mac(acc2, va, vb);
            }

            // Mutiply and accumulate with input-weight matrix.
            for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++) {
                acc1 = mac(acc1, *w_inp_iter++, *inp_iter++);
            }

            // Sum all of the accumulator lanes.
            sum1 = reduce_add(acc1.to_vector<float>(0));
            sum2 = reduce_add(acc2.to_vector<float>(0));
            sum1 = add(sum1, sum2);

            // Add bias to generate one block of gate results.
	    rslt = add(sum1, *b_iter);

	    // Since our reference network does not use all of the LSTM
	    // units, we need to discard some computations from tail.
	    if (i == partial_block_id) {
		for (int j = partial_block_size; j < VECTOR_LANES; j++) {
		    rslt[j] = 0.0f;
		}
	    } else if (i > partial_block_id) {
		rslt = zeros<float, VECTOR_LANES>();
	    }

	    // Save result to output buffer.
            *out_iter++ = rslt;
	    b_iter++;
        }
    }
}
