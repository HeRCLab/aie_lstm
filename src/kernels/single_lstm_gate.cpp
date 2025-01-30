#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

void single_lstm_gate(
    // Input buffers.
    adf::input_circular_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_circular_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,

    // Run Time Parameters.
    // State weight matrix is split horizontally into two because each
    // runtime parameter needs to fit within a single memory module of
    // 32KB (roughly).
    const float (&w_state_left)[SPLIT_W_STATE_SIZE],
    const float (&w_state_right)[SPLIT_W_STATE_SIZE],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS],

    // Output buffers.
    adf::output_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output) {

    // Input and state iterators are circular, since they will be
    // repeatedly used to multiply with each row of the input and state
    // weight matrix, respectively.
    auto inpIter = aie::cbegin_vector_circular<VECTOR_LANES, INPUT_DIM>(input);
    auto stateIter = aie::cbegin_vector_circular<VECTOR_LANES, LSTM_UNITS>(state);

    // Setup iterators for weights and bias run time parameters.
    auto wInpIter = aie::cbegin_vector<VECTOR_LANES>(w_input);
    auto wStateLIter = aie::cbegin_vector<VECTOR_LANES>(w_state_left);
    auto wStateRIter = aie::cbegin_vector<VECTOR_LANES>(w_state_right);
    auto bIter = aie::cbegin<LSTM_UNITS>(bias);

    auto outIter = aie::begin_vector<VECTOR_LANES>(output);

    aie::accumulator<accfloat, VECTOR_LANES> acc1, acc2;
    aie::vector<float, VECTOR_LANES> va, vb;
    float sum1, sum2;

    [[chess::prepare_for_pipelining]] {

        for (int row = 0; row < LSTM_UNITS_NON_EMPTY; row++) {
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
