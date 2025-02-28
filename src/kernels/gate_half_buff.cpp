#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

void gate_half_buff(
    // Input buffers.
    adf::input_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,

    // Run Time Parameters.
    const float (&w_state_part)[W_STATE_SIZE / 2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS / 2],
    const int partial_computation,

    // Output buffers.
    adf::output_buffer<float, adf::extents<LSTM_UNITS / 2>> &__restrict output) {

    // Set C compatible rounding mode.
    set_rounding(aie::rounding_mode::conv_even);

    auto inp_iter = cbegin_vector<VECTOR_LANES>(input);

    // Setup iterators for weights and bias run time parameters.
    auto w_inp_iter = cbegin_vector<VECTOR_LANES>(w_input);
    auto w_state_iter = cbegin_vector<VECTOR_LANES>(w_state_part);
    auto b_iter = cbegin_vector<VECTOR_LANES>(bias);
    auto out_iter = begin_vector<VECTOR_LANES>(output);
    auto state_iter = cbegin_vector<VECTOR_LANES>(state);

    float sum1, sum2, rslt;
    vector<float, VECTOR_LANES> va, vb, rslt_v;

    // NOTE: The reason for using two accumulators is to fill the
    // pipeline during the two idle cycles of float MACS, as
    // described in UG1079(v2024.1), Ch 3, Sec "Floating Point
    // Operations", page 54.
    accum<accfloat, VECTOR_LANES> acc1, acc2;

    constexpr int input_blocks = INPUT_DIM / VECTOR_LANES;
    constexpr int state_blocks = W_STATE_COLS / VECTOR_LANES;

    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++)
        chess_prepare_for_pipelining {
            if (partial_computation && (row > (LSTM_UNITS_REAL - LSTM_UNITS / 2))) {
                rslt = 0.0f;
            } else {
                acc1 = zeros<accfloat, VECTOR_LANES>();
                acc2 = zeros<accfloat, VECTOR_LANES>();

                // Multiply and accumulate with state weight matrix.
                for (int i = 0; i < W_STATE_COLS / VECTOR_LANES / 2; i++) {
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc1 = mac(acc1, va, vb);
                    va = *w_state_iter++;
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
                rslt = sum1 + sum2;

                // Reset input and state iterators.
                inp_iter -= input_blocks;
                state_iter -= state_blocks;
            }

            // Place the result scalar into the vector.
            int vi = row % VECTOR_LANES;
            rslt_v.set(rslt, vi);

            // If result vector is full, publish it to output.
            if ((row % VECTOR_LANES) == (VECTOR_LANES - 1)) {
                // Add bias to generate one block of gate results.
                rslt_v = add(*b_iter++, rslt_v);
                *out_iter++ = rslt_v;
                // print(rslt_v, true, "rslt_v=");
                rslt_v = zeros<float, VECTOR_LANES>();
            }
        }
}
