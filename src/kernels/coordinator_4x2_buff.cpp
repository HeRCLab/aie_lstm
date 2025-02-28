#include "activation_funcs.hpp"
#include "kernel_lib.hpp"
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

// LSTM recurrent/persistent state.
alignas(vector_decl_align) static float state[LSTM_UNITS] = {0};

// A variable to differentiate initial run so that initial state can
// be sent without waiting for gate results.
static int first_iteration = 1;

void coordinator_4x2_buff(
    // Input buffers.
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict ig_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict ig_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict fg_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict fg_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict og_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict og_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict mg_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict mg_down_rslt,

    // Run Time Parameters.
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,

    // Output buffers.
    adf::output_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state_out,
    // The minimum size of a buffer is 16 Bytes = 4 float
    // samples. Since each iteration will produce just a single
    // result, the rest of 12 bytes will be wasted.
    adf::output_buffer<float, adf::extents<OUTPUT_BLOCK_SIZE>> &__restrict output) {

    // Set C compatible rounding mode.
    set_rounding(aie::rounding_mode::conv_even);

    auto state_iter = begin_restrict_vector<VECTOR_LANES>(state);
    auto w_dense_iter = cbegin_vector<VECTOR_LANES>(w_dense);
    auto outp_iter = begin(output);

    constexpr int block_count = LSTM_UNITS / VECTOR_LANES;

    // Sent initial state down to the gates.
    if (first_iteration) {
        // printf("Sending initial state!\n");
        first_iteration = 0;

        state_out.acquire();
        auto state_out_iter = begin_vector<VECTOR_LANES>(state_out);
        for (int i = 0; i < block_count; i++) {
            *state_out_iter++ = *state_iter++;
        }
        state_out.release();

        // Reset state iterators.
        state_iter -= block_count;
        state_out_iter -= block_count;
    }

    // Wait for results to arrive from all four gates.
    ig_up_rslt.acquire();
    fg_up_rslt.acquire();
    og_up_rslt.acquire();
    mg_up_rslt.acquire();

    // Get access to state output buffer.
    state_out.acquire();

    // Setup iterators on acquired async buffers for upper half gates.
    auto ig_iter = cbegin_vector<VECTOR_LANES>(ig_up_rslt);
    auto fg_iter = cbegin_vector<VECTOR_LANES>(fg_up_rslt);
    auto og_iter = cbegin_vector<VECTOR_LANES>(og_up_rslt);
    auto mg_iter = cbegin_vector<VECTOR_LANES>(mg_up_rslt);
    auto state_out_iter = begin_vector<VECTOR_LANES>(state_out);

    float sum;
    vector<float, VECTOR_LANES> v1, v2, v3, out_lstm, next_state;
    accum<accfloat, VECTOR_LANES> acc, acc_dense;

    // Calculate next-state and output from upper half gates.
    // ----------------------------------------------------------------,
    sum = bias_dense;
    acc_dense = zeros<accfloat, VECTOR_LANES>();
    for (int i = 0; i < block_count / 2; i++)
        chess_prepare_for_pipelining {
            // Calculate next state.
            v1 = SIGM(*fg_iter++);
            acc = mul(v1, *state_iter);
            v2 = SIGM(*ig_iter++);
            v3 = TANH(*mg_iter++);
            acc = mac(acc, v2, v3);
            next_state = acc.to_vector<float>(0);

            // Calculate LSTM output.
            v1 = SIGM(*og_iter++);
            v2 = SIGM(next_state);
            out_lstm = add(v1, v2);

            // Multiply LSTM output with dense weights and accumulate them
            // partially.
            acc_dense = mac(acc_dense, out_lstm, *w_dense_iter++);

            *state_iter++ = next_state;
            *state_out_iter++ = next_state;
            // print(next_state, true, "next_state=");
        }
    // ----------------------------------------------------------------'

    // Release for upper half gates.
    ig_up_rslt.release();
    fg_up_rslt.release();
    og_up_rslt.release();
    mg_up_rslt.release();

    // Acquire buffers for lower half gates.
    ig_down_rslt.acquire();
    fg_down_rslt.acquire();
    og_down_rslt.acquire();
    mg_down_rslt.acquire();

    // Create iterators.
    ig_iter = cbegin_vector<VECTOR_LANES>(ig_down_rslt);
    fg_iter = cbegin_vector<VECTOR_LANES>(fg_down_rslt);
    og_iter = cbegin_vector<VECTOR_LANES>(og_down_rslt);
    mg_iter = cbegin_vector<VECTOR_LANES>(mg_down_rslt);

    constexpr int partial_block_id = (LSTM_UNITS_REAL - LSTM_UNITS/2) / VECTOR_LANES;
    constexpr int partial_block_size = (LSTM_UNITS_REAL - LSTM_UNITS/2) % VECTOR_LANES;

    // Computations for lower half gates.
    // ----------------------------------------------------------------,
    for (int i = 0; i < block_count / 2; i++)
        chess_prepare_for_pipelining {
            // Calculate next state.
            v1 = SIGM(*fg_iter++);
            acc = mul(v1, *state_iter);
            v2 = SIGM(*ig_iter++);
            v3 = TANH(*mg_iter++);
            acc = mac(acc, v2, v3);
            next_state = acc.to_vector<float>(0);

            // Calculate LSTM output.
            v1 = SIGM(*og_iter++);
            v2 = SIGM(next_state);
            out_lstm = add(v1, v2);

            // Since our reference network does not use all of the LSTM
            // units, we need to discard some computations from tail.
            if (i == partial_block_id) {
                for (int j = partial_block_size; j < VECTOR_LANES; j++) {
                    next_state[j] = 0.0f;
                    out_lstm[j] = 0.0f;
                }
            } else if (i > partial_block_id) {
                next_state = zeros<float, VECTOR_LANES>();
                out_lstm = zeros<float, VECTOR_LANES>();
            }

            // Multiply LSTM output with dense weights and accumulate them
            // partially.
            acc_dense = mac(acc_dense, out_lstm, *w_dense_iter++);

            *state_iter++ = next_state;
            *state_out_iter++ = next_state;
            // print(next_state, true, "next_state=");
        }
    // ----------------------------------------------------------------'

    // Release the lower half gate and output buffers.
    ig_down_rslt.release();
    fg_down_rslt.release();
    og_down_rslt.release();
    mg_down_rslt.release();
    state_out.release();

    // Set first sample of output buffer to dense network output.
    float col_sum = reduce_add(acc_dense.to_vector<float>(0));
    sum += col_sum;
    *outp_iter++ = sum;

    // printf("Lstm coord output: %.12f\n", sum);

    // For clarity, set rest of the output buffer to zero.
    for (int i = 0; i < 3; i++) {
        *outp_iter++ = 0.0f;
    }
}
