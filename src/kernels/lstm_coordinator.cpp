#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

// LSTM recurrent/persistent state.
alignas(aie::vector_decl_align) static float state[LSTM_UNITS] = {0};

// A variable to differentiate initial run so that initial state can
// be sent without waiting for gate results.
static int first_iteration = 1;

void lstm_coordinator(
    // Input buffers.
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict input_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict forget_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict modulation_gate_rslt,

    // Run Time Parameters.
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,

    // Output buffers.
    adf::output_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state_out,
    // The minimum size of a buffer is 16 Bytes = 4 float
    // samples. Since each iteration will produce just a single
    // result, the rest of 12 bytes will be wasted.
    adf::output_buffer<float, adf::extents<4>> &__restrict output) {

    auto stateIter = aie::cbegin_restrict_vector<VECTOR_LANES>(state);
    auto stateOutIter = aie::begin_vector<VECTOR_LANES>(state_out);
    auto iGIter = aie::cbegin_vector<VECTOR_LANES>(input_gate_rslt);
    auto fGIter = aie::cbegin_vector<VECTOR_LANES>(forget_gate_rslt);
    auto oGIter = aie::cbegin_vector<VECTOR_LANES>(output_gate_rslt);
    auto mGIter = aie::cbegin_vector<VECTOR_LANES>(modulation_gate_rslt);
    auto wDenseIter = aie::cbegin_restrict_vector<VECTOR_LANES>(w_dense);

    float sum;
    aie::vector<float, VECTOR_LANES> v1, v2, v3, out_lstm, next_s;
    aie::accumulator<accfloat, VECTOR_LANES> acc, acc_dense;

    // Sent initial state down to the gates.
    if (first_iteration) {
        first_iteration = 0;

        state_out.acquire();
        for (int i = 0; i < LSTM_UNITS / VECTOR_LANES; i++) {
            *stateOutIter++ = *stateIter++;
        }
        state_out.release();
    }

    // Wait for results to arrive from all four gates.
    input_gate_rslt.acquire();
    forget_gate_rslt.acquire();
    output_gate_rslt.acquire();
    modulation_gate_rslt.acquire();

    // Get access to state output buffer.
    state_out.acquire();

    // Fixme: This code block has scope for further optimisation.
    acc_dense = aie::zeros<float, VECTOR_LANES>();
    [[chess:prepare_for_pipelining]] for (int i = 0; i < LSTM_UNITS_NON_EMPTY; i++) {
        // Calculate next state.
        v1 = sigm(*fGIter++);
        acc = aie::mul(v1, *stateIter);
        v2 = sigm(*iGIter++);
        v3 = tanh(*mGIter++);
        acc = aie::mac(acc, v2, v3);
        next_state = acc.to_vector<float>(0);

        // Calculate LSTM output.
        v1 = sigm(*oGIter++);
        v2 = sigm(next_state);
        out_lstm = aie::add(v1, v2);

        // Accumulate partial dense layer output.
        acc_dense = aie::mac(dense_acc, out_lstm, *wDenseIter++);

        *stateIter++ = next_state;
        *outStateIter++ = next_state;
    }

    // Release the buffers.
    input_gate_rslt.release();
    forget_gate_rslt.release();
    output_gate_rslt.release();
    modulation_gate_rslt.release();
    state_out.release();

    // Set first sample of output buffer to dense network output.
    sum = aie::reduce_add(acc_dense);
    *output = sum + bias_dense;

    // For clarity, set rest of the output buffer to zero.
    for (int i = 0; i < 3; i++) {
        *output++ = 0.0f;
    }
}
