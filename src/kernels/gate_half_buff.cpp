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

#if defined(GATE_HALF_LOOP_V6) || defined(GATE_HALF_LOOP_V7) || defined(GATE_HALF_LOOP_V8)
    // Setup circular iterators for input, output and state buffers.
    const vector<float, VECTOR_LANES> *__restrict inp_p = (const vector<float, VECTOR_LANES> *)input.data();
    const vector<float, VECTOR_LANES> *__restrict state_p = (const vector<float, VECTOR_LANES> *)state.data();
    vector<float, VECTOR_LANES> *__restrict outp_p = (vector<float, VECTOR_LANES> *)output.data();

    const vector<float, VECTOR_LANES> *__restrict w_state_p = (const vector<float, VECTOR_LANES> *)w_state_part;
    const vector<float, VECTOR_LANES> *__restrict w_inp_p = (const vector<float, VECTOR_LANES> *)w_input;
    const vector<float, VECTOR_LANES> *__restrict bias_p = (const vector<float, VECTOR_LANES> *)bias;
#else
    // Setup iterators for input, output and state buffers. Since
    // buffers already declared 'restrict', we don't need to use
    // restrict-variant iterators.
    auto inp_iter = cbegin_vector<VECTOR_LANES>(input);
    auto state_iter = cbegin_vector<VECTOR_LANES>(state);
    auto out_iter = begin_vector<VECTOR_LANES>(output);

    // Setup iterators for weights and bias run time parameters.
    auto w_inp_iter = cbegin_restrict_vector<VECTOR_LANES>(w_input);
    auto w_state_iter = cbegin_restrict_vector<VECTOR_LANES>(w_state_part);
    auto b_iter = cbegin_restrict_vector<VECTOR_LANES>(bias);
#endif

    float sum1, sum2, rslt;
    vector<float, VECTOR_LANES> va, vb, vc, vd, rslt_v;

    // NOTE: The reason for using two accumulators is to fill the
    // pipeline during the two idle cycles of float MACS, as
    // described in UG1079(v2024.1), Ch 3, Sec "Floating Point
    // Operations", page 54.
    accum<accfloat, VECTOR_LANES> acc1, acc2;

    constexpr int input_blocks = INPUT_DIM / VECTOR_LANES;
    constexpr int state_blocks = W_STATE_COLS / VECTOR_LANES;

    // Record cycle count before the loop.
    auto n_cycles_before = get_cycles();

#if defined(GATE_HALF_LOOP_V1)
    // - pipelining at outer-most loop
    // - branch to skip null computation
    // - branch to seperate result block
    // - single accumulator
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++)
        chess_prepare_for_pipelining {
            if (partial_computation && (row > (LSTM_UNITS_REAL - LSTM_UNITS / 2))) {
                rslt = 0.0f;
            } else {
                acc1 = zeros<accfloat, VECTOR_LANES>();

                // Multiply and accumulate with state weight matrix.
                for (int i = 0; i < state_blocks; i++) {
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Mutiply and accumulate with input-weight matrix.
                for (int i = 0; i < input_blocks; i++) {
                    va = *w_inp_iter++;
                    vb = *inp_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Sum all of the accumulator lanes.
                rslt = reduce_add(acc1.to_vector<float>(0));

                // Reset input and state iterators.
                state_iter -= state_blocks;
                inp_iter -= input_blocks;
            }

            // Push the result scalar into the vector.
            rslt_v.push(rslt);

            // If result vector is full, publish it to output.
            if ((row % VECTOR_LANES) == (VECTOR_LANES - 1)) {
                rslt_v = reverse(rslt_v);
                // Add bias to generate one block of gate results.
                rslt_v = add(*b_iter++, rslt_v);
                *out_iter++ = rslt_v;
            }
        }
#elif defined(GATE_HALF_LOOP_V2)
    // - pipelining at outer-most loop
    // - branch to skip invalid computation REMOVED
    // - branch to seperate result block
    // - single accumulator
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++)
        chess_prepare_for_pipelining {
            acc1 = zeros<accfloat, VECTOR_LANES>();

            // Multiply and accumulate with state weight matrix.
            for (int i = 0; i < state_blocks; i++) {
                va = *w_state_iter++;
                vb = *state_iter++;
                acc1 = mac(acc1, va, vb);
            }

            // Mutiply and accumulate with input-weight matrix.
            for (int i = 0; i < input_blocks; i++) {
                va = *w_inp_iter++;
                vb = *inp_iter++;
                acc1 = mac(acc1, va, vb);
            }

            // Sum all of the accumulator lanes.
            rslt = reduce_add(acc1.to_vector<float>(0));

            // Push the result scalar into the vector.
            rslt_v.push(rslt);

            // If result vector is full, publish it to output.
            if ((row % VECTOR_LANES) == (VECTOR_LANES - 1)) {
                rslt_v = reverse(rslt_v);
                // Add bias to generate one block of gate results.
                rslt_v = add(*b_iter++, rslt_v);
                *out_iter++ = rslt_v;
            }

            // Reset input and state iterators.
            state_iter -= state_blocks;
            inp_iter -= input_blocks;
        }
#elif defined(GATE_HALF_LOOP_V3)
    // - pipelining at outer-most loop
    // - branch to seperate result-block REMOVED as a static loop
    // - single accumulator
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {

            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++) {
                acc1 = zeros<accfloat, VECTOR_LANES>();

                // Multiply and accumulate with state weight matrix.
                for (int i = 0; i < state_blocks; i++) {
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Mutiply and accumulate with input-weight matrix.
                for (int i = 0; i < input_blocks; i++) {
                    va = *w_inp_iter++;
                    vb = *inp_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Sum all of the accumulator lanes.
                rslt = reduce_add(acc1.to_vector<float>(0));

                // Place the result scalar into the vector.
                rslt_v.push(rslt);

                // Reset input and state iterators.
                state_iter -= state_blocks;
                inp_iter -= input_blocks;
            }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*b_iter, rslt_v);
            // Forward gate output to coordinator.
            *out_iter = rslt_v;

            b_iter++;
            out_iter++;
        }
#elif defined(GATE_HALF_LOOP_V4)
    // - pipelining at outer-most loop
    // - all inner loops flatten-ed
    // - single accumulator
    // - NOTE: This is the best one so far, with 4125 cycles.
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {

            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++)
                chess_flatten_loop {
                    acc1 = zeros<accfloat, VECTOR_LANES>();

                    // Multiply and accumulate with state weight matrix.
                    for (int i = 0; i < state_blocks; i++)
                        chess_flatten_loop {
                            va = *w_state_iter++;
                            vb = *state_iter++;
                            acc1 = mac(acc1, va, vb);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    for (int i = 0; i < input_blocks; i++)
                        chess_flatten_loop {
                            va = *w_inp_iter++;
                            vb = *inp_iter++;
                            acc1 = mac(acc1, va, vb);
                        }

                    // Sum all of the accumulator lanes.
                    rslt = reduce_add(acc1.to_vector<float>(0));

                    // Place the result scalar into the vector.
                    rslt_v.push(rslt);

                    // Reset input and state iterators.
                    state_iter -= state_blocks;
                    inp_iter -= input_blocks;
                }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*b_iter, rslt_v);
            // Forward gate output to coordinator.
            *out_iter = rslt_v;

            b_iter++;
            out_iter++;
        }
#elif defined(GATE_HALF_LOOP_V5)
    // - pipelining at outer-most loop
    // - all inner loops flatten-ed
    // - double accumulator
    // - NOTE: better with 3669 instructions
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {

            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++)
                chess_flatten_loop {
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();

                    // Multiply and accumulate with state weight matrix.
                    for (int i = 0; i < state_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_state_iter++;
                            vb = *state_iter++;
                            acc1 = mac(acc1, va, vb);
                            vc = *w_state_iter++;
                            vd = *state_iter++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    for (int i = 0; i < input_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_inp_iter++;
                            vb = *inp_iter++;
                            acc1 = mac(acc1, va, vb);
                            vc = *w_inp_iter++;
                            vd = *inp_iter++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Sum all of the accumulator lanes.
                    rslt = reduce_add(acc1.to_vector<float>(0));
                    rslt += reduce_add(acc2.to_vector<float>(0));

                    // Place the result scalar into the vector.
                    rslt_v.push(rslt);

                    // Reset input and state iterators.
                    state_iter -= state_blocks;
                    inp_iter -= input_blocks;
                }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*b_iter, rslt_v);
            // Forward gate output to coordinator.
            *out_iter = rslt_v;

            b_iter++;
            out_iter++;
        }
#elif defined(GATE_HALF_LOOP_V6)
    // - used pointers instead of iterators
    // - pipelining at outer loop
    // - inner loops completely flattened
    // - single accumulator
    // - NOTE: min cycle count = 4124!!!
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;
    constexpr int n_state_blocks = W_STATE_COLS / VECTOR_LANES;
    constexpr int n_input_blocks = INPUT_DIM / VECTOR_LANES;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {
            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++)
                chess_flatten_loop {
                    acc1 = zeros<accfloat, VECTOR_LANES>();

                    // Multiply and accumulate with state weight matrix.
                    for (int i = 0; i < n_state_blocks; i++)
                        chess_flatten_loop {
                            va = *w_state_p++;
                            vb = *state_p++;
                            acc1 = mac(acc1, va, vb);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    for (int i = 0; i < n_input_blocks; i++)
                        chess_flatten_loop {
                            va = *w_inp_p++;
                            vb = *inp_p++;
                            acc1 = mac(acc1, va, vb);
                        }

                    // Sum all of the accumulator lanes.
                    rslt = reduce_add(acc1.to_vector<float>(0));

                    // Place the result scalar into the vector.
                    rslt_v.push(rslt);

                    // Reset state and input pointers.
                    state_p -= n_state_blocks;
                    inp_p -= n_input_blocks;
                }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*bias_p++, rslt_v);
            // Forward gate output to coordinator.
            *outp_p++ = rslt_v;
        }
#elif defined(GATE_HALF_LOOP_V7)
    // - used pointers instead of iterators
    // - pipelining at outer loop
    // - inner loops completely flattened
    // - double accumulator
    // - NOTE: min cycle count = 3670 !!!
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;
    constexpr int n_state_blocks = W_STATE_COLS / VECTOR_LANES;
    constexpr int n_input_blocks = INPUT_DIM / VECTOR_LANES;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {
            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++)
                chess_flatten_loop {
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();

                    // Multiply and accumulate with state weight matrix.
                    for (int i = 0; i < n_state_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_state_p++;
                            vb = *state_p++;
                            acc1 = mac(acc1, va, vb);

                            vc = *w_state_p++;
                            vd = *state_p++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    for (int i = 0; i < n_input_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_inp_p++;
                            vb = *inp_p++;
                            acc1 = mac(acc1, va, vb);

                            vc = *w_inp_p++;
                            vd = *inp_p++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Sum all of the accumulator lanes.
                    rslt = reduce_add(acc1.to_vector<float>(0));
                    rslt += reduce_add(acc2.to_vector<float>(0));

                    // Place the result scalar into the vector.
                    rslt_v.push(rslt);

                    // Reset state and input pointers.
                    state_p -= n_state_blocks;
                    inp_p -= n_input_blocks;
                }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*bias_p++, rslt_v);
            // Forward gate output to coordinator.
            *outp_p++ = rslt_v;
        }
#elif defined(GATE_HALF_LOOP_V8)
    // - used pointers instead of iterators
    // - pipelining at outer loop
    // - inner loops completely flattened
    // - double accumulator
    // - Improved accumulator reduction
    // - NOTE: min cycle count = 3267
    constexpr int row_block_size = VECTOR_LANES;
    constexpr int n_row_blocks = LSTM_UNITS / 2 / row_block_size;
    constexpr int n_state_blocks = W_STATE_COLS / VECTOR_LANES;
    constexpr int n_input_blocks = INPUT_DIM / VECTOR_LANES;

    const vector<float, VECTOR_LANES> *__restrict state_start_p = state_p;
    const vector<float, VECTOR_LANES> *__restrict inp_start_p = inp_p;

    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining chess_pipeline_initiation_interval(201) chess_modulo_scheduling_budget_ratio(200) {
            // Compute a block of rows.
            for (int row = 0; row < row_block_size; row++)
                chess_flatten_loop {
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();
                    state_p = state_start_p;
                    inp_p = inp_start_p;

                    // Multiply and accumulate with state weight matrix.
                    for (int i = 0; i < n_state_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_state_p++;
                            vb = *state_p++;
                            acc1 = mac(acc1, va, vb);

                            vc = *w_state_p++;
                            vd = *state_p++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    for (int i = 0; i < n_input_blocks / 2; i++)
                        chess_flatten_loop {
                            va = *w_inp_p++;
                            vb = *inp_p++;
                            acc1 = mac(acc1, va, vb);

                            vc = *w_inp_p++;
                            vd = *inp_p++;
                            acc2 = mac(acc2, vc, vd);
                        }

                    // Sum all of the accumulator lanes.
                    va = acc1.to_vector<float>(0);
                    vb = acc2.to_vector<float>(0);
                    vc = add(va, vb);
                    rslt = reduce_add(vc);

                    // Place the result scalar into the vector.
                    rslt_v.push(rslt);
                }

            rslt_v = reverse(rslt_v);
            // Add bias.
            rslt_v = add(*bias_p++, rslt_v);
            // Forward gate output to coordinator.
            *outp_p++ = rslt_v;
        }
#endif
    auto n_cycles_after = get_cycles();
    auto cycle_count = n_cycles_after - n_cycles_before;
    printf("Loop cycle count: %llu\n", cycle_count);
}
