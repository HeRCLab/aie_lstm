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

#if defined(GATE_HALF_LOOP_V9) || defined(GATE_HALF_LOOP_V10)
    // Setup circular iterators for input, output and state buffers.
    const float *__restrict inp_p = (const float *)input.data();
    const float *__restrict state_p = (const float *)state.data();
    float *__restrict outp_p = (float *)output.data();
    const float *__restrict w_state_p = (const float *)w_state_part;
    const float *__restrict w_inp_p = (const float *)w_input;
    const float *__restrict bias_p = (const float *)bias;

#if defined(GATE_HALF_LOOP_V10)
    const float *__restrict inp_start_p = (const float *)input.data();
    const float *__restrict state_start_p = (const float *)state.data();
    float *__restrict outp_start_p = (float *)output.data();
    const float *__restrict w_state_start_p = (const float *)w_state_part;
    const float *__restrict w_inp_start_p = (const float *)w_input;
    const float *__restrict bias_start_p = (const float *)bias;
#endif
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
    vector<float, VECTOR_LANES> va, vb, rslt_v;

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
#elif defined(GATE_HALF_LOOP_V2)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++) {
        acc1 = zeros<accfloat, VECTOR_LANES>();
        acc2 = zeros<accfloat, VECTOR_LANES>();

        // Multiply and accumulate with state weight matrix.
        for (int i = 0; i < W_STATE_COLS / VECTOR_LANES / 2; i++)
            chess_prepare_for_pipelining {
                va = *w_state_iter++;
                vb = *state_iter++;
                acc1 = mac(acc1, va, vb);
                va = *w_state_iter++;
                vb = *state_iter++;
                acc2 = mac(acc2, va, vb);
            }

        // Mutiply and accumulate with input-weight matrix.
        for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++)
            chess_prepare_for_pipelining {
                acc1 = mac(acc1, *w_inp_iter++, *inp_iter++);
            }

        // Sum all of the accumulator lanes.
        sum1 = reduce_add(acc1.to_vector<float>(0));
        sum2 = reduce_add(acc2.to_vector<float>(0));
        rslt = sum1 + sum2;

        // Reset input and state iterators.
        inp_iter -= input_blocks;
        state_iter -= state_blocks;

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
#elif defined(GATE_HALF_LOOP_V3)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++) {
        acc1 = zeros<accfloat, VECTOR_LANES>();
        acc2 = zeros<accfloat, VECTOR_LANES>();

        // Multiply and accumulate with state weight matrix.
        va = *w_state_iter;
        vb = *state_iter;
        for (int i = 0; i < W_STATE_COLS / VECTOR_LANES; i++)
            chess_prepare_for_pipelining {
                acc1 = mac(acc1, va, vb);
                va = *(++w_state_iter);
                vb = *(++state_iter);
            }

        // Mutiply and accumulate with input-weight matrix.
        va = *w_inp_iter;
        vb = *inp_iter;
        for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++)
            chess_prepare_for_pipelining {
                acc1 = mac(acc1, va, vb);
                va = *(++w_inp_iter);
                vb = *(++inp_iter);
            }

        // Sum all of the accumulator lanes.
        rslt = reduce_add(acc1.to_vector<float>(0));

        // Reset input and state iterators.
        inp_iter -= input_blocks;
        state_iter -= state_blocks;

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
#elif defined(GATE_HALF_LOOP_V4)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row_blocks = 0; row_blocks < (LSTM_UNITS / 2 / VECTOR_LANES); row_blocks++) {

        acc1 = zeros<accfloat, VECTOR_LANES>();
        // Compute a block of rows.
        for (int r = 0; r < VECTOR_LANES; r++) {
            // Multiply and accumulate with state weight matrix.
            va = *w_state_iter;
            vb = *state_iter;
            for (int i = 0; i < W_STATE_COLS / VECTOR_LANES; i++)
                chess_prepare_for_pipelining {
                    acc1 = mac(acc1, va, vb);
                    va = *(++w_state_iter);
                    vb = *(++state_iter);
                }

            // Mutiply and accumulate with input-weight matrix.
            va = *w_inp_iter;
            vb = *inp_iter;
            for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++)
                chess_prepare_for_pipelining {
                    acc1 = mac(acc1, va, vb);
                    va = *(++w_inp_iter);
                    vb = *(++inp_iter);
                }

            // Sum all of the accumulator lanes.
            rslt = reduce_add(acc1.to_vector<float>(0));

            // Place the result scalar into the vector.
            rslt_v.set(rslt, r);

            // Reset accumulators.
            acc1 = zeros<accfloat, VECTOR_LANES>();

            // Reset input and state iterators.
            inp_iter -= input_blocks;
            state_iter -= state_blocks;
        }

        // Add bias.
        rslt_v = add(*b_iter, rslt_v);

        // Forward gate output to coordinator.
        *out_iter = rslt_v;

        b_iter++;
        out_iter++;
        rslt_v = zeros<float, VECTOR_LANES>();
    }
#elif defined(GATE_HALF_LOOP_V5)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row_blocks = 0; row_blocks < (LSTM_UNITS / 2 / VECTOR_LANES); row_blocks++) {

        acc1 = zeros<accfloat, VECTOR_LANES>();

        // Compute a block of rows.
        for (int r = 0; r < VECTOR_LANES; r++) {
            // Multiply and accumulate with state weight matrix.
            va = *w_state_iter;
            vb = *state_iter;
            for (int i = 0; i < W_STATE_COLS / VECTOR_LANES; i++)
                chess_prepare_for_pipelining chess_flatten_loop {
                    acc1 = mac(acc1, va, vb);
                    va = *(++w_state_iter);
                    vb = *(++state_iter);
                }

            // Mutiply and accumulate with input-weight matrix.
            va = *w_inp_iter;
            vb = *inp_iter;
            for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++)
                chess_prepare_for_pipelining chess_flatten_loop {
                    acc1 = mac(acc1, va, vb);
                    va = *(++w_inp_iter);
                    vb = *(++inp_iter);
                }

            // Sum all of the accumulator lanes.
            rslt = reduce_add(acc1.to_vector<float>(0));

            // Place the result scalar into the vector.
            rslt_v.set(rslt, r);

            // Reset accumulators.
            acc1 = zeros<accfloat, VECTOR_LANES>();

            // Reset input and state iterators.
            inp_iter -= input_blocks;
            state_iter -= state_blocks;
        }

        // Add bias.
        rslt_v = add(*b_iter, rslt_v);

        // Forward gate output to coordinator.
        *out_iter = rslt_v;

        b_iter++;
        out_iter++;
        rslt_v = zeros<float, VECTOR_LANES>();
    }
#elif defined(GATE_HALF_LOOP_V6)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row_blocks = 0; row_blocks < (LSTM_UNITS / 2 / VECTOR_LANES); row_blocks++) {

        acc1 = zeros<accfloat, VECTOR_LANES>();

        // Compute a block of rows.
        for (int r = 0; r < VECTOR_LANES; r++)
            chess_prepare_for_pipelining {
                // Multiply and accumulate with state weight matrix.
                va = *w_state_iter;
                vb = *state_iter;
                for (int i = 0; i < W_STATE_COLS / VECTOR_LANES; i++)
                    chess_flatten_loop {
                        acc1 = mac(acc1, va, vb);
                        va = *(++w_state_iter);
                        vb = *(++state_iter);
                    }

                // Mutiply and accumulate with input-weight matrix.
                va = *w_inp_iter;
                vb = *inp_iter;
                for (int i = 0; i < INPUT_DIM / VECTOR_LANES; i++)
                    chess_flatten_loop {
                        acc1 = mac(acc1, va, vb);
                        va = *(++w_inp_iter);
                        vb = *(++inp_iter);
                    }

                // Sum all of the accumulator lanes.
                rslt = reduce_add(acc1.to_vector<float>(0));

                // Place the result scalar into the vector.
                rslt_v.set(rslt, r);

                // Reset accumulators.
                acc1 = zeros<accfloat, VECTOR_LANES>();

                // Reset input and state iterators.
                inp_iter -= input_blocks;
                state_iter -= state_blocks;
            }

        // Add bias.
        rslt_v = add(*b_iter, rslt_v);

        // Forward gate output to coordinator.
        *out_iter = rslt_v;

        b_iter++;
        out_iter++;
        rslt_v = zeros<float, VECTOR_LANES>();
    }
#elif defined(GATE_HALF_LOOP_V7)
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row_blocks = 0; row_blocks < (LSTM_UNITS / 2 / VECTOR_LANES); row_blocks++) {

        acc1 = zeros<accfloat, VECTOR_LANES>();
        acc2 = zeros<accfloat, VECTOR_LANES>();

        // Compute a block of rows.
        for (int r = 0; r < VECTOR_LANES; r++)
            chess_prepare_for_pipelining {
                // Multiply and accumulate with state weight matrix.
                va = *w_state_iter;
                vb = *state_iter;
                for (int i = 0; i < (W_STATE_COLS / VECTOR_LANES / 2); i++)
                    chess_flatten_loop {
                        acc1 = mac(acc1, va, vb);

                        va = *(++w_state_iter);
                        vb = *(++state_iter);
                        acc2 = mac(acc2, va, vb);

                        va = *(++w_state_iter);
                        vb = *(++state_iter);
                    }

                // Mutiply and accumulate with input-weight matrix.
                va = *w_inp_iter;
                vb = *inp_iter;
                for (int i = 0; i < (INPUT_DIM / VECTOR_LANES / 2); i++)
                    chess_flatten_loop {
                        acc1 = mac(acc1, va, vb);

                        va = *(++w_inp_iter);
                        vb = *(++inp_iter);
                        acc2 = mac(acc2, va, vb);

                        va = *(++w_state_iter);
                        vb = *(++inp_iter);
                    }

                // Sum all of the accumulator lanes.
                sum1 = reduce_add(acc1.to_vector<float>(0));
                sum2 = reduce_add(acc2.to_vector<float>(0));
                rslt = sum1 + sum2;

                // Place the result scalar into the vector.
                rslt_v.set(rslt, r);

                // Reset accumulators.
                acc1 = zeros<accfloat, VECTOR_LANES>();
                acc2 = zeros<accfloat, VECTOR_LANES>();

                // Reset input and state iterators.
                inp_iter -= input_blocks;
                state_iter -= state_blocks;
            }

        // Add bias.
        rslt_v = add(*b_iter, rslt_v);

        // Forward gate output to coordinator.
        *out_iter = rslt_v;

        b_iter++;
        out_iter++;
        rslt_v = zeros<float, VECTOR_LANES>();
    }
#elif defined(GATE_HALF_LOOP_V8)
    // rslt_v = zeros<float, VECTOR_LANES>();
    acc1 = zeros<accfloat, VECTOR_LANES>();
    acc2 = zeros<accfloat, VECTOR_LANES>();

    constexpr int n_row_blocks = LSTM_UNITS / 2 / VECTOR_LANES;
    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining chess_loop_count(n_row_blocks) {

            // Compute a block of rows.
            for (int r = 0; r < VECTOR_LANES; r++)
                chess_flatten_loop chess_loop_count(VECTOR_LANES) {
                    // Multiply and accumulate with state weight matrix.
                    va = *w_state_iter;
                    vb = *state_iter;
                    constexpr int n_state_loop = (W_STATE_COLS / VECTOR_LANES / 2);
                    for (int i = 0; i < n_state_loop; i++)
                        chess_flatten_loop chess_loop_count(n_state_loop) {
                            acc1 = mac(acc1, va, vb);

                            va = *(++w_state_iter);
                            vb = *(++state_iter);
                            acc2 = mac(acc2, va, vb);

                            va = *(++w_state_iter);
                            vb = *(++state_iter);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    va = *w_inp_iter;
                    vb = *inp_iter;
                    constexpr int n_input_loop = (INPUT_DIM / VECTOR_LANES / 2);
                    for (int i = 0; i < n_input_loop; i++)
                        chess_flatten_loop chess_loop_count(n_input_loop) {
                            acc1 = mac(acc1, va, vb);

                            va = *(++w_inp_iter);
                            vb = *(++inp_iter);
                            acc2 = mac(acc2, va, vb);

                            va = *(++w_inp_iter);
                            vb = *(++inp_iter);
                        }

                    // Sum all of the accumulator lanes.
                    sum1 = reduce_add(acc1.to_vector<float>(0));
                    sum2 = reduce_add(acc2.to_vector<float>(0));
                    rslt = sum1 + sum2;

                    // Place the result scalar into the vector.
                    rslt_v.set(rslt, r);

                    // Reset accumulators.
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();

                    // Reset input and state iterators.
                    inp_iter -= input_blocks;
                    state_iter -= state_blocks;
                }

            // Add bias.
            rslt_v = add(*b_iter, rslt_v);

            // Forward gate output to coordinator.
            *out_iter = rslt_v;

            b_iter++;
            out_iter++;
            // rslt_v = zeros<float, VECTOR_LANES>();
        }
#elif defined(GATE_HALF_LOOP_V9)
    constexpr int n_state_loop = (W_STATE_COLS / VECTOR_LANES / 2);
    constexpr int n_input_loop = (INPUT_DIM / VECTOR_LANES / 2);
    constexpr int n_row_blocks = LSTM_UNITS / 2 / VECTOR_LANES;

    acc1 = zeros<accfloat, VECTOR_LANES>();
    acc2 = zeros<accfloat, VECTOR_LANES>();
    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining {

            // Compute a block of rows.
            for (int r = 0; r < VECTOR_LANES; r++)
                chess_flatten_loop chess_loop_count(VECTOR_LANES) {
                    // Multiply and accumulate with state weight matrix.
                    va = load_v<VECTOR_LANES>(w_state_p);
                    vb = load_v<VECTOR_LANES>(state_p);
                    for (int i = 0; i < n_state_loop; i++)
                        chess_flatten_loop chess_loop_count(n_state_loop) {
                            acc1 = mac(acc1, va, vb);

                            w_state_p += VECTOR_LANES;
                            va = load_v<VECTOR_LANES>(w_state_p);
                            state_p += VECTOR_LANES;
                            vb = load_v<VECTOR_LANES>(state_p);
                            acc2 = mac(acc2, va, vb);

                            w_state_p += VECTOR_LANES;
                            va = load_v<VECTOR_LANES>(w_state_p);
                            state_p += VECTOR_LANES;
                            vb = load_v<VECTOR_LANES>(state_p);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    va = load_v<VECTOR_LANES>(w_inp_p);
                    vb = load_v<VECTOR_LANES>(inp_p);
                    for (int i = 0; i < n_input_loop; i++)
                        chess_flatten_loop chess_loop_count(n_input_loop) {
                            acc1 = mac(acc1, va, vb);

                            w_inp_p += VECTOR_LANES;
                            va = load_v<VECTOR_LANES>(w_inp_p);
                            inp_p += VECTOR_LANES;
                            vb = load_v<VECTOR_LANES>(inp_p);
                            acc2 = mac(acc2, va, vb);

                            w_inp_p += VECTOR_LANES;
                            va = load_v<VECTOR_LANES>(w_inp_p);
                            inp_p += VECTOR_LANES;
                            vb = load_v<VECTOR_LANES>(inp_p);
                        }

                    // Sum all of the accumulator lanes.
                    sum1 = reduce_add(acc1.to_vector<float>(0));
                    sum2 = reduce_add(acc2.to_vector<float>(0));
                    rslt = sum1 + sum2;

                    // Place the result scalar into the vector.
                    rslt_v.set(rslt, r);

                    // Reset state and input pointers.
                    inp_p -= (VECTOR_LANES * input_blocks);
                    state_p -= (VECTOR_LANES * state_blocks);

                    // Reset accumulators.
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();
                }

            // Add bias.
            va = load_v<VECTOR_LANES>(bias_p);

            // Forward gate output to coordinator.
            rslt_v = add(va, rslt_v);
            store_v(outp_p, rslt_v);

            // Increment bias pointer.
            bias_p += VECTOR_LANES;
        }
#elif defined(GATE_HALF_LOOP_V10)
    // rslt_v = zeros<float, VECTOR_LANES>();
    constexpr int size_w_state = W_STATE_SIZE / 2;
    constexpr int size_w_inp = W_INPUT_SIZE;

    constexpr int size_state = LSTM_UNITS;
    constexpr int size_inp = INPUT_DIM;

    constexpr int size_bias = LSTM_UNITS / 2;
    constexpr int size_outp = LSTM_UNITS / 2;

    constexpr int n_state_loop = (W_STATE_COLS / VECTOR_LANES / 2);
    constexpr int n_input_loop = (INPUT_DIM / VECTOR_LANES / 2);
    constexpr int n_row_blocks = LSTM_UNITS / 2 / VECTOR_LANES;

    acc1 = zeros<accfloat, VECTOR_LANES>();
    acc2 = zeros<accfloat, VECTOR_LANES>();
    for (int row_block = 0; row_block < n_row_blocks; row_block++)
        chess_prepare_for_pipelining chess_loop_count(n_row_blocks) {

            // Compute a block of rows.
            for (int r = 0; r < VECTOR_LANES; r++)
                chess_flatten_loop chess_loop_count(VECTOR_LANES) {
                    // Multiply and accumulate with state weight matrix.
                    va = load_v<VECTOR_LANES>(w_state_p);
                    cyclic_add(w_state_p, VECTOR_LANES, w_state_start_p, size_w_state);
                    vb = load_v<VECTOR_LANES>(state_p);
                    cyclic_add(state_p, VECTOR_LANES, state_start_p, size_state);
                    for (int i = 0; i < n_state_loop; i++)
                        chess_flatten_loop chess_loop_count(n_state_loop) {
                            acc1 = mac(acc1, va, vb);

                            va = load_v<VECTOR_LANES>(w_state_p);
                            cyclic_add(w_state_p, VECTOR_LANES, w_state_start_p, size_w_state);
                            vb = load_v<VECTOR_LANES>(state_p);
                            cyclic_add(state_p, VECTOR_LANES, state_start_p, size_state);
                            acc2 = mac(acc2, va, vb);

                            va = load_v<VECTOR_LANES>(w_state_p);
                            cyclic_add(w_state_p, VECTOR_LANES, w_state_start_p, size_w_state);
                            vb = load_v<VECTOR_LANES>(state_p);
                            cyclic_add(state_p, VECTOR_LANES, state_start_p, size_state);
                        }

                    // Mutiply and accumulate with input-weight matrix.
                    va = load_v<VECTOR_LANES>(w_inp_p);
                    cyclic_add(w_inp_p, VECTOR_LANES, w_inp_start_p, size_w_inp);
                    vb = load_v<VECTOR_LANES>(inp_p);
                    cyclic_add(inp_p, VECTOR_LANES, inp_start_p, size_inp);
                    for (int i = 0; i < n_input_loop; i++)
                        chess_flatten_loop chess_loop_count(n_input_loop) {
                            acc1 = mac(acc1, va, vb);

                            va = load_v<VECTOR_LANES>(w_inp_p);
                            cyclic_add(w_inp_p, VECTOR_LANES, w_inp_start_p, size_w_inp);
                            vb = load_v<VECTOR_LANES>(inp_p);
                            cyclic_add(inp_p, VECTOR_LANES, inp_start_p, size_inp);
                            acc2 = mac(acc2, va, vb);

                            va = load_v<VECTOR_LANES>(w_inp_p);
                            cyclic_add(w_inp_p, VECTOR_LANES, w_inp_start_p, size_w_inp);
                            vb = load_v<VECTOR_LANES>(inp_p);
                            cyclic_add(inp_p, VECTOR_LANES, inp_start_p, size_inp);
                        }

                    // Sum all of the accumulator lanes.
                    sum1 = reduce_add(acc1.to_vector<float>(0));
                    sum2 = reduce_add(acc2.to_vector<float>(0));
                    rslt = sum1 + sum2;

                    // Place the result scalar into the vector.
                    rslt_v.set(rslt, r);

                    // Reset accumulators.
                    acc1 = zeros<accfloat, VECTOR_LANES>();
                    acc2 = zeros<accfloat, VECTOR_LANES>();
                }

            // Add bias.
            va = load_v<VECTOR_LANES>(bias_p);
            cyclic_add(bias_p, VECTOR_LANES, bias_start_p, size_bias);
            rslt_v = add(va, rslt_v);

            // Forward gate output to coordinator.
            store_v(outp_p, rslt_v);
            cyclic_add(outp_p, VECTOR_LANES, outp_start_p, size_outp);
        }
#endif
    auto n_cycles_after = get_cycles();
    auto cycle_count = n_cycles_after - n_cycles_before;
    printf("Loop cycle count: %llu\n", cycle_count);
}
