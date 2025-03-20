#include "kernel_lib.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

alignas(vector_decl_align) static float state_arr[LSTM_UNITS] = {0};
alignas(vector_decl_align) static float input_arr[INPUT_DIM] = {0};

void gate_half_strm(
    // Input streams.
    input_stream<float> *input_strm,
    input_stream<float> *state_strm,

    // Run Time Parameters.
    const float (&w_state_part)[W_STATE_SIZE / 2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS / 2],
    const int gate_id,

    // Output streams.
    output_pktstream *output_strm) {

    // Set C compatible rounding mode.
    set_rounding(aie::rounding_mode::conv_even);

    // For down gates, enable partial computation flag.
    int partial_computation = gate_id % 2;

    // Setup iterators for weights and bias run time parameters.
    auto w_inp_iter = cbegin_vector<VECTOR_LANES>(w_input);
    auto w_state_iter = cbegin_vector<VECTOR_LANES>(w_state_part);
    auto b_iter = begin(bias);

    // Setup iterators for LSTM state and input.
    auto state_iter = begin_restrict_vector<VECTOR_LANES>(state_arr);
    auto inp_iter = begin_restrict_vector<VECTOR_LANES>(input_arr);

    constexpr int input_blocks = INPUT_DIM / VECTOR_LANES;
    constexpr int state_blocks = W_STATE_COLS / VECTOR_LANES;
    constexpr int strm_width = 4;

    float rslt;
    int strm_data;
    bool strm_tlast;
    uint32 output_strm_id = getPacketid(output_strm, 0);
    vector<float, VECTOR_LANES> va, vb, rslt_v;
    vector<float, strm_width> vs;

    // NOTE: The reason for using two accumulators is to fill the
    // pipeline during the two idle cycles of float MACS, as
    // described in UG1079(v2024.1), Ch 3, Sec "Floating Point
    // Operations", page 54.
    accum<accfloat, VECTOR_LANES> acc1, acc2;

    // During the first row, we have not yet received full state or
    // input. As such, we need to receive the state and inputs 4-byte
    // value at a time and simultaneously do computation and store.
    // -------------------------------------------------------------,
    for (int i = 0; i < state_blocks; i++)
        chess_prepare_for_pipelining {
            // Fill a vector from the state stream.
            for (int j = 0; j < VECTOR_LANES / strm_width; j++) {
                vs = readincr_v<strm_width>(state_strm);
                va = va.insert(j, vs);
            }

            acc1 = mac(acc1, *w_state_iter++, va);
            *state_iter++ = va;
        }

    // Mutiply and accumulate with input-weight matrix.
    for (int i = 0; i < input_blocks; i++)
        chess_prepare_for_pipelining {
            // Fill a vector from the input stream.
            for (int j = 0; j < VECTOR_LANES / strm_width; j++) {
                vs = readincr_v<strm_width>(input_strm);
                va = va.insert(j, vs);
            }

            acc1 = mac(acc1, *w_inp_iter++, va);
            *inp_iter++ = va;
        }

    // Sum all of the accumulator lanes.
    rslt = (*b_iter++) + reduce_add(acc1.to_vector<float>(0));
#ifdef DEBUG
    printf("G_%d [gate ->> strm] = %.5f\n", gate_id, rslt);
#endif

    // Send result over output stream.
    writeHeader(output_strm, 0, output_strm_id);
    writeincr(output_strm, reinterpret_cast<uint32 &>(rslt), true);

    // Reset input and state iterators.
    inp_iter -= input_blocks;
    state_iter -= state_blocks;
    // -------------------------------------------------------------'

    for (int row = 1; row < LSTM_UNITS / 2; row++)
        chess_prepare_for_pipelining {
            if (!partial_computation || (row < (LSTM_UNITS_REAL - LSTM_UNITS / 2))) {

                rslt = *b_iter++;
                acc1 = zeros<accfloat, VECTOR_LANES>();
                acc2 = zeros<accfloat, VECTOR_LANES>();

                // Multiply and accumulate with state weight matrix.
                for (int i = 0; i < state_blocks / 2; i++) {
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc1 = mac(acc1, va, vb);
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc2 = mac(acc2, va, vb);
                }

                // Mutiply and accumulate with input-weight matrix.
                for (int i = 0; i < input_blocks; i++) {
                    acc1 = mac(acc1, *w_inp_iter++, *inp_iter++);
                }

                // Sum all of the accumulator lanes.
                rslt += reduce_add(acc1.to_vector<float>(0));
                rslt += reduce_add(acc2.to_vector<float>(0));

                // Reset input and state iterators.
                inp_iter -= input_blocks;
                state_iter -= state_blocks;

#ifdef DEBUG
                printf("G_%d [gate ->> strm] = %.5f\n", gate_id, rslt);
#endif
                // Send result over output stream.
                writeHeader(output_strm, 0, output_strm_id);
                writeincr(output_strm, reinterpret_cast<uint32 &>(rslt), true);
            }
        }
}
