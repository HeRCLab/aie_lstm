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
static int iter_count = 0;

// Table to associate packet stream ID with gate index.
static int32 gate_idx_table[N_HALF_GATES];

#define IG_UP_IDX 0
#define IG_DOWN_IDX 1
#define FG_UP_IDX 2
#define FG_DOWN_IDX 3
#define OG_UP_IDX 4
#define OG_DOWN_IDX 5
#define MG_UP_IDX 6
#define MG_DOWN_IDX 7

#define GATE_FIFO_MAX_SIZE 16

// A fifo implementation with circular buffers.
// --------------------------------------------------------,
struct GateFIFO {
    int32 begin = {0}; // Beginning index
    int32 size = {0};  // Current size
    float buff[GATE_FIFO_MAX_SIZE] = {0.0f};
};

inline
float get_gate_result(GateFIFO (&gate_fifos)[N_HALF_GATES],
                      input_pktstream *strm,
                      int32 gate_idx) {
    float rslt;
    int32 strm_head, strm_val, gidx;
    GateFIFO *fifo = &gate_fifos[gate_idx];

    if (fifo->size > 0) {
        rslt = fifo->buff[fifo->begin];

        // As we remove one entry from fifo:
        // - Beginning index increase by one with warping.
        // - Size will decrease by one.
        fifo->begin = (fifo->begin + 1) % GATE_FIFO_MAX_SIZE;
        fifo->size--;

#ifdef DEBUG
        printf("G_%d [fifo ->> output] = %.5f | fifo_size = %d\n", gate_idx, rslt, fifo->size);
#endif
        return rslt;
    } else {
        while (true) {
            // Read header and value from stream.
            strm_head = readincr(strm);
            strm_val = readincr(strm);
            rslt = reinterpret_cast<float &>(strm_val);

            // Extract gate index of the new data.
            gidx = gate_idx_table[strm_head & 0x1f];
            if (gidx == gate_idx) {
#ifdef DEBUG
                printf("G_%d [strm ->> output] = %.5f\n", gate_idx, rslt);
#endif
                // Return result.
                return rslt;
            } else {
                // Select appropriate fifo for latest packet.
                fifo = &gate_fifos[gidx];

                // Add new entry at the end of fifo. ⚠️ NOTE that we are
                // ignoring the possibility of fifo overflow!
                auto end = (fifo->begin + fifo->size) % GATE_FIFO_MAX_SIZE;
                fifo->buff[end] = rslt;
                fifo->size++;
#ifdef DEBUG
                printf("G_%d [fifo <<- strm] = %.5f | fifo_size = %d\n", gidx, rslt, fifo->size);
#endif
            }
        }
    }
}
// --------------------------------------------------------'

void coordinator_4x2_strm(
    // Input gate multiplexed packet streams. We are using two streams
    // because down gates produce data at faster rates at the tail
    // end.
    input_pktstream *gate_strm,

    // Run Time Parameters.
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,

    // Output streams.
    output_stream<float> *state_out_strm,
    output_stream<float> *output_strm) {

    // Set C compatible rounding mode.
    set_rounding(aie::rounding_mode::conv_even);

    int32 packet_id, gate_idx;
    float sum, out_lstm, t1, t2, next_state;
    float igu, igd, fgu, fgd, ogu, ogd, mgu, mgd;

    GateFIFO gate_fifos[N_HALF_GATES];

    vector<float, VECTOR_LANES> v1, v2, v3;
    accum<accfloat, VECTOR_LANES> acc, acc_dense;

    // Create pointers for iterating up gates and down gates.
    float *__restrict state_p = state;
    float *__restrict state_p_down = &state[LSTM_UNITS / 2];
    const float *__restrict w_dense_p = w_dense;
    const float *__restrict w_dense_p_down = &w_dense[LSTM_UNITS / 2];

    // Send initial state down to the gates.
    if (!iter_count++) {
        // Populate gate ID table.
        for (int i = 0; i < N_HALF_GATES; i++) {
            packet_id = getPacketid(gate_strm, i);
            gate_idx_table[packet_id & 0x1f] = i;
        }

        // Send initial state.
        for (int i = 0; i < LSTM_UNITS; i++) {
            writeincr(state_out_strm, *state_p++);
        }

        // Reset state pointer to beginning.
        state_p -= LSTM_UNITS;
    }

    // Calculate next-state and output from upper half gates.
    // ----------------------------------------------------------------,
    sum = bias_dense;
    acc_dense = zeros<accfloat, VECTOR_LANES>();
    for (int i = 0; i < LSTM_UNITS / 2; i++)
        chess_prepare_for_pipelining {
            // Always collect results from up gates.
            igu = get_gate_result(gate_fifos, gate_strm, IG_UP_IDX);
            fgu = get_gate_result(gate_fifos, gate_strm, FG_UP_IDX);
            ogu = get_gate_result(gate_fifos, gate_strm, OG_UP_IDX);
            mgu = get_gate_result(gate_fifos, gate_strm, MG_UP_IDX);

            // Collect results from down gates if computation is non-null.
            if (i < (LSTM_UNITS_REAL - (LSTM_UNITS / 2))) {
                igd = get_gate_result(gate_fifos, gate_strm, IG_DOWN_IDX);
                fgd = get_gate_result(gate_fifos, gate_strm, FG_DOWN_IDX);
                ogd = get_gate_result(gate_fifos, gate_strm, OG_DOWN_IDX);
                mgd = get_gate_result(gate_fifos, gate_strm, MG_DOWN_IDX);
            }

            // Process results from upper four gates.
            // ---------------------------------------------------,
            // Calculate next state.
            t1 = (*state_p) * SIGM_SCALAR(fgu);
            t2 = SIGM_SCALAR(igu) * TANH_SCALAR(mgu);
            next_state = t1 + t2;

            // Save next state.
            *state_p++ = next_state;

            // Calculate LSTM output.
            t1 = SIGM_SCALAR(ogu);
            t2 = SIGM_SCALAR(next_state);
            out_lstm = t1 + t2;
            sum += (out_lstm * (*w_dense_p++));
            // ----------------------------------------------------'

            // Process results from lower four gates.
            // ---------------------------------------------------,
            if (i < (LSTM_UNITS_REAL - (LSTM_UNITS / 2))) {
                // Calculate next state.
                t1 = (*state_p_down) * SIGM_SCALAR(fgd);
                t2 = SIGM_SCALAR(igd) * TANH_SCALAR(mgd);
                next_state = t1 + t2;

                // Save next state.
                *state_p_down++ = next_state;

                // Calculate LSTM output.
                t1 = SIGM_SCALAR(ogd);
                t2 = SIGM_SCALAR(next_state);
                out_lstm = t1 + t2;
                sum += (out_lstm * (*w_dense_p_down++));
            } else {
                // Skip computations at the tail.
                *state_p_down++ = 0.0f;
                w_dense_p_down++;
            }
            // ----------------------------------------------------'
        }

    // Send coordinator result through output stream.
    writeincr(output_strm, sum);

    // Push next state back to the gates for next iteration. The next
    // state will be queued in FIFO outgoing stream connections.
    state_p = state;
    for (int i = 0; i < LSTM_UNITS; i++) {
        writeincr(state_out_strm, *state_p++);
    }

#ifdef DEBUG
    print_state(state);
#endif

    printf("Lstm coord output: %.12f\n", sum);
}
