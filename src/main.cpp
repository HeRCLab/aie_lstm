#include "common_defs.hpp"
#include "data_reader.hpp"
#include "graphs.hpp"
#include <adf.h>

LSTMGraph gr;

int main(void) {
    adf::return_code ret;

    // Read network weights and input data from disk.
    NetworkParams net_params(WEIGHT_DATA_PATH);
    DropbearData dropbear_data(DROPBEAR_DATA_PATH);

    // Initiate graph.
    gr.init();

    // Set gate network parameters.
    for (int i = 0; i < N_GATES; i++) {
        int state_offset = i * W_STATE_SIZE;
        int input_offset = i * W_INPUT_SIZE;
        int bias_offset = i * LSTM_UNITS;
        gr.update(gr.rtp_w_state_left[i],
                  net_params.w_state[state_offset]);
        gr.update(gr.rtp_w_state_right[i],
                  net_params.w_state[state_offset + W_STATE_SIZE / 2]);
        gr.update(gr.rtp_w_input[i],
                  net_params.w_input[input_offset]);
        gr.update(gr.rtp_bias[i],
                  net_params.bias[bias_offset]);
    }

    // Set coordinator network parameters.
    gr.update(gr.rtp_w_dense, *net_params.w_dense);
    gr.update(gr.rtp_b_dense, net_params.b_dense);

    // TODO: Hook gmio.

    return 0;
}
