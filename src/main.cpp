#include "common_defs.hpp"
#include "data_reader.hpp"
#include "graphs.hpp"
#include <adf.h>

LSTMGraph gr;

#if defined (__AIESIM__) || defined(__X86SIM__)
int main(void) {
    adf::return_code ret;

    // Read network weights and input data from disk.
    printf("Reading weights ...\n");
    NetworkParams net_params(WEIGHT_DATA_PATH);
    printf("Reading data ...\n");
    DropbearData dropbear_data(DROPBEAR_DATA_PATH);

    // int n_sample_bytes = dropbear_data.n_samples * sizeof(float);
    constexpr int n_sample_bytes = N_TEST_SAMPLES * sizeof(float);

    // Initiate graph.
    printf("Initiating graph ...\n");
    gr.init();

    // Set gate network parameters.
    for (int i = 0; i < N_GATES; i++) {
        int state_offset = i * W_STATE_SIZE;
        int input_offset = i * W_INPUT_SIZE;
        int bias_offset = i * LSTM_UNITS;
        gr.update(gr.rtp_w_state_left[i], &net_params.w_state[state_offset], W_STATE_SIZE / 2);
        gr.update(gr.rtp_w_state_right[i], &net_params.w_state[state_offset + W_STATE_SIZE / 2], W_STATE_SIZE / 2);
        gr.update(gr.rtp_w_input[i], &net_params.w_input[input_offset], W_INPUT_SIZE);
        gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset], LSTM_UNITS);
    }

    // Set coordinator network parameters.
    gr.update(gr.rtp_w_dense, net_params.w_dense, LSTM_UNITS);
    gr.update(gr.rtp_b_dense, net_params.b_dense);

    // Setup GMIO buffers.
    //
    // NOTE: Each graph iteration will take as input an overlapping
    // window of INPUT_DIM input samples. Although AIE engine supports
    // 'margins' in buffers, they have an alignment requirement of
    // 32. I don't think our current data format can be shaped to take
    // advantage of that. Rather, we will do the sliding-window
    // copying in the input buffer itself.
    // ---------------------------------------------------------------,
    constexpr int usable_sample_size = N_TEST_SAMPLES - INPUT_DIM + 1;
    constexpr int input_buff_size = usable_sample_size * INPUT_DIM * sizeof(float);
    constexpr int output_buff_size = usable_sample_size * OUTPUT_BLOCK_SIZE * sizeof(float);

    printf("Usable sample size: %d\n", usable_sample_size);
    printf("Input buff size: %d\n", input_buff_size);
    printf("Output buff size: %d\n", output_buff_size);

    // Allocate buffers for GMIO communication.
    float *input_buff = reinterpret_cast<float *>(GMIO::malloc(input_buff_size));
    float *output_buff = reinterpret_cast<float *>(GMIO::malloc(output_buff_size));

    // Copy input data.
    for (int i = 0; i < usable_sample_size; i++) {
      memcpy(input_buff + i * INPUT_DIM * sizeof(float),
	     dropbear_data.vib + i * sizeof(float),
	     INPUT_DIM * sizeof(float));
    }
    // ---------------------------------------------------------------'

    // Attach GMIO buffers.
    gr.gmio_in.gm2aie_nb(input_buff, input_buff_size);
    gr.gmio_out.aie2gm_nb(output_buff, output_buff_size);

    // Run the graph.
    gr.run(usable_sample_size);
    gr.gmio_out.wait();
    gr.wait();

    // TODO: Write the output to file.

    // Free GMIO buffers.
    GMIO::free(input_buff);
    GMIO::free(output_buff);
    gr.end();

    return 0;
}
#endif
