#include "common_defs.hpp"
#include "data_reader.hpp"
#include "graphs.hpp"
#include <adf.h>

#if defined(VARIANT_4X1_BUFF)
LSTMGraph_4x1_buff gr;
#elif defined(VARIANT_4X2_BUFF)
LSTMGraph_4x2_buff gr;
#elif defined(VARIANT_4X2_STRM)
LSTMGraph_4x2_strm gr;
#else
static_assert(0, "VARIANT_XXXX preprocessor directive missing!");
#endif

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    adf::return_code ret;

#if defined(VARIANT_4X1_BUFF)
    printf("Program compiled with VARIANT=4x1_BUFF\n");
#elif defined(VARIANT_4X2_BUFF)
    printf("Program compiled with VARIANT=4x2_BUFF\n");
#elif defined(VARIANT_4X2_STRM)
    printf("Program compiled with VARIANT=4x2_STRM\n");
#else
    static_assert(0, "VARIANT_XXXX preprocessor directive missing!");
#endif

    // Read network weights and input data from disk.
#if defined(NO_DISK_IO)
    printf("Reading weights from array ...\n");
    NetworkParams net_params;
    printf("Reading data from array ...\n");
    DropbearData dropbear_data;
#else
    printf("Reading weights from disk ...\n");
    NetworkParams net_params(WEIGHT_DATA_PATH);
    printf("Reading data from disk ...\n");
    DropbearData dropbear_data(DROPBEAR_DATA_PATH);
#endif

    // Initiate graph.
    printf("Initiating graph ...\n");
    gr.init();

#if defined(VARIANT_4X1_BUFF)
    // Set gate network parameters.
    for (int i = 0; i < N_GATES; i++) {
        int state_offset = i * W_STATE_SIZE / 2;
        int input_offset = i * W_INPUT_SIZE;
        int bias_offset = i * LSTM_UNITS;
        gr.update(gr.rtp_w_state_left[i], &net_params.w_state_left[state_offset], W_STATE_SIZE / 2);
        gr.update(gr.rtp_w_state_right[i], &net_params.w_state_right[state_offset], W_STATE_SIZE / 2);
        gr.update(gr.rtp_w_input[i], &net_params.w_input[input_offset], W_INPUT_SIZE);
        gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset], LSTM_UNITS);
    }
#elif defined(VARIANT_4X2_BUFF)
    // Set gate network parameters.
    for (int i = 0; i < N_HALF_GATES; i++) {
        int state_offset = static_cast<int>(i / 2) * W_STATE_SIZE;
        int input_offset = static_cast<int>(i / 2) * W_INPUT_SIZE;
        int bias_offset = static_cast<int>(i / 2) * LSTM_UNITS;

        gr.update(gr.rtp_w_input[i], &net_params.w_input[input_offset], W_INPUT_SIZE);

        if (i % 2 == 0) {
            // Up gates.
            gr.update(gr.rtp_w_state[i], &net_params.w_state[state_offset], W_STATE_SIZE / 2);
            gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset], LSTM_UNITS / 2);
            gr.update(gr.partial_computation_flag[i], (int)0);
        } else {
            // Down gates.
            gr.update(gr.rtp_w_state[i], &net_params.w_state[state_offset + W_STATE_SIZE / 2], W_STATE_SIZE / 2);
            gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset + LSTM_UNITS / 2], LSTM_UNITS / 2);
            gr.update(gr.partial_computation_flag[i], (int)1);
        }
    }
#elif defined(VARIANT_4X2_STRM)
    // Set gate network parameters.
    for (int i = 0; i < N_HALF_GATES; i++) {
        int state_offset = static_cast<int>(i / 2) * W_STATE_SIZE;
        int input_offset = static_cast<int>(i / 2) * W_INPUT_SIZE;
        int bias_offset = static_cast<int>(i / 2) * LSTM_UNITS;

        gr.update(gr.rtp_w_input[i], &net_params.w_input[input_offset], W_INPUT_SIZE);
        gr.update(gr.gate_id[i], (int)i);

        if (i % 2 == 0) {
            // Up gates.
            gr.update(gr.rtp_w_state[i], &net_params.w_state[state_offset], W_STATE_SIZE / 2);
            gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset], LSTM_UNITS / 2);
        } else {
            // Down gates.
            gr.update(gr.rtp_w_state[i], &net_params.w_state[state_offset + W_STATE_SIZE / 2], W_STATE_SIZE / 2);
            gr.update(gr.rtp_bias[i], &net_params.bias[bias_offset + LSTM_UNITS / 2], LSTM_UNITS / 2);
        }
    }
#else
    static_assert(0, "VARIANT_XXXX preprocessor directive missing!");
#endif

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
    constexpr int input_buff_size = OUTPUT_SAMPLES * INPUT_DIM * sizeof(float);
    constexpr int output_buff_size = OUTPUT_SAMPLES * OUTPUT_BLOCK_SIZE * sizeof(float);

    printf("Output samples: %d\n", OUTPUT_SAMPLES);
    printf("Input buff size: %d\n", input_buff_size);
    printf("Output buff size: %d\n", output_buff_size);

    // Allocate buffers for GMIO communication.
    float *input_buff = reinterpret_cast<float *>(GMIO::malloc(input_buff_size));
    float *output_buff = reinterpret_cast<float *>(GMIO::malloc(output_buff_size));

    // Fill input buffer with zeros.
    memset(input_buff, 0, input_buff_size);

    // Copy input data.
    for (int i = 0; i < OUTPUT_SAMPLES; i++) {
        memcpy(input_buff + i * INPUT_DIM,
               dropbear_data.vib + i,
               INPUT_DIM_REAL * sizeof(float));
    }
    // ---------------------------------------------------------------'

    // Attach GMIO buffers.
    printf("Attaching GMIO buffers ..\n");
    gr.gmio_in.gm2aie_nb(input_buff, input_buff_size);
    gr.gmio_out.aie2gm_nb(output_buff, output_buff_size);

    // Run the graph.
    printf("Starting graph ...\n");
    gr.run(OUTPUT_SAMPLES);
    gr.gmio_out.wait();
    gr.wait();
    printf("Graph done.\n");

    // Print data to stdout in comparable format to ref implementation.
    for (int i = 0; i < OUTPUT_SAMPLES + INPUT_DIM_REAL - 1; i++) {
        if (i < (INPUT_DIM_REAL - 1)) {
            printf("#%.12f %.12f\n", dropbear_data.time[i], 0.0f);
        } else {
            printf("#%.12f %.12f\n",
                   dropbear_data.time[i],
                   *(output_buff + (i - INPUT_DIM_REAL + 1) * OUTPUT_BLOCK_SIZE));
        }
    }

    // Free GMIO buffers.
    GMIO::free(input_buff);
    GMIO::free(output_buff);
    gr.end();

    return 0;
}
#endif
