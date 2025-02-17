#pragma once

#include "common_defs.hpp"
#include <fstream>

using namespace std;

struct NetworkParams {
    float w_state_left[N_GATES * W_STATE_SIZE / 2] = {};
    float w_state_right[N_GATES * W_STATE_SIZE / 2] = {};
    float w_input[N_GATES * W_INPUT_SIZE] = {};
    float bias[N_GATES * LSTM_UNITS] = {};

    float w_dense[LSTM_UNITS] = {};
    float b_dense = 0.0f;

    NetworkParams(const char *weight_file_path) {
        fstream fin;

        fin.open(weight_file_path, ios::binary | ios::in);
        if (!fin) {
            printf("Failed to read network parameters!\n");
            return;
        }

        // Load LSTM gate parameters.
        int offset = 0;
        for (int i = 0; i < N_GATES; i++) {
            // Skip data for previous gate.
            offset = i * (W_STATE_SIZE_REAL + W_INPUT_SIZE_REAL + LSTM_UNITS_REAL) * sizeof(float);

            // Load state weights.
            fin.seekg(offset, fin.beg);
            for (int r = 0; r < LSTM_UNITS_REAL; r++) {
                fin.read(reinterpret_cast<char *>(w_state_left + i * (W_STATE_SIZE / 2) + r * (LSTM_UNITS / 2)),
                         (LSTM_UNITS / 2) * sizeof(float));
                fin.read(reinterpret_cast<char *>(w_state_right + i * (W_STATE_SIZE / 2) + r * (LSTM_UNITS / 2)),
                         (LSTM_UNITS_REAL - LSTM_UNITS / 2) * sizeof(float));
            }
            offset += W_STATE_SIZE_REAL * sizeof(float);

            // Load input weights.
            fin.seekg(offset, fin.beg);
            for (int r = 0; r < LSTM_UNITS_REAL; r++) {
                fin.read(reinterpret_cast<char *>(w_input + i * W_INPUT_SIZE + r * INPUT_DIM),
                         INPUT_DIM_REAL * sizeof(float));
            }
            offset += W_INPUT_SIZE_REAL * sizeof(float);

            // Load lstm biases.
            fin.seekg(offset, fin.beg);
            fin.read(reinterpret_cast<char *>(bias + i * LSTM_UNITS),
                     LSTM_UNITS_REAL * sizeof(float));
        }

        // Load dense layer parameters.
        offset = N_GATES * (W_STATE_SIZE_REAL + W_INPUT_SIZE_REAL + LSTM_UNITS_REAL) * sizeof(float);
        fin.seekg(offset, fin.beg);
        fin.read(reinterpret_cast<char *>(w_dense),
                 LSTM_UNITS_REAL * sizeof(float));
        offset += LSTM_UNITS_REAL * sizeof(float);

        fin.seekg(offset, fin.beg);
        fin.read(reinterpret_cast<char *>(&b_dense), sizeof(float));

        fin.close();
    }
};

struct DropbearData {
    float *vib;
    float *pin;
    float *time;
    int n_samples;

    DropbearData(const char *data_file_path) {
        fstream fin;
        fin.open(data_file_path, ios::binary | ios::in);
        if (!fin) {
            printf("Failed to read dropbear data file!\n");
            return;
        }

        fin.seekg(0, fin.end);
        int file_size = fin.tellg();
        n_samples = file_size / 3 / sizeof(float);

        vib = new float[n_samples];
        pin = new float[n_samples];
        time = new float[n_samples];

        fin.seekg(0, fin.beg);
        fin.read(reinterpret_cast<char *>(vib), n_samples * sizeof(float));
        fin.read(reinterpret_cast<char *>(pin), n_samples * sizeof(float));
        fin.read(reinterpret_cast<char *>(time), n_samples * sizeof(float));

        fin.close();
    }

    ~DropbearData() {
        if (vib)
            delete[] vib;
        if (pin)
            delete[] pin;
        if (time)
            delete[] time;
    }
};
