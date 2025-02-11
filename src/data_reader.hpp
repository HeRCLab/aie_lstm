#pragma once

#include "common_defs.hpp"
#include <fstream>

using namespace std;

struct NetworkParams {
    float w_state[N_GATES * W_STATE_SIZE] = {};
    float w_input[N_GATES * W_INPUT_SIZE] = {};
    float bias[N_GATES * LSTM_UNITS] = {};

    float w_dense[LSTM_UNITS] = {};
    float b_dense = 0.0f;

    NetworkParams(const char *weight_file_path) {
        fstream fin;

        fin.open(weight_file_path, ios::binary | ios::in);
        if (!fin) {
            printf("Failed to read network parameters!");
            return;
        }

        // Load LSTM gate parameters.
        for (int i = 0; i < N_GATES; i++) {
            // Load state weights.
            int offset = i * W_STATE_SIZE * sizeof(float);
            for (int r = 0; r < LSTM_UNITS_NON_EMPTY; r++) {
                fin.read(reinterpret_cast<char *>(w_state + offset + r * LSTM_UNITS),
                         LSTM_UNITS_NON_EMPTY * sizeof(float));
            }

            // Load input weights.
            offset = i * W_INPUT_SIZE * sizeof(float);
            for (int r = 0; r < INPUT_DIM_NON_EMPTY; r++) {
                fin.read(reinterpret_cast<char *>(w_input + offset + r * INPUT_DIM),
                         INPUT_DIM_NON_EMPTY * sizeof(float));
            }

            // Load lstm biases.
            offset = i * LSTM_UNITS;
            fin.read(reinterpret_cast<char *>(bias + offset),
                     LSTM_UNITS_NON_EMPTY * sizeof(float));
        }

        // Load dense layer parameters.
        fin.read(reinterpret_cast<char *>(w_dense),
                 LSTM_UNITS_NON_EMPTY * sizeof(float));
        fin.read(reinterpret_cast<char *>(&b_dense), sizeof(float));

        fin.close();
    }
};

struct DropbearData {
    float *vib;
    float *pin;
    float *time;
    int   n_samples;

    DropbearData(const char *data_file_path) {
        fstream fin;
        fin.open(data_file_path, ios::binary | ios::in);
        if (!fin) {
            printf("Failed to read dropbear data file!");
            return;
        }

        fin.seekg(0, fin.end);
        int file_size = fin.tellg();
        n_samples = file_size / 3 / sizeof(float);

        vib = new float[n_samples];
        pin = new float[n_samples];
        time = new float[n_samples];

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
