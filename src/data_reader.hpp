#pragma once

#include "common_defs.hpp"
#if defined(NO_DISK_IO)
#include "data/dropbear.hpp"
#include "data/weights.hpp"
#include <cstdio>
#include <cstring>
#else
#include <fstream>
#endif

using namespace std;

struct NetworkParams {
    /* # Network parameter layout in disk / array:

       | i-gate-state-weights | i-gate-input-weights | i-gate-bias |
       | f-gate-state-weights | f-gate-input-weights | f-gate-bias |
       | o-gate-state-weights | o-gate-input-weights | o-gate-bias |
       | m-gate-state-weights | m-gate-input-weights | m-gate-bias |
       | dense-weights | dense-bias|

       # Network parameter layout in memory:

       ## w_state_left (for 1 gate):

       |<------ LSTM_UNITS/2 ----->|
       +---------------------------+---
       |XXXXXXXXXXXXXXXXXXXXXXXXXXX| ^
       |XXXXXXXXXXXXXXXXXXXXXXXXXXX| |
       |XXXXXXXXXXXXXXXXXXXXXXXXXXX| | LSTM_UNITS_REAL
       |XXXXXXXXXXXXXXXXXXXXXXXXXXX| |
       |XXXXXXXXXXXXXXXXXXXXXXXXXXX| v
       +---------------------------+---
       |000000000000000000000000000| ^
       |000000000000000000000000000| | LSTM_UNITS - LSTM_UNITS_REAL
       |000000000000000000000000000| v
       +---------------------------+---


       ## w_state_right (for 1 gate):

       (LSTM_UNITS_REAL - LSTM_UNITS/2)
       |<------------------->|
       +---------------------+-----+---
       |XXXXXXXXXXXXXXXXXXXXX|00000| ^
       |XXXXXXXXXXXXXXXXXXXXX|00000| |
       |XXXXXXXXXXXXXXXXXXXXX|00000| | LSTM_UNITS_REAL
       |XXXXXXXXXXXXXXXXXXXXX|00000| |
       |XXXXXXXXXXXXXXXXXXXXX|00000| v
       +---------------------+-----+---
       |000000000000000000000|00000| ^
       |000000000000000000000|00000| | LSTM_UNITS - LSTM_UNITS_REAL
       |000000000000000000000|00000| v
       +---------------------+-----+---
       |<--- LSTM_UNITS/2 -------->|


       ## w_input (for 1 gate):

       INPUT_DIM_REAL
       |<--------->|
       +-----------+----+---
       |XXXXXXXXXXX|0000| ^
       |XXXXXXXXXXX|0000| |
       |XXXXXXXXXXX|0000| | LSTM_UNITS
       |XXXXXXXXXXX|0000| |
       |XXXXXXXXXXX|0000| v
       +-----------+----+---
       |00000000000|0000| ^
       |00000000000|0000| | LSTM_UNITS - LSTM_UNITS_REAL
       |00000000000|0000| v
       +-----------+----+---
       |<- INPUT_DIM -->|
     */
    float w_state_left[N_GATES * W_STATE_SIZE / 2] = {};
    float w_state_right[N_GATES * W_STATE_SIZE / 2] = {};
    float w_input[N_GATES * W_INPUT_SIZE] = {};
    float bias[N_GATES * LSTM_UNITS] = {};

    float w_dense[LSTM_UNITS] = {};
    float b_dense = 0.0f;

#if defined(NO_DISK_IO)
    // Load network parameters from array.
    NetworkParams() {
        // Load LSTM gate parameters.
        int offset = 0;
        for (int i = 0; i < N_GATES; i++) {
            // Skip data for previous gate.
            offset = i * (W_STATE_SIZE_REAL + W_INPUT_SIZE_REAL + LSTM_UNITS_REAL) * sizeof(float);

            // Load state weights.
            for (int r = 0; r < LSTM_UNITS_REAL; r++) {
                memcpy(reinterpret_cast<unsigned char *>(w_state_left + i * (W_STATE_SIZE / 2) + r * (LSTM_UNITS / 2)),
                       (weights_dat + offset + r * LSTM_UNITS_REAL * sizeof(float)),
                       (LSTM_UNITS / 2) * sizeof(float));

                memcpy(reinterpret_cast<unsigned char *>(w_state_right + i * (W_STATE_SIZE / 2) + r * (LSTM_UNITS / 2)),
                       (weights_dat + offset + (r * LSTM_UNITS_REAL + LSTM_UNITS / 2) * sizeof(float)),
                       (LSTM_UNITS_REAL - LSTM_UNITS / 2) * sizeof(float));
            }

            // Load input weights.
            offset += W_STATE_SIZE_REAL * sizeof(float);
            for (int r = 0; r < LSTM_UNITS_REAL; r++) {
                memcpy(reinterpret_cast<unsigned char *>(w_input + i * W_INPUT_SIZE + r * INPUT_DIM),
                       (weights_dat + offset + r * INPUT_DIM_REAL * sizeof(float)),
                       INPUT_DIM_REAL * sizeof(float));
            }

            // Load lstm biases.
            offset += W_INPUT_SIZE_REAL * sizeof(float);
            memcpy(reinterpret_cast<unsigned char *>(bias + i * LSTM_UNITS),
                   (weights_dat + offset),
                   LSTM_UNITS_REAL * sizeof(float));
        }

        // Load dense layer parameters.
        offset = N_GATES * (W_STATE_SIZE_REAL + W_INPUT_SIZE_REAL + LSTM_UNITS_REAL) * sizeof(float);
        memcpy(reinterpret_cast<unsigned char *>(w_dense),
               (weights_dat + offset),
               LSTM_UNITS_REAL * sizeof(float));

        offset += LSTM_UNITS_REAL * sizeof(float);
        memcpy(reinterpret_cast<unsigned char *>(&b_dense),
               (weights_dat + offset),
               sizeof(float));
    }
#else
    // Load network parameters from disk.
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
#endif
};

struct DropbearData {
    /* # Dropbear data layout in disk:
       | vibrations | pin-positions | time |
     */
    float *vib;
    float *pin;
    float *time;
    int n_samples;

#if defined(NO_DISK_IO)
    DropbearData() {
        n_samples = dropbear_dat_len / 3 / sizeof(float);

        vib = new float[n_samples];
        pin = new float[n_samples];
        time = new float[n_samples];

        int offset = 0;
        memcpy(reinterpret_cast<unsigned char *>(vib),
               (dropbear_dat + offset),
               n_samples * sizeof(float));

        offset = n_samples * sizeof(float);
        memcpy(reinterpret_cast<unsigned char *>(pin),
               (dropbear_dat + offset),
               n_samples * sizeof(float));

        offset = 2 * n_samples * sizeof(float);
        memcpy(reinterpret_cast<unsigned char *>(time),
               (dropbear_dat + offset),
               n_samples * sizeof(float));
    }
#else
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
#endif

    ~DropbearData() {
        if (vib)
            delete[] vib;
        if (pin)
            delete[] pin;
        if (time)
            delete[] time;
    }
};
