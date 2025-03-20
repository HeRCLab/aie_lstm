#pragma once

#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

// NOTE: All functions defined here need to inline for x86sim
// compilation to work! This is a bug of our project organization.

using namespace aie;

inline void print_state(const float (&state)[LSTM_UNITS]) {
    auto state_iter = begin_vector<VECTOR_LANES>(state);
    vector<float, VECTOR_LANES> v;
    printf("state_v: -- start --\n");
    for (int i = 0; i < LSTM_UNITS / VECTOR_LANES; i++) {
        v = *state_iter++;
        print(v, true, "state_v: ");
    }
    printf("state_v: -- end --\n");
}
