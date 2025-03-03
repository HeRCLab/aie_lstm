#pragma once

#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

inline __attribute__((noinline)) vector<float, VECTOR_LANES>
sigm_soft(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> y;
    float v, t;

    for (int i = 0; i < VECTOR_LANES; i++) {
        v = x[i];
        // Numerically stable sigmoid.
        if (v >= 0.0f) {
            y[i] = 1.0f / (1.0f + expf(-v));
        } else {
            t = expf(v);
            y[i] = t / (1.0f + t);
        }
    }

    return y;
}
