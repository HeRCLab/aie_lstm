#pragma once

#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

inline __attribute__((noinline)) vector<float, VECTOR_LANES>
tanh_soft(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> y;
    float v, e2x;

    // Numerically stable tanhf. We are not using the cmath `tanhf`
    // because it is not available in the library bundled with the
    // compiler.
    //
    // Ref: https://datascience.stackexchange.com/a/131332/177970
    for (int i = 0; i < VECTOR_LANES; i++) {
        v = x[i];
        if (v >= 0.0f) {
            e2x = expf(-2.0f * v);
            y[i] = -(1.0f - 2.0f / (e2x + 1.0f));
        } else {
            e2x = expf(2.0f * v);
            y[i] = -(2.0f / (e2x + 1.0f) - 1.0f);
        }
    }

    return y;
}

inline float tanh_soft_scalar(float x) {
    float e2x;
    if (x >= 0.0f) {
	e2x = expf(-2.0f * x);
	return -(1.0f - 2.0f / (e2x + 1.0f));
    } else {
	e2x = expf(2.0f * x);
	return -(2.0f / (e2x + 1.0f) - 1.0f);
    }

}
