#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include <cmath>

using namespace aie;

inline __attribute__((always_inline)) vector<float, VECTOR_LANES>
tanh_soft(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> y;

    for (int i = 0; i < VECTOR_LANES; i++) {
        y[i] = tanhf(x[i]);
    }

    return y;
}
