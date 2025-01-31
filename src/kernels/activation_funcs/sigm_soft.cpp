#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include <cmath>

using namespace aie;

inline __attribute__((always_inline)) vector<float, VECTOR_LANES>
sigm_soft(vector<float, VECTOR_LANES> &__restrict x) {
    vector<float, VECTOR_LANES> y;

    for (int i = 0; i < VECTOR_LANES; i++) {
        y[i] = 1.0f / (1.0f + expf(x[i]));
    }

    return y;
}
