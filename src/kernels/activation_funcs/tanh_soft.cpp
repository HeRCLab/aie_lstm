#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>
#include <cmath>

inline __attribute__((always_inline)) aie::vector<float, VECTOR_LANES>
tanh_soft(aie::vector<float, VECTOR_LANES> &__restrict x) {
    aie::vector<float, VECTOR_LANES> y;

    for (int i = 0; i < VECTOR_LANES; i++) {
        y[i] = tanhf(x[i]);
    }

    return y;
}
