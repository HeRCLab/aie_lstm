#include "common_defs.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

/*
Approximate sigmoidal function formula:
     sigma(x) = 1.0 / (1.0 + (1.0 - x/n)^n)

Ref: https://scicomp.stackexchange.com/a/42187
*/

inline __attribute__((always_inline)) aie::vector<float, VECTOR_LANES>
sigm_approx(aie::vector<float, VECTOR_LANES> &__restrict x) {
    aie::vector<float, VECTOR_LANES> y;
    aie::accumulator<accfloat, VECTOR_LANES> acc;

    acc = aie::mul(1.0f / 128, x);
    y = acc.to_vector<float>(0);
    y = aie::sub(1.0f, y);

    [[chess::prepare_for_pipelining]] {
        for (int i = 0; i < 7; i++) {
            acc = aie::mul_square(y);
            y = acc.to_vector<float>(0);
        }
    }

    y = aie::add(1.0, y);
    y = aie::inv(y);

    return y;
}
