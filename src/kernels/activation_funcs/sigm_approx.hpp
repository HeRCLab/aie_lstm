#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

/*
Approximate sigmoidal function formula:
     sigma(x) = 1.0 / (1.0 + (1.0 - x/n)^n)

Ref: https://scicomp.stackexchange.com/a/42187
*/

inline __attribute__((always_inline)) vector<float, VECTOR_LANES>
sigm_approx(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> y;
    accum<accfloat, VECTOR_LANES> acc;

    acc = mul(1.0f / 128, x);
    y = acc.to_vector<float>(0);
    y = sub(1.0f, y);

    /* [[chess::prepare_for_pipelining]] */ for (int i = 0; i < 7; i++) {
        acc = mul_square(y);
        y = acc.to_vector<float>(0);
    }

    y = add(1.0f, y);
    y = inv(y);

    return y;
}
