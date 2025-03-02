#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

inline vector<float, VECTOR_LANES>
tanh_soft(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> y;

    for (int i = 0; i < VECTOR_LANES; i++) {
      y[i] = tanh(x[i]);
    }

    return y;
}
