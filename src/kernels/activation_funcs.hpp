#pragma once
#include "common_defs.hpp"
#include <aie_api/aie.hpp>

using namespace aie;

vector<float, VECTOR_LANES>
sigm_soft(const vector<float, VECTOR_LANES> x);

vector<float, VECTOR_LANES>
tanh_soft(const vector<float, VECTOR_LANES> x);

vector<float, VECTOR_LANES>
sigm_pwl(const vector<float, VECTOR_LANES> x);

vector<float, VECTOR_LANES>
tanh_pwl(const vector<float, VECTOR_LANES> x);

vector<float, VECTOR_LANES>
sigm_approx(const vector<float, VECTOR_LANES> x);

vector<float, VECTOR_LANES>
tanh_approx(const vector<float, VECTOR_LANES> x);

#if defined(USE_PWL_ACTIVATION_FNS)

#define sigm sigm_pwl
#define tanh tanh_pwl

#elif defined(USE_APPROX_ACTIVATION_FNS)

#define sigm sigm_approx
#define tanh tanh_approx

#else

#define sigm sigm_soft
#define tanh tanh_soft

#endif
