#pragma once
#include "common_defs.hpp"

aie::vector<float, VECTOR_LANES>
sigm_soft(aie::vector<float, VECTOR_LANES> &__restrict x);

aie::vector<float, VECTOR_LANES>
tanh_soft(aie::vector<float, VECTOR_LANES> &__restrict x);

aie::vector<float, VECTOR_LANES>
sigm_pwl(aie::vector<float, VECTOR_LANES> &__restrict x);

aie::vector<float, VECTOR_LANES>
tanh_pwl(aie::vector<float, VECTOR_LANES> &__restrict x);

aie::vector<float, VECTOR_LANES>
sigm_approx(aie::vector<float, VECTOR_LANES> &__restrict x);

aie::vector<float, VECTOR_LANES>
tanh_approx(aie::vector<float, VECTOR_LANES> &__restrict x);

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
