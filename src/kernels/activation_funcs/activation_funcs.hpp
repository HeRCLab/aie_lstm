#pragma once

#include "common_defs.hpp"
#include <aie_api/aie.hpp>

#if defined(USE_PWL_ACTIVATION_FNS)

#include "sigm_pwl.hpp"
#include "tanh_pwl.hpp"
#define sigm sigm_pwl
#define tanh tanh_pwl

#elif defined(USE_APPROX_ACTIVATION_FNS)

#include "sigm_approx.hpp"
#include "tanh_approx.hpp"
#define sigm sigm_approx
#define tanh tanh_approx

#else

#include "sigm_soft.hpp"
#include "tanh_soft.hpp"
#define sigm sigm_soft
#define tanh tanh_soft

#endif
