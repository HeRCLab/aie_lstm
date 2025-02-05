#pragma once

#include "common_defs.hpp"
#include <aie_api/aie.hpp>

#if defined(USE_PWL_ACTIVATION_FNS)

#include "sigm_pwl.hpp"
#include "tanh_pwl.hpp"
#define SIGM(x) (sigm_pwl(x))
#define TANH(x) (tanh_pwl(x))

#elif defined(USE_APPROX_ACTIVATION_FNS)

#include "sigm_approx.hpp"
#include "tanh_approx.hpp"
#define SIGM(x) (sigm_approx(x))
#define TANH(x) (tanh_approx(x))

#else

#include "sigm_soft.hpp"
#include "tanh_soft.hpp"
#define SIGM(x) (sigm_soft(x))
#define TANH(x) (tanh_soft(x))

#endif
