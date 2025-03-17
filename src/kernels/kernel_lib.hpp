// This is a workaround for having proper source mapping for the
// purpose of debugging AIE kernels, as described in UG1079(v2024.1),
// Ch 7, Sec "Recommended Project Directory Structure", Page 108.
#pragma once

#include "common_defs.hpp"

#if defined(USE_PWL_ACTIVATION_FNS)

#include "activation_funcs/sigm_pwl.hpp"
#include "activation_funcs/tanh_pwl.hpp"
#define SIGM(x) (sigm_pwl(x))
#define TANH(x) (tanh_pwl(x))
#define SIGM_SCALAR(x) (sigm_pwl_scalar(x))
#define TANH_SCALAR(x) (tanh_pwl_scalar(x))

#elif defined(USE_APPROX_ACTIVATION_FNS)

#include "activation_funcs/sigm_approx.hpp"
#include "activation_funcs/tanh_approx.hpp"
#define SIGM(x) (sigm_approx(x))
#define TANH(x) (tanh_approx(x))

#else

#include "activation_funcs/sigm_soft.hpp"
#include "activation_funcs/tanh_soft.hpp"
#define SIGM(x) (sigm_soft(x))
#define TANH(x) (tanh_soft(x))
#define SIGM_SCALAR(x) (sigm_soft_scalar(x))
#define TANH_SCALAR(x) (tanh_soft_scalar(x))

#endif
