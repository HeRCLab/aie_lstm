#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

/*
Pseudocode for piecewise linear tanh(x) function:

    def tanh(x):
        if (0 <= |x| < 0.25)
            y = |x|

        elif (0.25 <= |x| < 0.5)
            y = (1 - 1/8) * |x| + 0.02935791015625
              = 0.875 * |x| + 0.02935791015625

        elif (0.5 <= |x| < 1)
            y = (1/2 + 1/8) * |x| + 0.15625
              = 0.625 * |x| + 0.15625

        elif (1 <= |x| < 1.5)
            y = (1/4 + 1/8) * |x| + 0.453125
              = 0.3125 * |x| + 0.453125

        elif (1.5 <= |x| < 2)
            y = (1/8 + 1/512) * |x| + 0.716796875
              = 0.126953125 * |x| + 0.716796875

        else:
            y = 1.0

        if (x < 0.0)
            y = -1.0 * y

        return y

Ref: Cost effective Tanh activation function circuits based on fast
     piecewise linear logic, Kezhu Liu1, Weiwei Shi, Canji Huang,
     Dongdong Zeng
*/
inline vector<float, VECTOR_LANES>
tanh_pwl(const vector<float, VECTOR_LANES> x) {
    vector<float, VECTOR_LANES> x_abs, y, y1, y2, y3, y4, y_neg;
    accum<accfloat, VECTOR_LANES> acc1, acc2;
    mask<VECTOR_LANES> comp_mask;

    x_abs = abs(x);

    // Prepare possible alternatives of y for each vector lane.
    // --------------------------------------------------------,
    acc1.from_vector(broadcast<float, VECTOR_LANES>(0.7167968f));
    y1 = mac(acc1, x_abs, 0.1269531f).to_vector<float>(0);

    acc2.from_vector(broadcast<float, VECTOR_LANES>(0.453125f));
    y2 = mac(acc2, x_abs, 0.3125f).to_vector<float>(0);

    acc1.from_vector(broadcast<float, VECTOR_LANES>(0.15625f));
    y3 = mac(acc1, x_abs, 0.625f).to_vector<float>(0);

    acc2.from_vector(broadcast<float, VECTOR_LANES>(0.02935791015625f));
    y4 = mac(acc2, x_abs, 0.875f).to_vector<float>(0);
    // --------------------------------------------------------'

    // Conditionally select y based on values of |x|.
    // --------------------------------------------------------,
    comp_mask = ge(x_abs, 0.25f);
    y = select(x_abs, y4, comp_mask);

    comp_mask = ge(x_abs, 0.5f);
    y = select(y, y3, comp_mask);

    comp_mask = ge(x_abs, 1.0f);
    y = select(y, y2, comp_mask);

    comp_mask = ge(x_abs, 1.5f);
    y = select(y, y1, comp_mask);

    comp_mask = ge(x_abs, 2.0f);
    y = select(y, 1.0f, comp_mask);
    // --------------------------------------------------------'

    // Calculate alternative value of y where x is negative.
    y_neg = neg(y);
    comp_mask = lt(x, 0.0f);
    y = select(y, y_neg, comp_mask);

    return y;
}
