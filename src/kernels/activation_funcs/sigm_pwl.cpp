#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

/*
Pseudocode for piecewise linear sigmoid:

    def sigmoid(x):
        if (|x| >= 5)
            y = 1
        elif (2.375 <= |x| < 5)
            y = 0.03125 * |x| + 0.84375
        elif (1 <= |x| < 2.375)
            y = 0.125 * |x| + 0.625
        elif (0 <= |x| < 1)
            y = 0.25 * |x| + 0.5

        if (X < 0)
            y = 1 - y

        return y

Ref: Piecewise linear approximation applied to nonlinear function of a
     neural network, H.Amin, K.M.Curtis, B.R.Hayes-Gill
*/
inline __attribute__((always_inline)) vector<float, VECTOR_LANES>
sigm_pwl(vector<float, VECTOR_LANES> &__restrict x) {
    vector<float, VECTOR_LANES> x_abs, y, y1, y2, y3, y_neg;
    vector<accfloat, VECTOR_LANES> acc1, acc2;
    mask<VECTOR_LANES> comp_mask;

    x_abs = abs(x);

    // Prepare possible alternatives of y for each vector lane.
    // --------------------------------------------------------,
    acc1 = broadcast<accfloat, VECTOR_LANES>(0.84375f);
    y1 = mac(acc1, x_abs, 0.03125f).to_vector<float>(0);

    acc2 = broadcast<accfloat, VECTOR_LANES>(0.625f);
    y2 = mac(acc2, x_abs, 0.125f).to_vector<float>(0);

    acc1 = broadcast<accfloat, VECTOR_LANES>(0.5f);
    y3 = mac(acc1, x_abs, 0.25f).to_vector<float>(0);
    // --------------------------------------------------------'

    // Conditionally select y based on values of |x|.
    // --------------------------------------------------------,
    comp_mask = lt(x_abs, 5.0f);
    y = select(y1, 1.0f, comp_mask);

    comp_mask = lt(x_abs, 2.375f);
    y = select(y2, y, comp_mask);

    comp_mask = lt(x_abs, 1.0f);
    y = select(y3, y, comp_mask);
    // --------------------------------------------------------'

    // Calculate alternative value of y where x is negative.
    y_neg = sub(1.0f, y);
    comp_mask = lt(x, 0.0f);
    y = select(y_neg, y, comp_mask);

    return y;
}
