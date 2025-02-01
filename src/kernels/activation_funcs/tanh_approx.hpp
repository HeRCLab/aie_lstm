#include "common_defs.hpp"
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#include <aie_api/utils.hpp>

using namespace aie;

/*
Approximate hyperbolic tangent formula using 7th-degree lambert
approximant:

     tanh(x) = (135135 * x + 17325 * x^3 + 378 * x^5 + x^7) / (135135
               + 62370 * x^2 + 3150 * x^4 + 28 * x^6)

Ref:
https://yaikhom.com/2020-04-28-localised-approximation-of-hyperbolic-tangents.html

Plain C implementation:
```
    float my_tanh(float x) {
        float x2 = x * x;
        float x3 = x * x2;
        float x4 = x2 * x2;
        float x5 = x2 * x3;
        float x6 = x3 * x3;
        float x7 = x3 * x4;

        return ((135135.0f * x + 17325.0f * x3 + 378.0f * x5 + x7) /
                (135135.0f + 62370.0f * x2 + 3150.0 * x4 + 28 * x6));
    }
```
*/
inline __attribute__((always_inline)) vector<float, VECTOR_LANES>
tanh_approx(const vector<float, VECTOR_LANES> x) {
    accum<accfloat, VECTOR_LANES> acc_num = 0.0f;
    accum<accfloat, VECTOR_LANES> acc_den = 135135.0f;
    vector<float, VECTOR_LANES> x2, x3, x4, x5, x6, x7;

    // The instructions are ordered this way to fill the pipeline
    // during two idle cycles of each float MACs.
    // ----------------------------------------------------------,
    x2 = mul_square(x).to_vector<float>(0);
    acc_num = mac(acc_num, x, 135135.0f);

    x3 = mul(x, x2).to_vector<float>(0);
    acc_den = mac(acc_den, x2, 62370.0f);

    x4 = mul_square(x2).to_vector<float>(0);
    acc_num = mac(acc_num, x3, 17325.0f);

    x5 = mul(x2, x3).to_vector<float>(0);
    acc_den = mac(acc_den, x4, 3150.0f);

    x6 = mul_square(x3).to_vector<float>(0);
    acc_num = mac(acc_num, x5, 378.0f);

    x7 = mul(x3, x4).to_vector<float>(0);
    acc_den = mac(acc_den, x6, 28.0f);
    acc_num = mac(acc_num, x7, 1.0f);
    // ----------------------------------------------------------'

    auto num = acc_num.to_vector<float>(0);
    auto den_inv = inv(acc_den.to_vector<float>(0));

    return mul(num, den_inv).to_vector<float>(0);
}
