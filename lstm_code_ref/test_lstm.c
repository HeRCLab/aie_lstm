#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// assume one LSTM layer and one dense layer
// additional LSTM layers can be added through some modifications
// to this code
#define UNITS 100
#define INPUTS 20

#include "dropbear.h"
#include "mydumblittleplotroutine.h"
#include "weights.h"

#define APPROX_SIGM_N 128.0f

#if defined(USE_APPROX_ACTIVATION_FNS)

/*
Approximate sigmoidal function formula:

    sigma(x) = 1.0 / (1.0 + (1.0 - x/n)^n)

Ref: https://scicomp.stackexchange.com/a/42187
*/
float sigm(float x) {
  float y = 1 - x / APPROX_SIGM_N;
  int m = (int)log2(APPROX_SIGM_N);
  for (int i = 0; i < m; i++) {
    y *= y;
  }

  return 1.0 / (1.0 + y);
}

#elif defined(USE_PWL_ACTIVATION_FNS)
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
float sigm(float x) {
  float x_abs = fabsf(x);
  float y;

  if (x_abs >= 5.0f) {
    y = 1.0f;
  } else if (x_abs >= 2.375f) {
    y = 0.03125f * x_abs + 0.84375f;
  } else if (x_abs >= 1.0f) {
    y = 0.125f * x_abs + 0.625f;
  } else {
    y = 0.25 * x_abs + 0.5;
  }

  if (x < 0.0f) {
    y = 1.0f - y;
  }

  return y;
}

#else

float sigm(float x) { return 1.f / (1.f + expf(-x)); }

#endif

#if defined(USE_APPROX_ACTIVATION_FNS)
/*
Approximate hyperbolic tangent formula using 7th-degree lambert
approximant:

    tanh(x) = (135135 * x + 17325 * x^3 + 378 * x^5 + x^7) / (135135 +
               62370 * x^2 + 3150 * x^4 + 28 * x^6)

Ref:
https://yaikhom.com/2020-04-28-localised-approximation-of-hyperbolic-tangents.html
*/
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

#elif defined(USE_PWL_ACTIVATION_FNS)
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
float my_tanh(float x) {
  float y;
  float x_abs = fabsf(x);

  if (x_abs >= 2.0f) {
    y = 1.0f;
  } else if (x_abs >= 1.5f) {
    y = 0.1269531f * x_abs + 0.7167968f;
  } else if (x_abs >= 1.0f) {
    y = 0.3125f * x_abs + 0.453125f;
  } else if (x_abs >= 0.5f) {
    y = 0.625f * x_abs + 0.15625f;
  } else if (x_abs >= 0.25f) {
    y = 0.875f * x_abs + 0.02935791015625f;
  } else {
    y = x_abs;
  }

  if (x < 0.0f) {
    y *= -1.0;
  }

  return y;
}

#else
float my_tanh(float x) { return tanhf(x); }
#endif

// computes c += Ab
// matrix is row-major
void mvm(float *c, float *A, float *b, int m, int n) {
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      c[i] += A[i * n + j] * b[j];
}

int main() {
  // read dropbear
  float *time, *vib, *pin;
  int nsamples = read_dropbear(&time, &vib, &pin);

  // read pre-trained LSTM model
  struct network_parameters mynet;
  read_network_parameters(&mynet);

  // run the model

  // allocate memory for LSTM output
  float *predicted_pin = (float *)malloc(nsamples * sizeof(float));

  // allocate LSTM state
  static float state[UNITS]; // use static to init to 0

  // rename the LSTM matrix (for code readability)
  float *lstm_matrix = mynet.lstm_weights;

  // LSTM input vector: concatenate the state and inputs
  // this allows us to perform only one matrix-vector multiply
  // per LSTM inferrence
  float i_vector[UNITS + INPUTS];

  for (int i = INPUTS - 1; i < nsamples; i++) {
    // copy inputs and state to vector
    for (int j = 0; j < UNITS + INPUTS; j++)
      i_vector[j] = j < UNITS ? state[j] : vib[i - INPUTS + j - UNITS + 1];

    // allocate output vectoro
    float gates[4 * UNITS];

    // add biases
    for (int j = 0; j < UNITS * 4; j++)
      gates[j] = mynet.lstm_bias[j];

    // matrix-vector mutliply (also adds biases)
    mvm(gates, lstm_matrix, i_vector, UNITS * 4, UNITS + INPUTS);

    // LSTM additional steps
    // legend:
    //   gates[0 : UNITS-1] : input gate
    //   gates[UNITS : 2*UNITS-1] : forget gate
    //   gates[UNITS*2 : 3*UNITS-1] : output gate
    //   gates[UNITS*3 : 4*UNITS-1] : modulation gate
    float output[UNITS];
    float sum = mynet.dense_bias[0];
    for (int j = 0; j < UNITS; j++) {
      state[j] = sigm(gates[UNITS + j]) * state[j] +
                 sigm(gates[j]) * my_tanh(gates[UNITS * 3 + j]);

      output[j] = sigm(gates[UNITS * 2 + j]) + sigm(state[j]);

      // done with LSTM, now perform dense layer
      sum += output[j] * mynet.dense_weights[j];
    }

    predicted_pin[i] = sum;
  }

  // plot results
  plot(time, pin, "pin position", predicted_pin, "pred. pin position",
       "LSTM results", nsamples);
  return 0;
}
