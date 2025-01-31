#pragma once

#include <aie_api/aie_adf.hpp>
#include <adf.h>

void single_lstm_gate(
    adf::input_circular_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_circular_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,
    const float (&w_state_left)[W_STATE_SIZE/2],
    const float (&w_state_right)[W_STATE_SIZE/2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS],
    adf::output_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output);

void lstm_coordinator(
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict input_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict forget_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict modulation_gate_rslt,
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,
    adf::output_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state_out,
    adf::output_buffer<float, adf::extents<4>> &__restrict output);
