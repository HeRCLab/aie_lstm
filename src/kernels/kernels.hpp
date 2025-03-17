#pragma once

#include "common_defs.hpp"
#include <aie_api/aie_adf.hpp>
#include <adf.h>

void gate_full_buff(
    adf::input_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,
    const float (&w_state_left)[W_STATE_SIZE/2],
    const float (&w_state_right)[W_STATE_SIZE/2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS],
    adf::output_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output);

void gate_half_buff(
    adf::input_buffer<float, adf::extents<INPUT_DIM>> &__restrict input,
    adf::input_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state,
    const float (&w_state_part)[W_STATE_SIZE / 2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS / 2],
    const int partial_computation,
    adf::output_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict output);

void gate_half_strm(
    input_stream<float> *input_strm,
    input_stream<float> *state_strm,
    const float (&w_state_part)[W_STATE_SIZE / 2],
    const float (&w_input)[W_INPUT_SIZE],
    const float (&bias)[LSTM_UNITS / 2],
    const int gate_id,
    output_pktstream *output_strm);

void coordinator_4x1_buff(
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict input_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict forget_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict output_gate_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict modulation_gate_rslt,
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,
    adf::output_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state_out,
    adf::output_buffer<float, adf::extents<OUTPUT_BLOCK_SIZE>> &__restrict output);

void coordinator_4x2_buff(
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict ig_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict ig_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict fg_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict fg_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict og_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict og_down_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict mg_up_rslt,
    adf::input_async_buffer<float, adf::extents<LSTM_UNITS/2>> &__restrict mg_down_rslt,

    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,
    adf::output_async_buffer<float, adf::extents<LSTM_UNITS>> &__restrict state_out,
    adf::output_buffer<float, adf::extents<OUTPUT_BLOCK_SIZE>> &__restrict output);

void coordinator_4x2_strm(
    input_pktstream *gate_strm,
    const float (&w_dense)[LSTM_UNITS],
    const float bias_dense,
    output_stream<float> *state_out_strm,
    output_stream<float> *output_strm);

