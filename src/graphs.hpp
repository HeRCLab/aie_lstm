#pragma once

#include "common_defs.hpp"
#include "kernels.hpp"
#include <adf.h>

using namespace adf;

class LSTMGraph_4x1_buff : public graph {
  private:
    kernel coord;
    kernel gates[N_GATES];

  public:
    input_gmio gmio_in;
    output_gmio gmio_out;

    // Ports for run time parameters.
    // -------------------------------,
    // For gates.
    input_port rtp_w_state_left[N_GATES];
    input_port rtp_w_state_right[N_GATES];
    input_port rtp_w_input[N_GATES];
    input_port rtp_bias[N_GATES];

    // For coordinator.
    input_port rtp_w_dense;
    input_port rtp_b_dense;
    // -------------------------------'

    LSTMGraph_4x1_buff() {
        // Create kernels.
        coord = kernel::create(coordinator_4x1_buff);
        for (int i = 0; i < N_GATES; i++) {
            gates[i] = kernel::create(gate_full_buff);
        }

        // Assign source files to each kernel.
        source(coord) = "coordinator_4x1_buff.cpp";
        for (int i = 0; i < N_GATES; i++) {
            source(gates[i]) = "gate_full_buff.cpp";
        }

        // Create GMIO ports.
        gmio_in = input_gmio::create("gmioIn", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        gmio_out = output_gmio::create("gmioOut", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);

        for (int i = 0; i < N_GATES; i++) {
            // Connect graph data input to gate data input.
            connect<>(gmio_in.out[0], gates[i].in[0]);

            // Connect coordinator state output to gate state input.
            connect<>(coord.out[0], gates[i].in[1]);

            // Connect run-time-parameters.
            connect<parameter>(rtp_w_state_left[i], async(gates[i].in[2]));
            connect<parameter>(rtp_w_state_right[i], async(gates[i].in[3]));
            connect<parameter>(rtp_w_input[i], async(gates[i].in[4]));
            connect<parameter>(rtp_bias[i], async(gates[i].in[5]));

            // Connect gate output to coordinator inputs.
            connect<>(gates[i].out[0], coord.in[i]);
        }

        // Set single buffer constraint for gate RTP-s.
        for (int i = 0; i < N_GATES; i++) {
            single_buffer(gates[i].in[2]);
            single_buffer(gates[i].in[3]);
            single_buffer(gates[i].in[4]);
            single_buffer(gates[i].in[5]);
        }

        // Set single buffer constraint for coordinator RTP-s.
        single_buffer(coord.in[4]);
        single_buffer(coord.in[5]);

        // Connect coordinator output to graph output port.
        connect<>(coord.out[1], gmio_out.in[0]);

        // Connect coordinator RTP-s.
        connect<parameter>(rtp_w_dense, async(coord.in[4]));
        connect<parameter>(rtp_b_dense, async(coord.in[5]));

        // Set runtime ratio high so that all kernels get their own AI
        // engines.
        runtime<ratio>(coord) = 0.9;
        for (int i = 0; i < N_GATES; i++) {
            runtime<ratio>(gates[i]) = 0.9;
        }

	// The gates are placed around the coordinator as close as
	// possible with enough memory tiles for each gates.
	location<kernel>(gates[0]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=1});
	location<kernel>(gates[1]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=1});
	location<kernel>(gates[2]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=0});
	location<kernel>(gates[3]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=0});
    }
};

class LSTMGraph_4x2_buff : public graph {
  private:
    kernel coord;
    kernel gates[N_HALF_GATES];

  public:
    input_gmio gmio_in;
    output_gmio gmio_out;

    // Ports for run time parameters.
    // -------------------------------,
    // For gates.
    input_port rtp_w_state[N_HALF_GATES];
    input_port rtp_w_input[N_HALF_GATES];
    input_port rtp_bias[N_HALF_GATES];
    input_port partial_computation_flag[N_HALF_GATES];

    // For coordinator.
    input_port rtp_w_dense;
    input_port rtp_b_dense;
    // -------------------------------'

    LSTMGraph_4x2_buff() {
        // Create kernels.
        coord = kernel::create(coordinator_4x2_buff);
        for (int i = 0; i < N_HALF_GATES; i++) {
            gates[i] = kernel::create(gate_half_buff);
        }

        // Assign source files to each kernel.
        source(coord) = "coordinator_4x2_buff.cpp";
        for (int i = 0; i < N_HALF_GATES; i++) {
            source(gates[i]) = "gate_half_buff.cpp";
        }

        // Create GMIO ports.
        gmio_in = input_gmio::create("gmioIn", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        gmio_out = output_gmio::create("gmioOut", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);

        for (int i = 0; i < N_HALF_GATES; i++) {
            // Connect graph data input to gate data input.
            connect<>(gmio_in.out[0], gates[i].in[0]);

            // Connect coordinator state output to gate state input.
            connect<>(coord.out[0], gates[i].in[1]);

            // Connect run-time-parameters.
            connect<parameter>(rtp_w_state[i], async(gates[i].in[2]));
            connect<parameter>(rtp_w_input[i], async(gates[i].in[3]));
            connect<parameter>(rtp_bias[i], async(gates[i].in[4]));
            connect<parameter>(partial_computation_flag[i], async(gates[i].in[5]));

            // Connect gate output to coordinator inputs.
            connect<>(gates[i].out[0], coord.in[i]);
        }

        // Set single buffer constraint for gate RTP-s.
        for (int i = 0; i < N_HALF_GATES; i++) {
            for (int j = 2; j < 6; j++) {
                single_buffer(gates[i].in[j]);
            }
        }

        // Set single buffer constraint for coordinator RTP-s.
        single_buffer(coord.in[8]);
        single_buffer(coord.in[9]);

        // Connect coordinator output to graph output port.
        connect<>(coord.out[1], gmio_out.in[0]);

        // Connect coordinator RTP-s.
        connect<parameter>(rtp_w_dense, async(coord.in[8]));
        connect<parameter>(rtp_b_dense, async(coord.in[9]));

        // Set runtime ratio high so that all kernels get their own AI
        // engines.
        runtime<ratio>(coord) = 0.9;
        for (int i = 0; i < N_HALF_GATES; i++) {
            runtime<ratio>(gates[i]) = 0.9;
        }

	// Place upper gates strictly up-down-left-right of the coordinator.
	location<kernel>(gates[0]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=0});
	location<kernel>(gates[2]) = location<kernel>(coord) + relative_offset({.col_offset=0, .row_offset=1});
	location<kernel>(gates[4]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=0});
	location<kernel>(gates[6]) = location<kernel>(coord) + relative_offset({.col_offset=0, .row_offset=-1});

	// Place lower gates at the corner of the coordinator.
	location<kernel>(gates[1]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=1});
	location<kernel>(gates[3]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=-1});
	location<kernel>(gates[5]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=1});
	location<kernel>(gates[7]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=-1});

    }
};

class LSTMGraph_4x2_strm : public graph {
  private:
    kernel coord;
    kernel gates[N_HALF_GATES];
    pktmerge<N_HALF_GATES> gate_output_merger;

  public:
    input_gmio gmio_in;
    output_gmio gmio_out;

    // Ports for run time parameters.
    // -------------------------------,
    // For gates.
    input_port rtp_w_state[N_HALF_GATES];
    input_port rtp_w_input[N_HALF_GATES];
    input_port rtp_bias[N_HALF_GATES];
    input_port gate_id[N_HALF_GATES];

    // For coordinator.
    input_port rtp_w_dense;
    input_port rtp_b_dense;
    // -------------------------------'

    LSTMGraph_4x2_strm() {
        // Create kernels.
        coord = kernel::create(coordinator_4x2_strm);
        for (int i = 0; i < N_HALF_GATES; i++) {
            gates[i] = kernel::create(gate_half_strm);
        }

        // Assign source files to each kernel.
        source(coord) = "coordinator_4x2_strm.cpp";
        for (int i = 0; i < N_HALF_GATES; i++) {
            source(gates[i]) = "gate_half_strm.cpp";
        }

        // Create gate output packet stream mergers.
        gate_output_merger = pktmerge<N_HALF_GATES>::create();

        // Create GMIO ports.
        gmio_in = input_gmio::create("gmioIn", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        gmio_out = output_gmio::create("gmioOut", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);

	// Connect coordinator state output to gate state input.
	connect<> s0 (coord.out[0], gates[0].in[1]);
	connect<> s1 (coord.out[0], gates[1].in[1]);
	connect<> s2 (coord.out[0], gates[2].in[1]);
	connect<> s3 (coord.out[0], gates[3].in[1]);
	connect<> s4 (coord.out[0], gates[4].in[1]);
	connect<> s5 (coord.out[0], gates[5].in[1]);
	connect<> s6 (coord.out[0], gates[6].in[1]);
	connect<> s7 (coord.out[0], gates[7].in[1]);
	fifo_depth(s0) = LSTM_UNITS;
	fifo_depth(s1) = LSTM_UNITS;
	fifo_depth(s2) = LSTM_UNITS;
	fifo_depth(s3) = LSTM_UNITS;
	fifo_depth(s4) = LSTM_UNITS;
	fifo_depth(s5) = LSTM_UNITS;
	fifo_depth(s6) = LSTM_UNITS;
	fifo_depth(s7) = LSTM_UNITS;

        for (int i = 0; i < N_HALF_GATES; i++) {
            // Connect graph data input to gate data input.
            connect<> (gmio_in.out[0], gates[i].in[0]);

	    // Connect coordinator state output to gate state input.
	    // connect<> (coord.out[0], gates[i].in[1]);

            // Connect run-time-parameters.
            connect<parameter>(rtp_w_state[i], async(gates[i].in[2]));
            connect<parameter>(rtp_w_input[i], async(gates[i].in[3]));
            connect<parameter>(rtp_bias[i], async(gates[i].in[4]));
            connect<parameter>(gate_id[i], async(gates[i].in[5]));
	    connect<>(gates[i].out[0], gate_output_merger.in[i]);
        }

        // Set single buffer constraint for gate RTP-s.
        for (int i = 0; i < N_HALF_GATES; i++) {
            for (int j = 2; j < 6; j++) {
                single_buffer(gates[i].in[j]);
            }
        }

        // Connect coordinator output to graph output port.
        connect<>(coord.out[1], gmio_out.in[0]);

        // Connect gate output packet-stream merger to coordinator.
        connect(gate_output_merger.out[0], coord.in[0]);

        // Connect coordinator RTP-s.
        connect<parameter>(rtp_w_dense, async(coord.in[1]));
        connect<parameter>(rtp_b_dense, async(coord.in[2]));

        // Set single buffer constraint for coordinator RTP-s.
        single_buffer(coord.in[1]);
        single_buffer(coord.in[2]);

        // Set runtime ratio high so that all kernels get their own AI
        // engines.
        runtime<ratio>(coord) = 0.9;
        for (int i = 0; i < N_HALF_GATES; i++) {
            runtime<ratio>(gates[i]) = 0.9;
        }

	// Place upper gates strictly up-down-left-right of the coordinator.
	location<kernel>(gates[0]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=0});
	location<kernel>(gates[2]) = location<kernel>(coord) + relative_offset({.col_offset=0, .row_offset=1});
	location<kernel>(gates[4]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=0});
	location<kernel>(gates[6]) = location<kernel>(coord) + relative_offset({.col_offset=0, .row_offset=-1});

	// Place lower gates at the corner of the coordinator.
	location<kernel>(gates[1]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=1});
	location<kernel>(gates[3]) = location<kernel>(coord) + relative_offset({.col_offset=1, .row_offset=-1});
	location<kernel>(gates[5]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=1});
	location<kernel>(gates[7]) = location<kernel>(coord) + relative_offset({.col_offset=-1, .row_offset=-1});
    }
};
