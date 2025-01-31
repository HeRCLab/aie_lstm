#include "common_defs.hpp"
#include "kernels.hpp"
#include <adf.h>

using namespace adf;

class LSTMGraph : public graph {
  private:
    kernel coord;
    kernel gates[N_GATES];

  public:
    input_gmio gmioIn;
    output_gmio gmioOut;

    // Ports for run time parameters.
    // -------------------------------,
    // For gates.
    input_port rtpWStateL[N_GATES];
    input_port rtpWStateR[N_GATES];
    input_port rtpWInput[N_GATES];
    input_port rtpBias[N_GATES];

    // For coordinator.
    input_port rtpWDense;
    input_port rtpBDense;
    // -------------------------------'

    LSTMGraph() {
        coord = kernel::create(lstm_coordinator);
        for (int i = 0; i < 4; i++) {
            gates[i] = kernel::create(single_lstm_gate);
        }

        in = input_gmio::create("gmioIn", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);
        out = output_gmio::create("gmioOut", GMIO_BURST_LENGTH, GMIO_BANDWIDTH);

        for (int i = 0; i < N_GATES; i++) {
            // Connect graph data input to gate data input.
            connect<>(in.out[0], gates[i].in[0]);
            // Connect coordinator state output to gate state input.
            connect<>(coord.out[0], gates[i].in[1]);

            // Connect run-time-parameters.
            connect<parameter>(rtpWStateL[i], async(gates[i].in[2]));
            connect<parameter>(rtpWStateR[i], async(gates[i].in[3]));
            connect<parameter>(rtpWInput[i], async(gates[i].in[4]));
            connect<parameter>(rtpBias[i], async(gates[i].in[5]));

            // Connect gate output to coordinator inputs.
            connect<>(gates[i].out[0], coord.in[i]);
        }

        // Connect coordinator output to graph output port.
        connect<>(coord.out[1], out.in[0]);
        // Connect coordinator run-time parameters.
        connect<parameter>(rtpWDense, async(out.in[4]));
        connect<parameter>(rtpWBias, async(out.in[5]));

        // Map source files for debugging.
        source(coord) = "lstm_coordinator.cpp";
        for (int i = 0; i < N_GATES; i++) {
            source(gate[i]) = "simple_lstm_gate.cpp";
        }

        // Set runtime ratio high so that all kernels get their own AI
        // engines.
        runtime<ratio>(coord) = 0.9;
        for (int i = 0; i < N_GATES; i++) {
            runtime<ratio>(gates[i]) = 0.9
        }
    }
};
