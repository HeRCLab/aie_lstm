struct network_parameters {
	float lstm_weights[4*UNITS*(UNITS+INPUTS)];
	float lstm_bias[4*UNITS];
	float hidden_state[UNITS];

	float dense_weights[UNITS];
	float dense_bias[1];
};

// read pre-trained LSTM from datafiles
void read_network_parameters (struct network_parameters *net) {

	// open the dropbear LSTM
	FILE *myFile = fopen("weights.dat","r");

	// allocate LSTM gate weights and biases
	static float input_gate_recurrent_weights[UNITS*UNITS],
				 input_gate_input_weights[INPUTS*UNITS],
				 input_gate_bias[UNITS],
				 forget_gate_recurrent_weights[UNITS*UNITS],
				 forget_gate_input_weights[INPUTS*UNITS],
				 forget_gate_bias[UNITS],
				 output_gate_recurrent_weights[UNITS*UNITS],
				 output_gate_input_weights[INPUTS*UNITS],
				 output_gate_bias[UNITS],
				 modulation_gate_recurrent_weights[UNITS*UNITS],
				 modulation_gate_input_weights[INPUTS*UNITS],
				 modulation_gate_bias[UNITS],
				 dense_weights[INPUTS*UNITS],
				 dense_bias[UNITS];

	// read weights from file
	fread((void *)input_gate_recurrent_weights,UNITS*UNITS,sizeof(float),myFile);
	fread((void *)input_gate_input_weights,INPUTS*UNITS,sizeof(float),myFile);
	fread((void *)input_gate_bias,UNITS,sizeof(float),myFile);

	fread((void *)forget_gate_recurrent_weights,UNITS,sizeof(float)*UNITS,myFile);
	fread((void *)forget_gate_input_weights,INPUTS*UNITS,sizeof(float),myFile);
	fread((void *)forget_gate_bias,UNITS,sizeof(float),myFile);

	fread((void *)output_gate_recurrent_weights,UNITS,sizeof(float)*UNITS,myFile);
	fread((void *)output_gate_input_weights,INPUTS*UNITS,sizeof(float),myFile);
	fread((void *)output_gate_bias,UNITS,sizeof(float),myFile);

	fread((void *)modulation_gate_recurrent_weights,UNITS,sizeof(float)*UNITS,myFile);
	fread((void *)modulation_gate_input_weights,INPUTS*UNITS,sizeof(float),myFile);
	fread((void *)modulation_gate_bias,UNITS,sizeof(float),myFile);

	fread((void *)dense_weights,UNITS,sizeof(float),myFile);
	fread((void *)dense_bias,sizeof(float),1,myFile);

	fclose(myFile);
	// now, copy the individual weight matrices into a single, large matrix so
	// we can perform the workload of all four gates as a single matrix-vector
	// mutliply.
	//
	// the "unified" LSTM cell weight matrix concatenates the recurrent and
	// input weights horizontally and those of the different gates vertically.
	//
	for (int i=0;i<UNITS*4;i++)
		for (int j=0;j<UNITS+INPUTS;j++)
			if (i<UNITS) // input gate
				if (j<UNITS) // recurrent weights
					net->lstm_weights[i*(UNITS+INPUTS) + j] =
						input_gate_recurrent_weights[i%UNITS * UNITS + j];
				else // input weights
					net->lstm_weights[i*(UNITS+INPUTS) + j] =
						input_gate_input_weights[i%UNITS * INPUTS + j-UNITS];
			else if (i<UNITS*2) // forget gate
				if (j<UNITS) // recurrent weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        forget_gate_recurrent_weights[i%UNITS * UNITS + j];
                else // input weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        forget_gate_input_weights[i%UNITS * INPUTS + j-UNITS];
            else if (i<UNITS*3) // output gate
                if (j<UNITS) // recurrent weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        output_gate_recurrent_weights[i%UNITS * UNITS + j];
                else // input weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        output_gate_input_weights[i%UNITS * INPUTS + j-UNITS];
			else // modulation gate
                if (j<UNITS) // recurrent weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        modulation_gate_recurrent_weights[i%UNITS * UNITS + j];
                else // input weights
                    net->lstm_weights[i*(UNITS+INPUTS) + j] =
                        modulation_gate_input_weights[i%UNITS * INPUTS + j-UNITS];

	// copy biases
	for (int i=0;i<UNITS*4;i++)
		if (i<UNITS) net->lstm_bias[i]=input_gate_bias[i%UNITS];
		else if (i<UNITS*2) net->lstm_bias[i]=forget_gate_bias[i%UNITS];
		else if (i<UNITS*3) net->lstm_bias[i]=output_gate_bias[i%UNITS];
		else net->lstm_bias[i]=modulation_gate_bias[i%UNITS];

	for (int i=0;i<UNITS;i++) net->dense_weights[i]=dense_weights[i];
	net->dense_bias[0] = dense_bias[0];
}

