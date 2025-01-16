#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// assume one LSTM layer and one dense layer
// additional LSTM layers can be added through some modifications
// to this code
#define UNITS	100
#define INPUTS	20

#include "dropbear.h"
#include "weights.h"
#include "mydumblittleplotroutine.h"

float sigm(float x) {
	return 1.f/(1.f+expf(-x));
}

// computes c += Ab
// matrix is row-major
void mvm (float *c,float *A,float *b,int m,int n) {
	for (int i=0;i<m;i++)
		for (int j=0;j<n;j++)
			c[i] += A[i*n+j] * b[j];
}

int main() {
	// read dropbear
	float *time,*vib,*pin;	
	int nsamples = read_dropbear(&time,&vib,&pin);

	// read pre-trained LSTM model
	struct network_parameters mynet;
	read_network_parameters(&mynet);

	// run the model

	// allocate memory for LSTM output
	float *predicted_pin = (float *)malloc(nsamples*sizeof(float));

	// allocate LSTM state
	static float state[UNITS]; // use static to init to 0

	// rename the LSTM matrix (for code readability)
	float *lstm_matrix = mynet.lstm_weights;

	// LSTM input vector: concatenate the state and inputs
	// this allows us to perform only one matrix-vector multiply
	// per LSTM inferrence
	float i_vector[UNITS+INPUTS];

	for (int i=INPUTS-1;i<nsamples;i++) {
		// copy inputs and state to vector
		for (int j=0;j<UNITS+INPUTS;j++)
			i_vector[j]=j<UNITS ? state[j] : vib[i-INPUTS+j-INPUTS];
		
		// allocate output vectoro
		float gates[4*UNITS];

		// add biases
		for (int j=0;j<UNITS*4;j++) gates[j] = mynet.lstm_bias[j];

		// matrix-vector mutliply (also adds biases)
		mvm(gates,lstm_matrix,i_vector,UNITS*4,UNITS+INPUTS);

		// LSTM additional steps
		// legend:
		//   gates[0 : UNITS-1] : input gate
		//   gates[UNITS : 2*UNITS-1] : forget gate
		//   gates[UNITS*2 : 3*UNITS-1] : output gate
		//   gates[UNITS*3 : 4*UNITS-1] : modulation gate
		float output[UNITS];
		float sum = mynet.dense_bias[0];
		for (int j=0;j<UNITS;j++) {
			state[j] =
				sigm(gates[UNITS + j]) * state[j] +
				sigm(gates[i]) * tanhf(gates[UNITS*3+i]);

			output[j] =
				sigm(gates[UNITS*2+j]) + sigm(state[j]);

			// done with LSTM, now perform dense layer
			sum += output[j] * mynet.dense_weights[j];
		}

		predicted_pin[i] = sum;
	}
	
	// plot results
	plot(time,pin,"pin position",
		 predicted_pin,"pred. pin position",
		 "LSTM results",nsamples); 
	return 0;
}

