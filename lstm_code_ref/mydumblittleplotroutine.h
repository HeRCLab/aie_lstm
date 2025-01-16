#if defined(USE_APPROX_ACTIV_FNS)
#define DATA0_FNAME "data0_approx.txt"
#define DATA1_FNAME "data1_approx.txt"
#else
#define DATA0_FNAME "data0.txt"
#define DATA1_FNAME "data1.txt"
#endif


// hardcoded for exactly two signals
void plot (float *time,
		   float *sig1,const char *name1,
		   float *sig2,const char *name2,
		   const char *title,
		   int nsamples) {

	// dump signal
	char str[4096];
	FILE *myFile[2];

	// open a file to dump each trace
	myFile[0] = fopen(DATA0_FNAME,"w+");
	if (!myFile[0]) {
		perror("ERROR: can't create temp file in PWD");
		exit(1);
	}
	myFile[1] = fopen(DATA1_FNAME,"w+");
	if (!myFile[1]) {
		perror("ERROR: can't create temp file in PWD");
		exit(1);
	}
	
	for (int i=0;i<nsamples;i++) {
		fprintf (myFile[0],"%0.12f %0.12f\n",time[i],sig1[i]);
		fprintf (myFile[1],"%0.12f %0.12f\n",time[i],sig2[i]);
	}

	// close all files
	for (int i=0;i<2;i++) fclose(myFile[i]);

	// set up plot command
	// add -p for persist
	snprintf(str,1024,"/usr/bin/gnuplot -p -e \""
			"set title '%s';"
			"set xlabel 'time (s)';"
			"set ylabel 'mm';"
			//"set key off;"
			"plot "
	                   "'" DATA0_FNAME "' title '%s' with lines,"
			   "'" DATA1_FNAME "' title '%s' with lines;\"",title,name1,name2);

	FILE *myplot = popen(str,"w");
	
	if (!myplot) {
		perror("Error opening gnuplot");
		exit(1);
	}
	
	pclose(myplot);
}

