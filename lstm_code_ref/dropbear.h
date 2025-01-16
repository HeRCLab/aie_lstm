int read_dropbear (float **time,float **vib,float **pin) {
	FILE *myFile = fopen("dropbear.dat","r");

	fseek(myFile,0,SEEK_END);
	int size = ftell(myFile) / 3 / sizeof(float);
	fseek(myFile,0,SEEK_SET);

	*time = (float *)malloc(size*sizeof(float));
	*vib = (float *)malloc(size*sizeof(float));
	*pin = (float *)malloc(size*sizeof(float));

	fread((void *)(*vib),sizeof(float),size,myFile);
	fread((void *)(*pin),sizeof(float),size,myFile);
	fread((void *)(*time),sizeof(float),size,myFile);	

	fclose(myFile);
	return size;
}

