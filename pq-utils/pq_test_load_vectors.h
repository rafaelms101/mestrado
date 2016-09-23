#ifndef H_LOAD_VECTORS
#define H_LOAD_VECTORS

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
extern "C" {
#include "../yael/matrix.h"
#include "../yael/vector.h"	
#include "../yael/nn.h"		
}

using namespace std;

typedef struct{
	float *mat;
	int n,
		d;
}mat;

typedef struct{
	int *mat;
	int n,
		d;
}matI;

typedef struct{
	mat train,
		base,
		query;
	
	matI ids_gnd;
}data;

typedef struct{
	char 	*base,
			*query,
			*train,
			*groundtruth;
}namefile;

data pq_test_load_vectors(char *);
void load_random (float *v, int n, int d);
int ivecs_read (const char *fname, int d, int n, int *a);

#endif
