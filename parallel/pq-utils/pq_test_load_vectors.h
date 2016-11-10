#ifndef H_LOAD_VECTORS
#define H_LOAD_VECTORS

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
extern "C" {
#include "../yael_needs/matrix.h"
#include "../yael_needs/vector.h"	
#include "../yael_needs/nn.h"		
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

mat pq_test_load_train(char* dataset);
mat pq_test_load_base(char* dataset);
mat pq_test_load_query(char* dataset);
data pq_test_load_vectors(char *);
void load_random (float *v, int n, int d);
int ivecs_read (const char *fname, int d, int n, int *a);

#endif
