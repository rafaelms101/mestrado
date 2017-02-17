#ifndef H_LOAD_VECTORS
#define H_LOAD_VECTORS

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <string.h>
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
	char 	*train,
			*groundtruth;
}namefile;

data pq_test_load_vectors(char * dataset, int tam, int my_rank);
mat pq_test_load_query(char* dataset);
mat pq_test_load_base(char* dataset, int my_rank, int num, int offset);
void load_random (float *v, int n, int d);
int ivecs_read (const char *fname, int d, int n, int *a);
int my_bvecs_read (int offset, const char *fname, int d, int n, float *a);

#endif
