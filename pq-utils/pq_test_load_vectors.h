#ifndef H_LOAD_VECTORS
#define H_LOAD_VECTORS

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
extern "C" {
#include "../yael/matrix.h"
#include "../yael/vector.h"	
}

using namespace std;

typedef struct{
	float *mat;
	int n,
		d;
}mat;

typedef struct{
	mat train,
		base,
		query;
}data;

data pq_test_load_vectors();

void load_random (float *v, int n, int d);

#endif
