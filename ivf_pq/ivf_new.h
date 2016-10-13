#ifndef H_IVFNEW
#define H_IVFNEW

	#include <stdio.h>
	#include <stdlib.h>
	extern "C"{
	#include "../yael_needs/vector.h"
	#include "../yael_needs/nn.h"
	#include "../yael_needs/kmeans.h"
	}
	#include "../pq-utils/pq_test_load_vectors.h"
	#include "../pq-utils/pq_new.h"
	#include "myIVF.h"

	void subtract(mat v, float* v2, int* idx, int c_d);
	ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain);
	void printMat(float* mat, int n, int d);
	void printMatI(int* mat, int n, int d);
	void copySubVectorsI(int* qcoaidx, int* coaidx, int query, int nq,int w);
	void copySubVectors2(float* vout, float* vin, int dim, int nvec, int subn);


#endif
