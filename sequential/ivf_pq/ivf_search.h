#ifndef H_IVFSEARCH
#define H_IVFSEARCH

	#include<stdio.h>
	#include<stdlib.h>
	extern "C"{
	#include "../yael_needs/vector.h"
	#include "../yael_needs/nn.h"
	#include "../yael_needs/kmeans.h"
	}
	#include "../pq-utils/pq_test_load_vectors.h"
	#include "../pq-utils/pq_new.h"
	#include "../pq-utils/pq_search.h"
	#include "myIVF.h"
	#include "ivf_new.h"

	int min(int a, int b);
	float * sumidxtab2(mat D, matI ids, int offset);
	mat bsxfunMINUS(mat vin, float* vin2, int dim, int nq, int* qcoaidx, int ncoaidx);
	void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, int k, int w, int* ids, float* dis, mat vbase);
	int* imat_new_transp (const int *a, int ncol, int nrow);
	void fillY(float** y, int ids, float* v , ivfpq_t ivfpq, ivf_t *ivf, int k);
	void L2_distance(float* vin, float* vin2, int n, float* dis);
	void myKnn_single(float* v, float *base, int basen, int ds ,int *idxsaida);


#endif
