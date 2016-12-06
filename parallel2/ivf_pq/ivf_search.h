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

	typedef struct {
	mat dis;
	matI idx;
	}dis_t;

	int min(int a, int b);
	void bsxfunMINUS(mat mout, mat vin, float* vin2, int nq, int* qcoaidx, int ncoaidx);
	dis_t ivfpq_search(ivf_t *ivf, mat residual, pqtipo pq, int centroid_idx);
	int* imat_new_transp (const int *a, int ncol, int nrow);

#endif
