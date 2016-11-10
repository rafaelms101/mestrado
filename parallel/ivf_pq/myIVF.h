#ifndef H_MYIVF
#define H_MYIVF

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include "../pq-utils/pq_assign.h"
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_search.h"
extern "C" {
	#include "../yael_needs/vector.h"
	#include "../yael_needs/kmeans.h"
}

typedef struct ivfpq{
	pqtipo pq;
	int coarsek;
	float *coa_centroids;
	int coa_centroidsn;
	int coa_centroidsd;
}ivfpq_t;

typedef struct ivf{
	int* ids;
	int idstam;
	matI codes;
}ivf_t;

#endif