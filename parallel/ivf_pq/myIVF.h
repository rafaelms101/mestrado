#ifndef H_MYIVF
#define H_MYIVF

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include "../pq-utils/pq_assign.h"
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_search.h"

extern "C" {
	#include "../yael_needs/vector.h"
	#include "../yael_needs/kmeans.h"
}

typedef struct {
	mat dis;
	matI idx;
}dis_t;

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

typedef struct ivf_threads{
	ivf_t *ivf;
	ivfpq_t ivfpq;
	int thread;
	int k;
}ivf_threads_t;

void my_k_min(dis_t q, int ktmp, float *dis, int *ids);
static void constroiHeap (int n, float *qdis, int *qidx);
static void trocarRaiz (int n, float *qdis, int *qidx);

#endif