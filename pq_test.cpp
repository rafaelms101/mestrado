#include <stdio.h>
#include <math.h>
#include "pq-utils/pq_assign.h"
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_search.h"
extern "C" {
#include "yael/vector.h"
#include "yael/kmeans.h"
#include "yael/ivf.h"
}

int main(){
	int nsq=8;
	matI codebook;
	data v;
	pqtipo pq;

	check_new();
	check_assign();

	int *vec = ivec_new_set(2, 0);
	int *vec2 = ivec_new_set(2, 1);

	printf("vec = ");
	ivec_print(vec, 2);

	printf("vec2 = ");
	ivec_print(vec2, 2);

	ivec_concat(vec, 2, vec2, 2);

	printf("vec = ");
	ivec_print(vec, 4);
	free(vec);
	free(vec2);

	v = pq_test_load_vectors();

	mat BASE;
	BASE.mat = (float*)malloc(sizeof(float)*v.base.n*v.base.d);
	BASE.n = v.base.n;
	BASE.d = v.base.d;
	memcpy(BASE.mat, v.base.mat, sizeof(float)*BASE.n*BASE.d);

	pq = pq_new(nsq, v.train);

	//fvec_print(pq.centroids.mat, pq.centroids.n);

	//fvec_print(BASE.mat, BASE.n*BASE.d);
	codebook = pq_assign(pq, BASE);

	//free(pq.centroids.mat);
	//free(codebook.mat);
	//free(base.base.mat);
	//free(base.query.mat);
	//free(base.train.mat);

	return 0;
}
