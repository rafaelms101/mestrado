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
	int nsq,
		k,
		*ids;

	float 	*dis;
			
	matI codebook;
	data v;
	pqtipo pq;

	check_new();
	check_assign();

	v = pq_test_load_vectors();
	
	nsq=8;
	k=100;
	dis = (float*)malloc(sizeof(float)*v.query.n*k);
	ids = (int*)malloc(sizeof(int)*v.query.n*k);

	pq = pq_new(nsq, v.train);

	codebook = pq_assign(pq, v.base);

	pq_search(pq, codebook, v.query, k, dis , ids);

	fvec_print(dis,v.query.n*k);
	//ivec_print(ids,v.query.n*k);

	free(dis);
	free(ids);
	free(pq.centroids);
	free(codebook.mat);
	free(v.base.mat);
	free(v.query.mat);
	free(v.train.mat);

	return 0;
}
