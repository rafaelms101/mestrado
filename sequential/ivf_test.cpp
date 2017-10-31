#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"
#include "pq-utils/pq_test_compute_stats.h"

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_new.h"
#include "ivf_pq/ivf_search.h"

int main(int argv, char **argc){

	struct timeval start, end;
  	int k,
  		nsq,
  		coarsek,
  		w;
  	char* dataset;

  	data v;

	if(argv != 7){
		cout << "Usage: ./ivfpq_test <dataset>  <k> <kl> <coarsek> <w> <nsq>" << endl;
		return -1;
	}

	dataset = argc[1];

  	k = atoi(argc[2]);
	int kl = atoi(argc[3]);

	coarsek = atoi(argc[4]);

	w = atoi(argc[5]);
	nsq = atoi(argc[6]);

	gettimeofday(&start, NULL);
	v = pq_test_load_vectors(dataset);
	gettimeofday(&end, NULL);
	printf("Loading vectors %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

	mat vbase;

	vbase.d = v.base.d;
	vbase.n = v.base.n;
	vbase.mat = (float*)malloc(sizeof(float)*vbase.n*vbase.d);
	memcpy(vbase.mat, v.base.mat, sizeof(float)*vbase.n*vbase.d);

	gettimeofday(&start, NULL);
	ivfpq_t ivfpq = ivfpq_new(coarsek, nsq, v.train);
	gettimeofday(&end, NULL);
	printf("Learnig %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

	free(v.train.mat);

	gettimeofday(&start, NULL);
	ivf_t *ivf = ivfpq_assign(ivfpq, v.base);
	gettimeofday(&end, NULL);
	printf("Encoding %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

	free(v.base.mat);

	int *ids = (int*)malloc(sizeof(int)*v.query.n*k);
	float *dis = (float*)malloc(sizeof(float)*v.query.n*k);

	gettimeofday(&start, NULL);
	ivfpq_search(ivfpq, ivf, v.query, vbase, k, kl, w, ids, dis);
	gettimeofday(&end, NULL);
	printf("Searching %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

	free(vbase.mat);

	pq_test_compute_stats2(ids, v.ids_gnd, k);

	free(dis);
	free(ids);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	free(v.query.mat);

	for (int i = 0; i < ivfpq.coarsek ; i++) {
		free(ivf[i].codes.mat);
		free(ivf[i].ids);
	}

	return 0;
}
