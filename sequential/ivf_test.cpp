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
  	int k, kl=0,
  		nsq,
  		coarsek,
  		w;
  	char* dataset;

  	data v;

	if(argv < 3 && argv > 4){
		cout << "Usage: ./pq_test <dataset>  <k>  <kl>" << endl;
		return -1;
	}

	dataset = argc[1];

  	k = atoi(argc[2]);
  	nsq = 8;
  	coarsek = 256;
  	w = 4;

    #ifdef RERANK
      printf("RE-RANKING ==== ON\n");
      kl = atoi(argc[3]);
    #endif

  gettimeofday(&start, NULL);
  v = pq_test_load_vectors(dataset);
  gettimeofday(&end, NULL);
	printf("Loading vectors %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

  gettimeofday(&start, NULL);
  ivfpq_t ivfpq = ivfpq_new(coarsek, nsq, v.train);
  gettimeofday(&end, NULL);
	printf("Learnig %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

  gettimeofday(&start, NULL);
  ivf_t *ivf = ivfpq_assign(ivfpq, v.base);
  gettimeofday(&end, NULL);
	printf("Encoding %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);

  #ifdef RERANK
    int *ids = (int*)malloc(sizeof(int)*v.query.n*kl);
    float *dis = (float*)malloc(sizeof(float)*v.query.n*kl);
  #else
    int *ids = (int*)malloc(sizeof(int)*v.query.n*k);
    float *dis = (float*)malloc(sizeof(float)*v.query.n*k);
  #endif

  gettimeofday(&start, NULL);
  ivfpq_search(ivfpq, ivf, v.query, k, kl, w, ids, dis, v.base);
  gettimeofday(&end, NULL);
  printf("Searching %lfs\n", difftime(end.tv_sec, start.tv_sec)+ (double) (end.tv_usec - start.tv_usec)/1000000);


  #ifdef RERANK
    pq_test_compute_stats2(ids, v.ids_gnd, kl);
  #else
    pq_test_compute_stats2(ids, v.ids_gnd, k);
  #endif

  free(dis);
  free(ids);
  free(ivfpq.pq.centroids);
  free(ivfpq.coa_centroids);
  free(v.base.mat);
  free(v.query.mat);
  free(v.train.mat);
  for (int i = 0; i < ivfpq.coarsek ; i++) {
    free(ivf[i].codes.mat);
    free(ivf[i].ids);
  }

  return 0;
}
