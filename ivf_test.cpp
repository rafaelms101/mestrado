#include <stdio.h>
#include <math.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_new.h"
#include "ivf_pq/ivf_search.h"

int main(int argv, char **argc){
  	int k,
  		nsq,
  		coarsek,
  		w;
  	char* dataset;

  	data v;

	if(argv < 2){
		cout << "Usage: ./pq_test <dataset>" << endl;
		return -1;
	}

	dataset = argc[1];

  	v = pq_test_load_vectors(dataset);

  	k = 100;
  	nsq = 8;
  	coarsek = 256;
  	w = 4;

  	ivfpq_t ivfpq = ivfpq_new(coarsek, nsq, v.train);


	ivf_t *ivf = ivfpq_assign(ivfpq, v.base);

	int *ids = (int*)malloc(sizeof(int)*v.query.n*k);
	float *dis = (float*)malloc(sizeof(float)*v.query.n*k);

	ivfpq_search(ivfpq, ivf, v.query, k, w, ids, dis);


  return 0;
}
