#ifndef H_IVFSEARCH
#define H_IVFSEARCH

  #include<stdio.h>
  #include<stdlib.h>
  extern "C"{
  #include "../yael/vector.h"
  #include "../yael/nn.h"
  #include "../yael/kmeans.h"
  }
  #include "../pq-utils/pq_test_load_vectors.h"
  #include "../pq-utils/pq_new.h"
  #include "../pq-utils/pq_search.h"
  #include "myIVF.h"

  void ivfpq_search(ivfpq_t ivfpq, ivf_t ivf, mat vquery, int k, int w, int* ids, int* dis);
  float* bsxfunMINUS(float* vin, float* vin2, int dim, int nq, int nqcoaidx);


#endif
