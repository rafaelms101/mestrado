#ifndef H_MYIVF
#define H_MYIVF

#include <stdio.h>
#include <math.h>
#include "../pq-utils/pq_assign.h"
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_search.h"
extern "C" {
#include "../yael/vector.h"
#include "../yael/kmeans.h"
#include "../yael/ivf.h"
}


  typedef struct ivfpq{
    pqtipo pq;
    int coarsek;
    float **coa_centroids;
    int coa_centroidsn;
    int coa_centroidsd;
  }ivfpq_t;

  typedef struct ivf{
    int* ids;
    int* codes;
  }ivf_t;



#endif
