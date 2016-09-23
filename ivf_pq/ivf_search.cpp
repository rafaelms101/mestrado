#include "ivf_search.h"

void ivfpq_search(ivfpq_t ivfpq, ivf_t ivf, mat vquery, int k, int w, int* ids, int* dis){
  int nq, d,ds, ks, nsq;
  nq = vquery.n;
  d = vquery.d;
  ds = ivfpq.pq.ds;
  ks = ivfpq.pq.ks;
  nsq =  ivfpq.pq.nsq;

  float* distab = (float*)malloc(sizeof(float)*ks*nsq);
  float* dis = (float*)malloc(sizeof(float)*nq*k);
  int* ids = (int*)malloc(sizeof(int)*nq*k);

  int* coaidx = (int*)malloc(sizeof(int)*nq*k);
  float* coadis = (float*)malloc(sizeof(float)*nq*k);

  //find the w vizinhos mais prximos
  knn_full(2, nq, ivfpq.coa_centroidsn, d, w,
           ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);

  for (int i = 0; i < nq; i++) {

    memcpy(qcoaidx, coaidx + i*d*nq, sizeof(int)*d);
  }

}
