#include "ivf_search.h"

void ivfpq_search(ivfpq_t ivfpq, ivf_t ivf, mat vquery, int k, int w, int* ids, int* dis){
  int nq, d,ds, ks, nsq;
  nq = vquery.n;
  d = vquery.d;
  ds = ivfpq.pq.ds;
  ks = ivfpq.pq.ks;
  nsq =  ivfpq.pq.nsq;

  int qcoaidx;
  float *v;

  float* distab = (float*)malloc(sizeof(float)*ks*nsq);
  float* dis = (float*)malloc(sizeof(float)*nq*k);
  int* ids = (int*)malloc(sizeof(int)*nq*k);

  int* coaidx = (int*)malloc(sizeof(int)*vquery.n);
  float* coadis = (float*)malloc(sizeof(float)*vquery.n);


  mat vsub;

  //find the w vizinhos mais prximos
  knn_full(2, nq, ivfpq.coa_centroidsn, d, w,
           ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);

  for (int query = 0; query < nq; query++) {
      qcoaidx =  coaidx[i];

      //compute the w residual vectors
      v = bsxfunMINUS(vquery, ivfpq.coa_centroids, vquery.d, query, qcoaidx);

      //TODO --- qidx, qids

      for (int j = 0; j < w; j++) {
        for (int q = 0; q < nsq; q++) {
          // copySubVectors(vsub.mat,vquery ,pq.ds,q,j,j);
    			// compute_cross_distances(vsub.d, vsub.n, pq.ks, vsub.mat, pq.centroids[j], distab_temp);
    			// memcpy(distab.mat+j*pq.ks*vsub.n, distab_temp, sizeof(float)*pq.ks*vsub.n);
        }
      }
  }



}


//
float* bsxfunMINUS(float* vin, float* vin2, int dim, int nq, int nqcoaidx){
  static float* vout = (float*)malloc(sizeof(float)*dim);

  for (int i = 0; i < dim; i++) {
    vout[i] = vin[nq*dim-1+i] - vin2[nqcoaidx*dim-1+i];
  }

  return vout;
}
