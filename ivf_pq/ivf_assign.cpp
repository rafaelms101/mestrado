#include "ivf_assign.h"

ivf* ivf_assign(ivfpq_t ivfpq, mat vbase){
  int k = 1;
  int* codebook;

  int *assign = (int*)malloc(sizeof(int)*ivfpq.centroidsn*k);
  float *dis = (float*)malloc(sizeof(float)*ivfpq.centroidsn*k);

  //acha os indices para o coarse quantizer
  knn_full(L2, ivfpq.centroidsn, vbase.n, ivfpq.centroidsd, k, vbase.mat, ivfpq.centroids, NULL, assign, dis);

  //residuos
  subtract(vbase, ivfpq.centroids, assign);

  codebook = pq_assign(ivfpq.pq, vbase);

   static ivf_t* ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

   float *assignf;
   ivec_to_fvec(assign, assignf, ivfpq.centroidsn*k);

   int* hist = fvec_new_histogram_clip (-1.0 , 1.0 , ivfpq.coarsek , assignf, ivfpq.centroidsn*k);

   //TODO
   // -- Sort on assign, new codebook with sorted ids as identifiers for codebook

   
   int pos = 0, nextpos;
   for (int i = 0; i < ivfpq.coarsek; i++) {
     ivf[i].ids = (int*)malloc(sizeof(int)*hist[i]);
     ivf[i].codes = (int*)malloc(sizeof(int)*hist[i]*vbase.d);

     nextpos = pos+hist[i];
     memcpy(ivf[i].ids, ids+pos, sizeof(int)*hist[i]);
     memcpy(ivf[i].codes, codebook+pos+i*vbase.d, sizeof(int)*hist[i]*vbase.d);
     pos += hist[i];
   }

   return ivf;
}
