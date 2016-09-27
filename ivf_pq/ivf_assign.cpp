#include "ivf_assign.h"

ivf* ivfpq_assign(ivfpq_t ivfpq, mat vbase){
  int k = 1;
  //int* codebook;
  printf("ivfpq.coa_centroidsn = %d\n", ivfpq.coa_centroidsn);
  printf("ivfpq.coa_centroidsd = %d\n", ivfpq.coa_centroidsd);
  printf("k = %d\n", k);
  printf("vbase.n = %d\n", vbase.n);
  printf("vbase.d = %d\n", vbase.d);

  int *assign = (int*)malloc(sizeof(int)*vbase.n);
  float *dis = (float*)malloc(sizeof(float)*vbase.n);

  //acha os indices para o coarse quantizer
  knn_full(L2, vbase.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, k,
           ivfpq.coa_centroids, vbase.mat, NULL, assign, dis);

  //ivec_print(assign, vbase.n);

  //residuos
  subtract(vbase, ivfpq.coa_centroids, assign);

  matI codebook = pq_assign(ivfpq.pq, vbase);

   static ivf_t* ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

  int* hist =  ivec_new_histogram(256, assign, vbase.n);
   ivec_print(hist, vbase.n);

   // -- Sort on assign, new codebook with sorted ids as identifiers for codebook
   int* ids = (int*)malloc(sizeof(int)*vbase.n);
   ivec_sort_index(assign, vbase.n, ids);

   int pos = 0, nextpos;
   for (int i = 0; i < ivfpq.coarsek; i++) {
     ivf[i].ids = (int*)malloc(sizeof(int)*hist[i]);
     ivf[i].codes = (int*)malloc(sizeof(int)*hist[i]*vbase.d);

     nextpos = pos+hist[i];
     memcpy(ivf[i].ids, ids+pos, sizeof(int)*hist[i]);
     memcpy(ivf[i].codes, codebook.mat+pos, sizeof(int)*hist[i]*vbase.d);
     pos += hist[i];
   }

   return ivf;
}
