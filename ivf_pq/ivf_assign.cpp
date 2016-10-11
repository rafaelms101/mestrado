#include "ivf_assign.h"

ivf* ivfpq_assign(ivfpq_t ivfpq, mat vbase){
  int k = 1;
  // printf("ivfpq.coa_centroidsn = %d\n", ivfpq.coa_centroidsn);
  // printf("ivfpq.coa_centroidsd = %d\n", ivfpq.coa_centroidsd);
  // printf("k = %d\n", k);
  // printf("vbase.n = %d\n", vbase.n);
  // printf("vbase.d = %d\n", vbase.d);

  int *assign = (int*)malloc(sizeof(int)*vbase.n);
  float *dis = (float*)malloc(sizeof(float)*vbase.n);

  //acha os indices para o coarse quantizer
  knn_full(L2, vbase.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, k,
           ivfpq.coa_centroids, vbase.mat, NULL, assign, dis);

  //residuos
  subtract(vbase, ivfpq.coa_centroids, assign, ivfpq.coa_centroidsd);

  matI codebook = pq_assign(ivfpq.pq, vbase);

  // printMatI(codebook.mat, codebook.n, codebook.d);
  // getchar();

  static ivf_t* ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

  int* hist =  histogram(assign, vbase.n ,ivfpq.coarsek);
  //ivec_print(hist, ivfpq.coarsek);

   // -- Sort on assign, new codebook with sorted ids as identifiers for codebook
   int* ids = (int*)malloc(sizeof(int)*vbase.n);
   ivec_sort_index(assign, vbase.n, ids);

   int pos = 0, nextpos;
   for (int i = 0; i < ivfpq.coarsek; i++) {
     ivf[i].ids = (int*)malloc(sizeof(int)*hist[i]);
     ivf[i].idstam = hist[i];
     ivf[i].codes.mat = (int*)malloc(sizeof(int)*hist[i]*ivfpq.pq.nsq);
     ivf[i].codes.n = hist[i];
     ivf[i].codes.d = ivfpq.pq.nsq;

     nextpos = pos+hist[i];
     memcpy(ivf[i].ids, ids+pos, sizeof(int)*hist[i]);
     //memcpy(ivf[i].codes.mat, codebook.mat+pos, sizeof(int)*hist[i]*ivfpq.pq.nsq);
     for (int p = pos; p < nextpos; p++) {
       copySubVectorsI(ivf[i].codes.mat, codebook.mat, p, 0, ivf[i].codes.d);
     }
     pos += hist[i];

     printMatI(ivf[i].codes.mat, ivf[i].codes.n, ivf[i].codes.d);
     getchar();

    //  printMatI(ivf[i].codes.mat, ivf[i].codes.n ,ivf[i].codes.d );
   }

   free(hist);
   return ivf;
}

int* histogram(const int* vec, int n, int range){
  static int * hist = (int*) malloc(sizeof(int)*range);

  for (int i = 0; i < range; i++) {
    hist[i] = (int) ivec_count_occurrences(vec, n, i);
  }
  // for (int i = 0; i < ; i++) {
  //   hist[vec[i]] = (int) ivec_count_occurrences(vec, n, i);
  // }

  //printf("range = %d\n", range);
  return hist;
}
