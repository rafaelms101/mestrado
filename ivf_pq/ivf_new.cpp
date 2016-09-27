#include "ivf_new.h"
#include "../pq-utils/pq_new.h"

ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain){

  ivfpq_t ivfpq;
  ivfpq.coarsek = ivfpq.coa_centroidsn = coarsek;
  ivfpq.coa_centroidsd = vtrain.d;
  ivfpq.coa_centroids = (float*)malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);

  float * dis = (float*)malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);

  //definicao de variaveis
  int flags = flags & KMEANS_INIT_RANDOM;
  int* assign = (int*)malloc(sizeof(int)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);

  kmeans(vtrain.d, vtrain.n, coarsek, 50, vtrain.mat, flags, 1, 1, ivfpq.coa_centroids, NULL, NULL, NULL);

  //calculo do vetores residuais
  knn_full(L2, ivfpq.coa_centroidsn, vtrain.n, ivfpq.coa_centroidsd, 1, vtrain.mat, ivfpq.coa_centroids, NULL, assign, dis);
  subtract(vtrain, ivfpq.coa_centroids, assign);

  //aprendizagem do produto residual
  ivfpq.pq = pq_new(nsq, vtrain);

  free(assign);
  free(dis);

  return ivfpq;
}


void subtract(mat v, float* v2, int* idx){
  int VEC = 0;

  for (int i = 0; i < v.d*v.n; i+=v.d) {
    for (int j = 0; j < v.d; j++) {
      v.mat[i+j] = v.mat[i+j] - v2[idx[VEC]];
    }
    VEC++;
  }
}


// ivfpq.coa_centroids = (float**)malloc(sizeof(float*)*nsq);
// for (int i=0; i<nsq; i++){
//   ivfpq.coa_centroids[i]=(float *) malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);
// }
