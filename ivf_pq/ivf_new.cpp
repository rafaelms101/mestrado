#include "ivf_new.h"

ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain){

  ivfpq_t ivfpq;
  ivfpq.coarsek = ivfpq.centroidsn = coarsek;
  ivfpq.centroidsd = vtrain.d;
  ivfpq.centroids = (float*)malloc(sizeof(float)*ivfpq.centroidsn*ivfpq.centroidsd);

  float * dis = (float*)malloc(sizeof(float)*ivfpq.centroidsn*ivfpq.centroidsd);

  //definicao de variaveis
  int flags = flags & KMEANS_INIT_RANDOM;
  int* assign = (int*)malloc(sizeof(int)*ivfpq.centroidsn*ivfpq.centroidsd);

  kmeans(vtraind.d, vtrain.n, coarsek, vtrain.mat, flags, 1, ivfpq.centroids, NULL, NULL, NULL);

  //calculo do vetores residuais
  knn_full(L2, ivfpq.centroidsn, vtrain.n, ivfpq.centroidsd, 1, vtrain.mat, ivfpq.centroids, NULL, assign, dis);
  subtract(vtrain, ivfpq.centroids, assign);

  //aprendizagem do produto residual
  ivfpq.pq = pq_new(nsq, vtrain);

  free(assign);
  free(dis);

  return ivfpq;
}


void subtract(mat v, float* v2, int* idx){
  int VEC = 0

  for (int i = 0; i < v.d*v.n; i+=v.d) {
    for (int j = 0; j < v.d; j++) {
      v[i+j] = v[i+j] - v2[idx[VEC]]
    }
    VEC++;
  }
}
