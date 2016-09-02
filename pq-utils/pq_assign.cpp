#include <iostream>
#include <math.h>
extern "C" {
#include "../yael/vector.h"
#include "../yael/nn.h"
}

/*
* pq : estrutura do quantizador
* v : base de vetores a ser usada
* n :
*/

int* pq_assign (pqtipo pq, float *v, int n){

  int* idx = NULL;
  float* vsub = (float*)malloc(sizeof(float)*(ds));
  int* assigns;
  float* dis;

  for (int i = 0; i < pq.m; i++) {

    createIdx((i-1)*pq.ds, i*pq.ds, idx);
    fvec_cpy_subvectors (v, idx, d, n, vsub);
    //TODO modificar knn, para não precisar realizar as copias
    knn_full(L2, nq, nb, pq.ds, 100, vsub, pq.centroids, NULL, assigns, dis);
    //TODO

  }

  return
}

void createIdx(int ini, int fim, int* vout){
  if(vout != NULL){
    free(vout);
  }
  vout = (int *)malloc(sizeof(int)* (fim-ini));

  for (int i = ini; i < fim+1; i++) {
    idx[i] = i;
  }

  return
}
