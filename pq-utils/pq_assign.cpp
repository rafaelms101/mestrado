#include "pq_assign.h"

using namespace std;

void check_assign(){
  cout << ":::PQ_ASSIGN OK:::" << endl;
}

/*
* pq : estrutura do quantizador
* v : base de vetores a ser usada
* n : tamanho do vetor
*/

int* pq_assign (pqtipo pq, mat v, int n){

  //aloca um vetor de varios subvetores de dimencao ds
  float* vsub = (float*)malloc(sizeof(float)*(pq.ds)*(v.n/v.d));
  int* assigns = (int*)malloc(sizeof(int)*100*v.n);    //indice dos k elementos mais proximos
  float* dis;      //distancia deles para os definitivos

  //codebook a ser gerado
  int* code = NULL;
  int code_n = 0;
  //code = (int*)malloc(sizeof(int)*100*pq.centroids.n);

  for (int i = 0; i < pq.nsq ; i++) {

    copySubVectors(vsub, v, (i)*pq.ds, (i+1)*pq.ds);
    //TODO modificar knn, para não precisar realizar as copias
    knn_full(L2, pq.centroids.n / pq.centroids.d , (v.n/v.d), pq.ds, 100, vsub,
             pq.centroids.mat , NULL, assigns, dis);

    ivec_concat(code, code_n, assigns, 100*v.n);
    code_n += 100*v.n;

  }

  return code;
}

void copySubVectors(float *vout, mat vin, int ini, int fim) {
  int ds = fim-ini;

  for (int i = 0; i < ds ; i+= ds) {
      memcpy(vout + i, vin.mat + i + ini, sizeof(float)*(fim - ini));
  }

}

// void createIdx(int ini, int fim, int* vout, int n, int d){
//   if(vout != NULL){
//     free(vout);
//   }
//   vout = (int *)malloc(sizeof(int)* (fim-ini));
//
//   for(int i = 0; i < n; i =  i + (n/d)){
//       vout[i] = i;
//   }
//
//   return ;
// }
