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

matI pq_assign (pqtipo pq, mat v){

    mat vsub;

    //aloca um vetor de varios subvetores de dimensao ds
    vsub.mat = (float*)malloc(sizeof(float)*(pq.ds)*(v.n));
    vsub.d=pq.ds;
    vsub.n=v.n;
    int* assigns = (int*)malloc(sizeof(int)*v.n);    //indice dos k elementos mais proximos
    float* dis = (float*)malloc(sizeof(float)*v.n);      //distancia deles para os definitivos

    //codebook a ser gerado
    static matI code;
    code.mat = (int*)malloc(sizeof(int)*v.n*pq.nsq);
    code.n = v.n;
    code.d = 0;

    for (int i = 0; i < pq.nsq ; i++) {
        copySubVectors(vsub.mat, v, pq.ds,i, 0, (v.n)-1);
        //TODO modificar knn, para não precisar realizar as copias
        knn_full(2, vsub.n, pq.ks, pq.ds, 1 ,
                 pq.centroids[i], vsub.mat, NULL , assigns, dis);
        ivec_concat_transp(code,assigns,pq.nsq);
        code.d++;  
    }

    free(vsub.mat);
    free(assigns);

    return code;
}