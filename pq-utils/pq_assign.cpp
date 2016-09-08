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

    fvec_print(pq.centroids.mat , pq.centroids.n );

    //aloca um vetor de varios subvetores de dimencao ds
    float* vsub = (float*)malloc(sizeof(float)*(pq.ds)*(v.n));
    int* assigns = (int*)malloc(sizeof(int)*v.n);    //indice dos k elementos mais proximos
    float* dis;      //distancia deles para os definitivos

    //codebook a ser gerado
    static matI code;
    code.mat = NULL;
    code.n = 0;
    code.d = pq.nsq;

    for (int i = 0; i < pq.nsq ; i++) {
        printf("ds = %d\n", i);
        copySubVectors(vsub, v, pq.ds, 0, v.n-1);
        //TODO modificar knn, para não precisar realizar as copias
        knn_full(L2, pq.centroids.n / pq.centroids.d , (v.n/v.d), pq.ds, 100, vsub,
                pq.centroids.mat , NULL, assigns, dis);

        ivec_print(assigns, v.n);
        ivec_concat(code.mat, code.n, assigns, v.n);
        code.n += v.n;
    }

    free(vsub);
    free(assigns);

    printf("code entght  = %d\n", code.n);

    return code;
}
