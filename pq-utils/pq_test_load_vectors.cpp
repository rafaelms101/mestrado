#include "pq_test_load_vectors.h"

data pq_test_load_vectors(){
	int i,
		j,
        d=16;

	data v;    

	v.train.n=10000;
    v.train.d=d;
    v.train.mat= fmat_new (v.train.d, v.train.n);

    v.base.n=1000000;
    v.base.d=d;
    v.base.mat= fmat_new (v.base.d, v.base.n);

    v.query.n=1000;
    v.query.d=d;
    v.query.mat= fmat_new (v.query.d, v.query.n);

    srand (time(NULL));

    ///inicializa com valores aleatórios

    load_random(v.train.mat, v.train.n, v.train.d);
    load_random(v.base.mat, v.base.n, v.base.d);
    load_random(v.query.mat, v.query.n, v.query.d);

    return v;
}

void load_random (float *v, int n, int d){
    int i;

    for(i=0;i<n*d;i++){
        v[i]= (float) rand()/RAND_MAX;
    }
}