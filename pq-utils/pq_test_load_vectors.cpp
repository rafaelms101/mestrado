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

    for(i=1;i<=d;i++){
    	for(j=0;j<v.train.n;j++){
    		v.train.mat[i*j]= (float) rand()/RAND_MAX;
    	}
    	for(j=0;j<v.base.n;j++){
    		v.base.mat[i*j]= (float) rand()/RAND_MAX;
    	}
    	for(j=0;j<v.query.n;j++){
    		v.query.mat[i*j]= (float) rand()/RAND_MAX;
    	}
    }

    return v;
}