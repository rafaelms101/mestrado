#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pq-utils/pq_assign.h"
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_search.h"
extern "C" {
#include "yael/vector.h"
#include "yael/kmeans.h"
#include "yael/ivf.h"
}

int main(int argv, char** argc){
	int nsq,
		k,
		*ids;

	float* dis;
	char* dataset ;
			
	matI codebook;
	data v;
	pqtipo pq;

	if(argv < 2){
		cout << "Usage: ./pq_test <dataset>" << endl;
		return -1;
	}

	dataset=argc[1];

	check_new();
	check_assign();

	v = pq_test_load_vectors(dataset);

	//for(int l=0; l<3200000; l++){
	//	if(l%128==0)printf("\ncoluna  %d", l/100);;
	//	printf("%g ", v.train.mat[l]);
	//}
	
	nsq=8;
	k=100;
	dis = (float*)malloc(sizeof(float)*v.query.n*k);
	ids = (int*)malloc(sizeof(int)*v.query.n*k);

	pq = pq_new(nsq, v.train);

	//for(int j=0; j<8; j++){
	//	printf("\nsubdimensao    %d", j);
	//	for(int i=0; i<4096; i++){
	//		if(i%256==0)printf("\ncoluna  %d", i/256);;
	//		printf("%g ", pq.centroids[j][i]);
	//	}
	//}

	codebook = pq_assign(pq, v.base);

	//for(int k=0; k<80000; k++){
	//	if(k%8==0)printf("\ncoluna%d 	", k/8);;
	//	printf("%d ", codebook.mat[k]);
	//}

	pq_search(pq, codebook, v.query, k, dis , ids);

	//fvec_print(dis,v.query.n*k);
	//ivec_print(ids,v.query.n*k);

	for(int k=0; k<10000; k++){
		if(k%100==0)printf("\ncoluna%d 	", k/100);;
		printf("%d ", ids[k]);
	}

	free(dis);
	free(ids);
	free(pq.centroids);
	free(codebook.mat);
	free(v.base.mat);
	free(v.query.mat);
	free(v.train.mat);

	return 0;
}
