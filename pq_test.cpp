#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "pq-utils/pq_assign.h"
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_search.h"
extern "C" {
#include "yael/vector.h"
#include "yael/kmeans.h"
#include "yael/ivf.h"
}

int main(int argv, char** argc){
	struct timeval inicio, final;

	int nsq,
		k,
		tmili,
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

	gettimeofday(&inicio, NULL);
	v = pq_test_load_vectors(dataset);
	gettimeofday(&final, NULL);
	tmili = (int) (1000 * (final.tv_sec - inicio.tv_sec) + (final.tv_usec - inicio.tv_usec) / 1000);
	printf("Tempo load vectors: %d\n", tmili);
	
	nsq=8;
	k=100;
	dis = (float*)malloc(sizeof(float)*v.query.n*k);
	ids = (int*)malloc(sizeof(int)*v.query.n*k);

	gettimeofday(&inicio, NULL);
	pq = pq_new(nsq, v.train);
	gettimeofday(&final, NULL);
	tmili = (int) (1000 * (final.tv_sec - inicio.tv_sec) + (final.tv_usec - inicio.tv_usec) / 1000);
	printf("Tempo new: %d\n", tmili);

	//for(int k=0; k<256*16; k++){
	//	if(k%16==0)printf("\ncoluna%d 	", k/16);;
	//	printf("%g ", pq.centroids[1][k]);
	//}

	gettimeofday(&inicio, NULL);
	codebook = pq_assign(pq, v.base);
	gettimeofday(&final, NULL);
	tmili = (int) (1000 * (final.tv_sec - inicio.tv_sec) + (final.tv_usec - inicio.tv_usec) / 1000);
	printf("Tempo assign: %d\n", tmili);

	//for(int k=0; k<80000; k++){
	//	if(k%8==0)printf("\ncoluna%d 	", k/8);;
	//	printf("%d ", codebook.mat[k]);
	//}

	gettimeofday(&inicio, NULL);
	pq_search(pq, codebook, v.query, k, dis , ids);
	gettimeofday(&final, NULL);
	tmili = (int) (1000 * (final.tv_sec - inicio.tv_sec) + (final.tv_usec - inicio.tv_usec) / 1000);
	printf("Tempo search: %d\n", tmili);

	//for(int k=0; k<10000; k++){
	//	if(k%100==0)printf("\ncoluna%d 	", k/100);;
	//	printf("%d ", ids[k]);
	//}

	free(dis);
	free(ids);
	free(pq.centroids);
	free(codebook.mat);
	free(v.base.mat);
	free(v.query.mat);
	free(v.train.mat);

	return 0;
}
