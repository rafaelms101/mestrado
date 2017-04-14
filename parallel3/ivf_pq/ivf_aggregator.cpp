#include "ivf_aggregator.h"

void parallel_aggregator(int k, int w, int my_rank, int comm_sz, int tam_base){
	static int last_assign, last_search, last_aggregator;
	dis_t *q;
	matI ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, rest, id, rank, queryn, tam, l=0, ktmp, *in_q, in=0, ttam=0;
	double start=0, start2=0, end;
	char arquivo[15] = "testes.txt";

	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);
	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe o vetor contendo os indices da lista invertida
	MPI_Recv(&queryn, 1, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	coaidx = (int*)malloc(sizeof(int)*queryn*w);
	MPI_Recv(&coaidx[0], queryn*w, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	q = (dis_t*)malloc(sizeof(dis_t)*queryn);
	in_q = (int*)calloc(queryn,sizeof(in_q));

	for(int i=0; i<queryn; i++){
		q[i].dis.mat = (float*)malloc(sizeof(float));
		q[i].idx.mat = (int*)malloc(sizeof(int));
		q[i].dis.n = 0;
		q[i].idx.n = 0;
	}
	
	dis = (float*)malloc(sizeof(float));
	ids = (int*)malloc(sizeof(int));
	int i=0;
	printf("a");
	while(i<queryn*w*(last_search-last_assign)){
		printf("b");
		MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE , 100000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&id, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);	
		MPI_Recv(&tam, 1, MPI_INT, rank, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("c");
		q[(id-rank-1)/w].dis.mat = (float*)realloc(q[(id-rank-1)/w].dis.mat,sizeof(float)*(q[(id-rank-1)/w].dis.n+tam));
		q[(id-rank-1)/w].idx.mat = (int*)realloc(q[(id-rank-1)/w].idx.mat,sizeof(int)*(q[(id-rank-1)/w].idx.n+tam));
		printf("d");	
		MPI_Recv(&q[(id-rank-1)/w].idx.mat[q[(id-rank-1)/w].idx.n], tam, MPI_INT, rank, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&q[(id-rank-1)/w].dis.mat[q[(id-rank-1)/w].dis.n], tam, MPI_FLOAT, rank, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("e");
		q[(id-rank-1)/w].dis.n += tam;
		q[(id-rank-1)/w].idx.n += tam;
		in_q[(id-rank-1)/w]++;
		printf("f");	
		if(in_q[in]==w*(last_search-last_assign)-1){
			printf("g");
			ktmp = min(q[in].idx.n, k);
			printf("h");
			my_k_min(q[in], ktmp, dis2, ids2);
			printf("i");
			dis = (float*)realloc(dis,sizeof(float)*(ttam+ktmp));
			ids = (int*)realloc(ids,sizeof(int)*(ttam+ktmp));
			printf("j");
			memcpy(&dis[0] + ttam, dis2, sizeof(float)*ktmp);
			memcpy(&ids[0] + ttam, ids2, sizeof(int)*ktmp);
			in++;
			ttam+=ktmp;
			printf("k");
		}
		i++;
	}
	printf("l");
	while(in<queryn){
		printf("m");
		ktmp = min(q[in].idx.n, k);
		printf("n");
		my_k_min(q[in], ktmp, dis2, ids2);
		printf("o");
		dis = (float*)realloc(dis,sizeof(float)*(ttam+ktmp));
		ids = (int*)realloc(ids,sizeof(int)*(ttam+ktmp));
		printf("p");
		memcpy(&dis[0] + ttam, dis2, sizeof(float)*ktmp);
		memcpy(&ids[0] + ttam, ids2, sizeof(int)*ktmp);
		in++;
		ttam+=ktmp;
		printf("q");
	}
	
	end=MPI_Wtime();
	printf("r");
	free(q);
	free(in_q);
	free(dis2);
	free(ids2);
	free(coaidx);
	free(dis);
	printf("s");
	ids = (int*)realloc(ids,sizeof(int)*k*queryn);	
	printf("t");
	MPI_Recv(&start, 1, MPI_DOUBLE, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Recv(&start2, 1, MPI_DOUBLE, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("u");
	FILE *fp;

    	fp = fopen(arquivo, "a");

	fprintf(fp,"w=%d, tasks=%d, tamanho da base=%d\n", w, last_aggregator+1, tam_base);
	fprintf(fp,"Tempo de busca antes do bcast: %g\n",end*1000-start*1000);
	fprintf(fp,"Tempo de busca apos o bcast: %g\n\n",end*1000-start2*1000);
	printf("v");
	fclose(fp);
	printf("x");
	//Recebe o ids_gnd para calcular as estatisticas da busca
	MPI_Recv(&ids_gnd, sizeof(matI), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ids_gnd.mat = (int*)malloc(sizeof(int)*ids_gnd.d*ids_gnd.n);
	MPI_Recv(&ids_gnd.mat[0], ids_gnd.d*ids_gnd.n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	printf("w");
	pq_test_compute_stats2(ids, ids_gnd,k);
	printf("y");
	free(ids_gnd.mat);
	free(ids);
}
