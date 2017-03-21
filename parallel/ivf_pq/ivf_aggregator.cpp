#include "ivf_aggregator.h"

void parallel_aggregator(int k, int w, int my_rank, int comm_sz){
	static int last_assign, last_search, last_aggregator;
	dis_t *q;
	matI ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, rest, source, queryn, tam, n, l=0, id, *part, ktmp;
	double start=0, end;
	char arquivo[15] = "testes.txt";

	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);
	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe o vetor contendo os indices da lista invertida
	MPI_Recv(&queryn, 1, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	coaidx = (int*)malloc(sizeof(int)*queryn*w);
	MPI_Recv(&coaidx[0], queryn*w, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	q = (dis_t*)malloc(sizeof(dis_t)*queryn);
	part = (int*)malloc(sizeof(int)*queryn);

	for(int i=0; i<queryn; i++){
		q[i].dis.mat = (float*)malloc(sizeof(float)*w*k);
		q[i].idx.mat = (int*)malloc(sizeof(int)*w*k);
		q[i].dis.n = 0;
		q[i].idx.n = 0;
		part[i] = 0;
	}
	n=queryn/(last_aggregator-last_search);
	if((my_rank-last_search) <= queryn%(last_aggregator-last_search)) n++;

	dis = (float*)malloc(sizeof(float)*k*n);
	ids = (int*)malloc(sizeof(int)*k*n);

	for(int i=(my_rank-last_search-1); i<queryn; i+=(last_aggregator-last_search)){
		//Recebe os resultados da busca nos vetores assinalados ao agregador
		for(int j=0; j<w; j++){
			MPI_Recv(&id, 1, MPI_INT, MPI_ANY_SOURCE, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			rest = coaidx[id]%(last_search-last_assign);
			source=rest+last_assign+1;
			
			MPI_Recv(&tam, 1, MPI_INT, source, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			MPI_Recv(&q[id/w].idx.mat[q[id/w].idx.n], tam, MPI_INT, source, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Recv(&q[id/w].dis.mat[q[id/w].dis.n], tam, MPI_FLOAT, source, id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			q[id/w].dis.n += tam;
			q[id/w].idx.n += tam;
			part[id/w]++;

			if (part[l]==w){
				ktmp = min(q[l].idx.n, k);
	
				my_k_min(q[l], ktmp, dis2, ids2);
	
				memcpy(dis + l*k, dis2, sizeof(float)*k);
				memcpy(ids + l*k, ids2, sizeof(float)*k);

				l++;
			}
		}
	}
	end=MPI_Wtime();
	while (l<queryn){
		ktmp = min(q[l].idx.n, k);
	
		my_k_min(q[l], ktmp, dis2, ids2);
	
		memcpy(dis + l*k, dis2, sizeof(float)*k);
		memcpy(ids + l*k, ids2, sizeof(float)*k);

		l++;
	}
	
	free(q);
	free(dis2);
	free(ids2);
	free(coaidx);
	free(dis);
	free(part);
	

	MPI_Recv(&start, 1, MPI_DOUBLE, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	FILE *fp;

    fp = fopen(arquivo, "a");

    fprintf(fp,"w=%d, tasks=%d\n", w, last_aggregator+1);
	fprintf(fp,"Tempo de busca: %g\n\n",end*1000-start*1000);

	fclose(fp);

	//Recebe o ids_gnd para calcular as estatisticas da busca
	MPI_Recv(&ids_gnd, sizeof(matI), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ids_gnd.mat = (int*)malloc(sizeof(int)*ids_gnd.d*ids_gnd.n);
	MPI_Recv(&ids_gnd.mat[0], ids_gnd.d*ids_gnd.n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	pq_test_compute_stats2(ids, ids_gnd,k);

	free(ids_gnd.mat);
	free(ids);
}
