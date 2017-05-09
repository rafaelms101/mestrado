#include "ivf_aggregator.h"

void parallel_aggregator(int k, int w, int my_rank, int comm_sz, int tam_base, int threads){
	static int last_assign, last_search, last_aggregator;
	dis_t *q;
	matI ids_gnd;
	float  *dis;
	int *coaidx, *ids, rank, queryn, tam, l=0, ktmp, *in_q;
	double start=0, end;
	char arquivo[15] = "testes.txt";

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
	int i=0, finish=0;
	
	#pragma omp parallel num_threads(2)
	{
		int omp_rank = omp_get_thread_num();
		
		if(omp_rank==0){//Recebe os resultados do vetor de busca
			int num;
			while(finish==0){
				float *dis2;
				int *ids2;
				int ttam=0;
				
				MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE , 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&num, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				query_id_t *element = (query_id_t*)malloc(sizeof(query_id_t)*num);
				MPI_Recv(&element[0], sizeof(query_id_t)*num, MPI_BYTE, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for(int j=0; j<num; j++){
					ttam+=element[j].tam;
					q[element[j].id].dis.mat = (float*)realloc(q[element[j].id].dis.mat,sizeof(float)*(q[element[j].id].dis.n+element[j].tam));
					q[element[j].id].idx.mat = (int*)realloc(q[element[j].id].idx.mat,sizeof(int)*(q[element[j].id].idx.n+element[j].tam));
				}

				dis2 = (float*)malloc(sizeof(float)*ttam);
				ids2 = (int*)malloc(sizeof(int)*ttam);

				MPI_Recv(&ids2[0], ttam, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&dis2[0], ttam, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&finish, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				ttam=0;
				for(int j=0; j<num; j++){
					memcpy(&q[element[j].id].idx.mat[0] + q[element[j].id].idx.n, &ids2[ttam], sizeof(int)*element[j].tam);
					memcpy(&q[element[j].id].dis.mat[0] + q[element[j].id].dis.n, &dis2[ttam], sizeof(float)*element[j].tam);
					q[element[j].id].dis.n += element[j].tam;
					q[element[j].id].idx.n += element[j].tam;
					ttam+=element[j].tam;
					in_q[element[j].id]++;
				}
				free(ids2);
				free(dis2);
				free(element);	
			}
			end=MPI_Wtime();
		}	
		else{ //Agrega os resultados
			float *dis2 = (float*)malloc(sizeof(float)*k);
			int *ids2 = (int*)malloc(sizeof(int)*k);
			int ttam=0, in=0;

			while(in<queryn){	
				if(in_q[in]==(last_search-last_assign)){
					
					ktmp = min(q[in].idx.n, k);
			
					my_k_min(q[in], ktmp, dis2, ids2);
			
					dis = (float*)realloc(dis,sizeof(float)*(ttam+ktmp));
					ids = (int*)realloc(ids,sizeof(int)*(ttam+ktmp));
			
					memcpy(&dis[0] + ttam, dis2, sizeof(float)*ktmp);
					memcpy(&ids[0] + ttam, ids2, sizeof(int)*ktmp);
					free(q[in].dis.mat);
					free(q[in].idx.mat);
					in++;
					ttam+=ktmp;
				}
			}
			free(dis2);
			free(ids2);
		}
	}
	
	free(q);
	free(in_q);
	free(coaidx);
	free(dis);
	
	ids = (int*)realloc(ids,sizeof(int)*k*queryn);	
	
	MPI_Recv(&start, 1, MPI_DOUBLE, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	FILE *fp;

    	fp = fopen(arquivo, "a");

	fprintf(fp,"w=%d, tasks=%d, tamanho da base=%d\n", w, last_aggregator+1, tam_base);
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
