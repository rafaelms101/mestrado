#define TRAINER 1
#define ASSIGN 2
#define SEARCH 3
#define AGGREGATOR 4
#define FINISH 5

#include "ivf_parallel.h"

void parallel_training (char *dataset, int coarsek, int nsq, int last_search, int last_aggregator, int last_assign){
	data v;
	ivfpq_t ivfpq;
	ivf_t *ivf;
	int dest;

	//Le os vetores
	v = pq_test_load_vectors(dataset);

	//Envia o ids_gnd para o agregador calcular as estatisticas da busca
	for(int i=last_search+1; i<=last_aggregator; i++){
		MPI_Send(&v.ids_gnd, sizeof(matI), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&v.ids_gnd.mat[0], v.ids_gnd.d*v.ids_gnd.n, MPI_INT, i, TRAINER, MPI_COMM_WORLD);
	}

	//Cria centroides a partir dos vetores de treinamento
	ivfpq = ivfpq_new(coarsek, nsq, v.train);

	//Envia os centroides para os processos de recebimento da query e de indexação e busca
	for(int i=1; i<=last_search; i++){
		MPI_Send(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsd*ivfpq.coa_centroidsn, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
	}

	//Cria a lista invertida
	ivf = ivfpq_assign(ivfpq, v.base);

	//Envia trechos da lista invertida assinalados a cada processo de busca
	for(int i=0; i<ivfpq.coarsek; i++){
		dest = i%(last_search-last_assign)+last_assign+1;
		MPI_Send(&ivf[i], sizeof(ivf_t), MPI_BYTE, dest, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivf[i].ids[0], ivf[i].idstam, MPI_INT, dest, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivf[i].codes.mat[0], ivf[i].codes.d*ivf[i].codes.n, MPI_INT, dest, TRAINER, MPI_COMM_WORLD);
	}
}

void parallel_assign (char *dataset, int last_search, int last_assign, int w, int last_aggregator){
	mat vquery, residual;
	ivfpq_t ivfpq;
	int *coaidx, dest;
	float *coadis;

	vquery = pq_test_load_query(dataset);

	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Calcula e envia o residuo de cada vetor da query para os processos de busca
	residual.mat = (float*)malloc(sizeof(float)*vquery.d);
	residual.d = vquery.d;
	residual.n = 1;

	coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	coadis = (float*)malloc(sizeof(float)*vquery.n*w);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);
	for(int i=last_search+1; i<=last_aggregator; i++){
		MPI_Send(&vquery.n, 1, MPI_INT, i, ASSIGN, MPI_COMM_WORLD);
		MPI_Send(&coaidx[0], vquery.n*w, MPI_INT, i, ASSIGN, MPI_COMM_WORLD);
	}

	for(int i=0;i<vquery.n; i++){
		for(int j=0;j<w;j++){
			
			bsxfunMINUS(residual, vquery, ivfpq.coa_centroids, i, &coaidx[i*w+j], 1);
			dest=coaidx[i*w+j]%(last_search-last_assign)+last_assign+1;
			MPI_Send(&coaidx[i*w+j], 1, MPI_INT, dest, ASSIGN, MPI_COMM_WORLD);
			MPI_Send(&residual.d, 1, MPI_INT, dest, ASSIGN, MPI_COMM_WORLD);
			MPI_Send(&residual.mat[0], residual.n*residual.d, MPI_FLOAT, dest, ASSIGN, MPI_COMM_WORLD);
		}
	}
	char finish='s';
	for(int i=last_assign+1; i<=last_search; i++){
		MPI_Send(&finish, 1, MPI_CHAR, i, FINISH, MPI_COMM_WORLD);
	}
}

void parallel_search (int nsq, int last_search, int my_rank, int last_aggregator, int k, int last_assign){

	ivfpq_t ivfpq;
	ivf_t *ivf;
	int tam, flag, flag2, l=2, coaidx, centroid_idx, rank_source, *ids2;
	float *dis2;
	char finish;
	MPI_Status status, status2;
	MPI_Request request, request2;
	mat residual;
	dis_t q;	
		
	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Recebe o trecho da lista invertida assinalada ao processo
	tam = ivfpq.coarsek/(last_search-last_assign);
	if(my_rank<=ivfpq.coarsek%(last_search-last_assign)+last_assign)tam++;
	ivf = (ivf_t*)malloc(sizeof(ivf_t)*tam);
	for(int i=0;i<tam;i++){
		MPI_Recv(&ivf[i], sizeof(ivf_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf[i].ids = (int*)malloc(sizeof(int)*ivf[i].idstam);
		MPI_Recv(&ivf[i].ids[0], ivf[i].idstam, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf[i].codes.mat = (int*)malloc(sizeof(int)*ivf[i].codes.n*ivf[i].codes.d);
		MPI_Recv(&ivf[i].codes.mat[0], ivf[i].codes.n*ivf[i].codes.d, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	//Recebe o resíduo de um vetor da query
	finish = 'n';
	residual.n=1;

	MPI_Irecv(&finish, 1, MPI_CHAR, MPI_ANY_SOURCE, FINISH, MPI_COMM_WORLD, &request2);
	while(1){	
		MPI_Irecv(&coaidx, 1, MPI_INT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, &request);
		do{
			MPI_Test(&request, &flag, &status);
			MPI_Test(&request2, &flag2, &status2);
		}while(flag !=1 && flag2 !=1);
		if (finish=='s' && flag==0)break;	
		MPI_Recv(&residual.d, 1, MPI_INT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		residual.mat=(float*)malloc(sizeof(float)*residual.d);
		MPI_Recv(&residual.mat[0], residual.d, MPI_FLOAT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		centroid_idx = coaidx/(last_search-last_assign);
		//Faz a busca no vetor assinalado e envia o resultado ao agregador
		q=ivfpq_search(ivf, residual, ivfpq.pq, centroid_idx);

		MPI_Send(&q.idx.n, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
		MPI_Send(&q.idx.mat[0], q.idx.d*q.idx.n, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
		MPI_Send(&q.dis.mat[0], q.dis.d*q.dis.n, MPI_FLOAT, last_search+1+(coaidx%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
	}
}

void parallel_aggregator(int k, int w, int my_rank, int last_aggregator, int last_search, int last_assign){
	mat qdis;
	matI qidx, ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, nextdis, nextidx, source, queryn, tam, n, l=0;


	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);
	qdis.mat = (float*)malloc(sizeof(float)*1);
	qidx.mat = (int*)malloc(sizeof(int)*1);

	//Recebe o ids_gnd para calcular as estatisticas da busca
	MPI_Recv(&ids_gnd, sizeof(matI), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ids_gnd.mat = (int*)malloc(sizeof(int)*ids_gnd.d*ids_gnd.n);
	MPI_Recv(&ids_gnd.mat[0], ids_gnd.d*ids_gnd.n, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Recebe o vetor contendo os indices da lista invertida
	MPI_Recv(&queryn, 1, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	coaidx = (int*)malloc(sizeof(int)*queryn*w);
	MPI_Recv(&coaidx[0], queryn*w, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


	n=queryn/(last_aggregator-last_search);
	if((my_rank-last_search) <= queryn%(last_aggregator-last_search)) n++;

	dis = (float*)malloc(sizeof(float)*k*n);
	ids = (int*)malloc(sizeof(int)*k*n);

	for(int i=(my_rank-last_search-1); i<queryn; i+=(last_aggregator-last_search)){
		nextidx=0;
		nextdis=0;
		qidx.n = 0;
		qidx.d = 1;
		qdis.n = 0;
		qdis.d = 1;
		//Recebe os resultados da busca nos vetores assinalados ao agregador
		for(int j=0; j<w; j++){
			source=coaidx[i*w+j]%(last_search-last_assign)+last_assign+1;
			MPI_Recv(&tam, 1, MPI_INT, source, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			qdis.n += tam;
			qidx.n += tam;
			qdis.mat = (float*)realloc(qdis.mat, sizeof(float)*(qdis.n));
			qidx.mat = (int*)realloc(qidx.mat, sizeof(int)*(qidx.n));
			MPI_Recv(&qidx.mat[0]+nextidx, tam, MPI_INT, source, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&qdis.mat[0]+nextdis, tam, MPI_FLOAT, source, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			nextdis+=tam;
			nextidx+=tam;
		}

		//for(int h=0;h<qidx.n;h++){
		//	printf("%d ", qidx.mat[h]);
		//}
		//printf("\n\n");

		//Concatena os resultados e define os vetores mais proximos
		int ktmp = min(qidx.n, k);
		k_min(qdis, ktmp, dis2, ids2);
		
		memcpy(dis + l*k, dis2, sizeof(float)*k);
		for(int b = 0; b < k ; b++){
			ids[l*k + b] = qidx.mat[ids2[b]-1];
		}
		l++;
	}
	/*for (int i=0; i<k*queryn; i++){
		printf("%g ", dis[i]);
	}*/
	//int *ids_tr = imat_new_transp (ids, queryn, k);
	//calcula as estatísticas
	pq_test_compute_stats2(ids, ids_gnd,k);
}