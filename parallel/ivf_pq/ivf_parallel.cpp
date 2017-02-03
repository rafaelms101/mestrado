#define TRAINER 1
#define ASSIGN 2
#define SEARCH 3
#define AGGREGATOR 4
#define FINISH 5
#define THREAD 6

#include "ivf_parallel.h"

static int threads;
static int last_assign;
static int last_search;
static int last_aggregator;
pthread_mutex_t lock;

void set_last (int comm_sz, int num_threads){

	threads = num_threads;
	last_assign=1;
	last_search=comm_sz-2;
	last_aggregator=comm_sz-1;
}

void parallel_training (char *dataset, int coarsek, int nsq, int tam){
	data v;
	ivfpq_t ivfpq;
	ivf_t *ivf;
	int dest, my_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//Le os vetores
	v = pq_test_load_vectors(dataset, tam, my_rank, last_assign);

	if (my_rank==0){
		//Cria centroides a partir dos vetores de treinamento
		ivfpq = ivfpq_new(coarsek, nsq, v.train);

		free(v.train.mat);

		//Envia os centroides para os processos de recebimento da query e de indexação e busca
		for(int i=1; i<=last_search; i++){
			MPI_Send(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsd*ivfpq.coa_centroidsn, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
		}
	}
	else {
		MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
		MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
		MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	v.base = pq_test_load_base(dataset, tam, my_rank, last_assign);

	//Cria a lista invertida
	ivf = ivfpq_assign(ivfpq, v.base);

	free(v.base.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);

	if(my_rank!=0){
		//Envia trechos da lista invertida assinalados a cada processo de busca
		for(int i=0; i<ivfpq.coarsek; i++){
			MPI_Send(&ivf[i], sizeof(ivf_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivf[i].ids[0], ivf[i].idstam, MPI_INT, 0, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivf[i].codes.mat[0], ivf[i].codes.d*ivf[i].codes.n, MPI_INT, 0, TRAINER, MPI_COMM_WORLD);
		}
	}
	else{
		ivf_t ivf2;
		int idstam, codesn;
		
		//Envia trechos da lista invertida assinalados a cada processo de busca
		for(int i=0; i<ivfpq.coarsek; i++){
			idstam = ivf[i].idstam;
			codesn = ivf[i].codes.n;
			
			for(int j=1; j<last_assign; j++){
				MPI_Recv(&ivf2, sizeof(ivf_t), MPI_BYTE, j, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				ivf2.ids = (int*)malloc(sizeof(int)*ivf2.idstam);
				MPI_Recv(&ivf2.ids[0], ivf2.idstam, MPI_INT, j, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				ivf2.codes.mat = (int*)malloc(sizeof(int)*ivf2.codes.n*ivf2.codes.d);
				MPI_Recv(&ivf2.codes.mat[0], ivf2.codes.n*ivf2.codes.d, MPI_INT, j, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				
				idstam += ivf2.idstam;
				codesn += ivf2.codes.n;

				ivf[i].ids = (int*)realloc(ivf[i].ids,sizeof(int)*idstam);
				memcpy (ivf[i].ids+ivf[i].idstam, ivf2.ids, sizeof(int)*ivf2.idstam);
				ivf[i].idstam = idstam;
				ivf[i].codes.mat = (int*)realloc(ivf[i].codes.mat,sizeof(int)*codesn*ivf[i].codes.d);
				memcpy (ivf[i].codes.mat+ivf[i].codes.n*ivf[i].codes.d, ivf2.codes.mat, sizeof(int)*ivf2.codes.n*ivf2.codes.d);
				ivf[i].codes.n = codesn;

				free(ivf2.ids);
				free(ivf2.codes.mat);
			}
			dest = i%(last_search-last_assign)+last_assign+1;
			MPI_Send(&ivf[i], sizeof(ivf_t), MPI_BYTE, dest, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivf[i].ids[0], ivf[i].idstam, MPI_INT, dest, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&ivf[i].codes.mat[0], ivf[i].codes.d*ivf[i].codes.n, MPI_INT, dest, TRAINER, MPI_COMM_WORLD);
		}
		//Envia o ids_gnd para o agregador calcular as estatisticas da busca
		for(int i=last_search+1; i<=last_aggregator; i++){
			MPI_Send(&v.ids_gnd, sizeof(matI), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
			MPI_Send(&v.ids_gnd.mat[0], v.ids_gnd.d*v.ids_gnd.n, MPI_INT, i, TRAINER, MPI_COMM_WORLD);
		}
	}	
	free(v.ids_gnd.mat);
}

void parallel_assign (char *dataset, int w){
	mat vquery, residual;
	ivfpq_t ivfpq;
	int *coaidx, dest, rest,id;
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
	for(int i=last_assign+1; i<=last_search; i++){
		MPI_Ssend(&residual.d, 1, MPI_INT, i, ASSIGN, MPI_COMM_WORLD);
	}
	double start;

	start=MPI_Wtime();
	for(int i=0;i<vquery.n; i++){
		for(int j=0;j<w;j++){
			id=i*w+j;

			bsxfunMINUS(residual, vquery, ivfpq.coa_centroids, i, &coaidx[i*w+j], 1);
			rest= coaidx[id]%(last_search-last_assign);
			
			dest= rest+last_assign+1;

			MPI_Ssend(&coaidx[id], 1, MPI_INT, dest, THREAD, MPI_COMM_WORLD);
			MPI_Ssend(&id, 1, MPI_INT, dest, THREAD, MPI_COMM_WORLD);
			MPI_Ssend(&residual.mat[0], residual.d, MPI_FLOAT, dest, THREAD, MPI_COMM_WORLD);
		}
	}

	char finish='s';
	for(int i=last_assign+1; i<=last_search; i++){
		for(int j=0; j<threads; j++){
			MPI_Send(&finish, 1, MPI_CHAR, i, FINISH, MPI_COMM_WORLD);
		}
	}

	MPI_Send(&start, 1, MPI_DOUBLE, last_aggregator, ASSIGN, MPI_COMM_WORLD);

	free(coaidx);
	free(coadis);
	free(residual.mat);
	free(vquery.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
}

void parallel_search (int nsq, int my_rank, int k, char* arquivo){

	ivf_threads_t ivf_threads, *ivf_threads_v;
	pthread_t *thread_handles;
	int tam, thread;

	ivf_threads.k = k;

	//Recebe os centroides
	MPI_Recv(&ivf_threads.ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivf_threads.ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivf_threads.ivfpq.pq.centroidsn*ivf_threads.ivfpq.pq.centroidsd);
	MPI_Recv(&ivf_threads.ivfpq.pq.centroids[0], ivf_threads.ivfpq.pq.centroidsn*ivf_threads.ivfpq.pq.centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivf_threads.ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivf_threads.ivfpq.coa_centroidsd*ivf_threads.ivfpq.coa_centroidsn);
	MPI_Recv(&ivf_threads.ivfpq.coa_centroids[0], ivf_threads.ivfpq.coa_centroidsn*ivf_threads.ivfpq.coa_centroidsd, MPI_FLOAT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	//Recebe o trecho da lista invertida assinalada ao processo
	tam = ivf_threads.ivfpq.coarsek/(last_search-last_assign);
	if(my_rank<=ivf_threads.ivfpq.coarsek%(last_search-last_assign)+last_assign)tam++;
	ivf_threads.ivf = (ivf_t*)malloc(sizeof(ivf_t)*tam);
	for(int i=0;i<tam;i++){
		MPI_Recv(&ivf_threads.ivf[i], sizeof(ivf_t), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf_threads.ivf[i].ids = (int*)malloc(sizeof(int)*ivf_threads.ivf[i].idstam);
		MPI_Recv(&ivf_threads.ivf[i].ids[0], ivf_threads.ivf[i].idstam, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf_threads.ivf[i].codes.mat = (int*)malloc(sizeof(int)*ivf_threads.ivf[i].codes.n*ivf_threads.ivf[i].codes.d);
		MPI_Recv(&ivf_threads.ivf[i].codes.mat[0], ivf_threads.ivf[i].codes.n*ivf_threads.ivf[i].codes.d, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Recv(&ivf_threads.residuald, 1, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivf_threads_v = (ivf_threads_t*)malloc(sizeof(ivf_threads_t)*threads);
	thread_handles = (pthread_t*)malloc(sizeof(pthread_t)*threads);
	for (thread =0; thread<threads; thread++){
		ivf_threads_v[thread].thread = thread;
		ivf_threads_v[thread].k = k;
		ivf_threads_v[thread].ivf = ivf_threads.ivf;
		ivf_threads_v[thread].ivfpq = ivf_threads.ivfpq;
		ivf_threads_v[thread].residuald = ivf_threads.residuald;
		pthread_create(&thread_handles[thread], NULL, search_threads, (void*) &ivf_threads_v[thread]);
	}
	for(thread =0; thread<threads; thread++){
		pthread_join(thread_handles[thread], NULL);	
	}
	
	free(thread_handles);
	free(ivf_threads.ivfpq.pq.centroids);
	free(ivf_threads.ivfpq.coa_centroids);
	free(ivf_threads.ivf);
	free(ivf_threads_v);
}

void parallel_aggregator(int k, int w, int my_rank, char *arquivo){
	dis_t q;
	matI ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, rest, nextdis, nextidx, source, queryn, tam, n, l=0, *dest_count;
	double start=0, end;

	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);
	q.dis.mat = (float*)malloc(sizeof(float));
	q.idx.mat = (int*)malloc(sizeof(int));
	dest_count = (int*)calloc(last_search-last_assign,sizeof(int));

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
		q.idx.n = 0;
		q.idx.d = 1;
		q.dis.n = 0;
		q.dis.d = 1;
		//Recebe os resultados da busca nos vetores assinalados ao agregador
		for(int j=0; j<w; j++){
			rest = coaidx[i*w+j]%(last_search-last_assign);
			source=rest+last_assign+1;
			dest_count[rest] = (dest_count[rest]+1)%threads;
			
			MPI_Recv(&tam, 1, MPI_INT, source, THREAD+i*w+j+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			q.dis.n += tam;
			q.idx.n += tam;
			q.dis.mat = (float*)realloc(q.dis.mat, sizeof(float)*(q.dis.n));
			q.idx.mat = (int*)realloc(q.idx.mat, sizeof(int)*(q.idx.n));
			
			MPI_Recv(&q.idx.mat[0]+nextidx, tam, MPI_INT, source, THREAD+i*w+j+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			MPI_Recv(&q.dis.mat[0]+nextdis, tam, MPI_FLOAT, source, THREAD+i*w+j+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			nextdis+=tam;
			nextidx+=tam;
		}

		//Concatena os resultados e define os vetores mais proximos
		int ktmp = min(q.idx.n, k);
		
		my_k_min(q, ktmp, dis2, ids2);
	
		memcpy(dis + l*k, dis2, sizeof(float)*k);
		memcpy(ids + l*k, ids2, sizeof(float)*k);

		l++;
	}
	end=MPI_Wtime();

	free(dis2);
	free(ids2);
	free(q.dis.mat);
	free(q.idx.mat);
	free(coaidx);
	free(dis);

	MPI_Recv(&start, 1, MPI_DOUBLE, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	FILE *fp;

        fp = fopen(arquivo, "a");

	fprintf(fp,"Tempo de busca: %g\n",end*1000-start*1000);

	fclose(fp);

	//Recebe o ids_gnd para calcular as estatisticas da busca
	MPI_Recv(&ids_gnd, sizeof(matI), MPI_BYTE, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ids_gnd.mat = (int*)malloc(sizeof(int)*ids_gnd.d*ids_gnd.n);
	MPI_Recv(&ids_gnd.mat[0], ids_gnd.d*ids_gnd.n, MPI_INT, 0, TRAINER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	pq_test_compute_stats2(ids, ids_gnd,k);

	free(ids_gnd.mat);
}

void *search_threads(void *ivf_threads_recv){
	ivf_threads_t *ivf_threads;
	char finish;
	int coaidx, id, flag, flag2, centroid_idx, *ids, ktmp,my_rank;
	float *dis;
	dis_t q;
	mat residual;
	MPI_Status status, status2;
	MPI_Request request, request2;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	ivf_threads = (ivf_threads_t*)ivf_threads_recv;

	finish = 'n';

	residual.d = ivf_threads->residuald;
	residual.n=1;

	residual.mat=(float*)malloc(sizeof(float)*residual.d);

	MPI_Irecv(&finish, 1, MPI_CHAR, last_assign, FINISH, MPI_COMM_WORLD, &request2);
	while(1){
		pthread_mutex_lock(&lock);
		MPI_Irecv(&coaidx, 1, MPI_INT, last_assign, THREAD, MPI_COMM_WORLD, &request);
		do{
			
			MPI_Test(&request, &flag, &status);
			
			MPI_Test(&request2, &flag2, &status2);
			
		}while(flag !=1 && flag2 !=1);

		if (finish=='s' && flag==0){
			pthread_mutex_unlock(&lock);
			break;
		}
		MPI_Recv(&id, 1, MPI_INT, last_assign, THREAD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&residual.mat[0], residual.d, MPI_FLOAT, last_assign, THREAD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		pthread_mutex_unlock(&lock);
		centroid_idx = coaidx/(last_search-last_assign);
		

		//Faz a busca no vetor assinalado e envia o resultado ao agregador

		q=ivfpq_search(ivf_threads->ivf, residual, ivf_threads->ivfpq.pq, centroid_idx);
		
		ktmp = min(q.idx.n, ivf_threads->k);
		
		dis = (float*)malloc(sizeof(float)*ktmp);
		ids = (int*)malloc(sizeof(int)*ktmp);
		
		my_k_min(q, ktmp, dis, ids);
		
		MPI_Send(&ktmp, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+id+1, MPI_COMM_WORLD);
		
		MPI_Send(&ids[0], ktmp, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+id+1, MPI_COMM_WORLD);
		
		MPI_Send(&dis[0], ktmp, MPI_FLOAT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+id+1, MPI_COMM_WORLD);

		free(dis);
		free(ids);
	}
	free(residual.mat);
	pthread_exit(NULL);
}