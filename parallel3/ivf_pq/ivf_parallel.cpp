#define TRAINER 1
#define ASSIGN 2
#define SEARCH 3
#define AGGREGATOR 4
#define FINISH 5

#include "ivf_parallel.h"

typedef struct args_t{
	long thread;     //ok
	ivfpq_t ivfpq;   //ok
	ivf_t *ivf;     //ok
	int centroid_idx; //ok
	mat *residual;   //ok
	dis_t *q;      //ok
	float **dis;   //ok
	int **ids, *ktmp, i, *pontos;  //ok  // ok   //ok  //ok
} args_t;

void* threadsSearch(void * arguments);

void parallel_training (char *dataset, int coarsek, int nsq, int last_search, int last_aggregator, int last_assign, int tam){
	data v;
	ivfpq_t ivfpq;
	ivf_t *ivf;
	int dest;

	//Le os vetores
	v = pq_test_load_vectors(dataset, tam);

	//free(v.query.mat);

	//Cria centroides a partir dos vetores de treinamento
	ivfpq = ivfpq_new(coarsek, nsq, v.train);

	free(v.train.mat);

	//Envia os centroides para os processos de recebimento da query e de indexação e busca
	for(int i=1; i<=last_search; i++){
		MPI_Send(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
		MPI_Send(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsd*ivfpq.coa_centroidsn, MPI_FLOAT, i, TRAINER, MPI_COMM_WORLD);
	}

	//Cria a lista invertida
	ivf = ivfpq_assign(ivfpq, v.base);

	free(v.base.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);

	//Envia trechos da lista invertida assinalados a cada processo de busca
	for(int i=0; i<ivfpq.coarsek; i++){
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

	free(v.ids_gnd.mat);
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
	double start;

	char finish='s';
	for(int i=last_assign+1; i<=last_search; i++){
		MPI_Ssend(&finish, 1, MPI_CHAR, i, FINISH, MPI_COMM_WORLD);
	}
	start= MPI_Wtime();

	MPI_Send(&start, 1, MPI_DOUBLE, last_aggregator, ASSIGN, MPI_COMM_WORLD);

	free(coaidx);
	free(coadis);
	free(residual.mat);
	free(vquery.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
}

//Rodar varias queries ao memso tempo, no lugar de paralelizar para uma so query
void parallel_search (int nsq, int last_search, int my_rank, int last_aggregator, int k, int last_assign, char* arquivo, int threads){

	ivfpq_t ivfpq;
	ivf_t *ivf;
	int tam, *ktmp, flag, flag2, l=2, *coaidx=(int*)malloc(sizeof(int)), centroid_idx, rank_source, **ids, entrou = 0, pontos = 0;
	float **dis;
	char finish;
	MPI_Status status, status2;
	MPI_Request request, request2;
	mat *residual=(mat*)malloc(sizeof(mat));
	dis_t *q;
	FILE *fp;
	double start=0, end=0, time =0;
	args_t arguments;

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

	fp = fopen(arquivo, "a");

	MPI_Irecv(&finish, 1, MPI_CHAR, MPI_ANY_SOURCE, FINISH, MPI_COMM_WORLD, &request2);
	while(1){
		coaidx=(int*)realloc(coaidx,sizeof(int)*(entrou+1));
		residual=(mat*)realloc(residual,sizeof(mat)*(entrou+1));
		residual[entrou].n=1;

		MPI_Irecv(&coaidx[entrou], 1, MPI_INT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, &request);
		do{
			MPI_Test(&request, &flag, &status);
			MPI_Test(&request2, &flag2, &status2);
		}while(flag !=1 && flag2 !=1);
		if (finish=='s' && flag==0)break;

		MPI_Recv(&residual[entrou].d, 1, MPI_INT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		residual[entrou].mat=(float*)malloc(sizeof(float)*residual[entrou].d);
		MPI_Recv(&residual[entrou].mat[0], residual[entrou].d, MPI_FLOAT, MPI_ANY_SOURCE, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		entrou++;
	}
	start= MPI_Wtime();

	//Criacao das Threads ...
	int thread;
	pthread_t *thread_handles;
	thread_handles= (pthread_t*) malloc(threads*sizeof(pthread_t));

	q = (dis_t*)malloc(sizeof(dis_t)*entrou);
	dis = (float**)malloc(sizeof(float*)*entrou);
	ids = (int**)malloc(sizeof(int*)*entrou);
	ktmp = (int*)malloc(sizeof(int)*entrou);

	//associando os argumentos
	arguments.q = q;
	arguments.dis = dis;
	arguments.ids = ids;
	arguments.ivfpq = ivfpq;
	arguments.ivf = ivf;
	arguments.residual = residual;
	arguments.ktmp = ktmp;
	arguments.pontos = &pontos;

	for(int i=0; i<entrou; i+=threads){
		arguments.i = i;
		//Faz a busca no vetor assinalado e envia o resultado ao agregador
		for (thread= 0; thread < threads; thread++){
			if(i + thread < entrou){
				centroid_idx = coaidx[i + thread]/(last_search-last_assign);

				arguments.centroid_idx = centroid_idx;
				arguments.thread = thread;
				pthread_create(&thread_handles[thread] ,NULL, threadsSearch, (void*) &arguments);
			}
		}
		//free(residual[i].mat);
	}

	for (thread= 0; thread < threads; thread++){
		pthread_join(thread_handles[thread], NULL);
	}

	end= MPI_Wtime();
	time=end*1000-start*1000;

	free(residual);
	for(int i=0; i<entrou;i++){
		MPI_Send(&ktmp[i], 1, MPI_INT, last_search+1+(coaidx[i]%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
		MPI_Send(&ids[i][0], ktmp[i], MPI_INT, last_search+1+(coaidx[i]%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
		MPI_Send(&dis[i][0], ktmp[i], MPI_FLOAT, last_search+1+(coaidx[i]%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
	}
	fprintf(fp,"my_rank: %d, entrou: %d, pontos: %d time: %g\n",my_rank,entrou, pontos,time);
	fclose(fp);


	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	free(ivf);
}

void parallel_aggregator(int k, int w, int my_rank, int last_aggregator, int last_search, int last_assign, char *arquivo){
	mat qdis;
	matI qidx, ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, nextdis, nextidx, source, queryn, tam, n, l=0;
	double start=0, end;

	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);
	qdis.mat = (float*)malloc(sizeof(float));
	qidx.mat = (int*)malloc(sizeof(int));

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
			if(l==0 && j==0)end=MPI_Wtime();
			qdis.n += tam;
			qidx.n += tam;
			qdis.mat = (float*)realloc(qdis.mat, sizeof(float)*(qdis.n));
			qidx.mat = (int*)realloc(qidx.mat, sizeof(int)*(qidx.n));
			MPI_Recv(&qidx.mat[0]+nextidx, tam, MPI_INT, source, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&qdis.mat[0]+nextdis, tam, MPI_FLOAT, source, SEARCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			nextdis+=tam;
			nextidx+=tam;
		}

		//Concatena os resultados e define os vetores mais proximos
		int ktmp = min(qidx.n, k);
		k_min(qdis, ktmp, dis2, ids2);
		memcpy(dis + l*k, dis2, sizeof(float)*k);
		for(int b = 0; b < k ; b++){
			ids[l*k + b] = qidx.mat[ids2[b]-1];
		}
		l++;
	}

	free(dis2);
	free(ids2);
	free(qdis.mat);
	free(qidx.mat);
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

void *threadsSearch(void * arguments){

	args_t *arg = (args_t*) arguments;

	arg->q[i + thread]=ivfpq_search(arg->ivf, arg->residual[i + thread], arg->ivfpq.pq, arg->centroid_idx);
	arg->ktmp[i + thread] = min(arg->q[i + thread].idx.n, k);
	arg->dis[i + thread] = (float*)malloc(sizeof(float)*arg->ktmp[i + thread]);
	arg->ids[i + thread] = (int*)malloc(sizeof(int)*arg->ktmp[i + thread]);
	k_min(arg->q[i + thread].dis, arg->ktmp[i + thread], arg->dis[i + thread], arg->ids[i + thread]);
	for(int b = 0; b < arg->ktmp[arg->i + arg->thread] ; b++){
		ids[arg->i + arg->thread][b] = arg->q[arg->i + arg->thread].idx.mat[arg->ids[arg->i + arg->thread][b]-1];
	}
	arg->(*pontos) += arg->q[arg->i + arg->thread].idx.n;
}
