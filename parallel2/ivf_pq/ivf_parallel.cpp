#define TRAINER 1
#define ASSIGN 2
#define SEARCH 3
#define AGGREGATOR 4
#define FINISH 5
#define THREAD 25

#include "ivf_parallel.h"

static int threads;
static int last_assign;
static int last_search;
static int last_aggregator;
pthread_mutex_t lock, lock2, lock3;

void set_last (int comm_sz, int num_threads){

	threads = num_threads;
	last_assign=1;
	last_search=comm_sz-2;
	last_aggregator=comm_sz-1;
}

void parallel_training (char *dataset, int coarsek, int nsq, int tam){
	data v;
	ivfpq_t ivfpq;
	char file[15] = "file_ivfpq.bin";
	char file2[15] = "cent_ivfpq.bin";
	char file3[15] = "coa_ivfpq.bin";

	#ifdef TRAIN

		v = pq_test_load_vectors(dataset, tam, 0);

		ivfpq = ivfpq_new(coarsek, nsq, v.train);

		FILE *arq, *arq2, *arq3;

		arq = fopen(file, "wb");
		arq2 = fopen(file2, "wb");
		arq3 = fopen(file3, "wb");

		if (arq == NULL){
        	printf("Problemas na CRIACAO do arquivo\n");
   			return;
    	}

    	printf("%d\n", ivfpq.coa_centroidsn);
    	printf("%d\n", ivfpq.coa_centroidsd);

    	fwrite (&ivfpq.pq.nsq, sizeof(int), 1, arq);
    	fwrite (&ivfpq.pq.ks, sizeof(int), 1, arq);
    	fwrite (&ivfpq.pq.ds, sizeof(int), 1, arq);
    	fwrite (&ivfpq.pq.centroidsn, sizeof(int), 1, arq);
    	fwrite (&ivfpq.pq.centroidsd, sizeof(int), 1, arq);
    	fwrite (&ivfpq.coarsek, sizeof(int), 1, arq);
    	fwrite (&ivfpq.coa_centroidsn, sizeof(int), 1, arq);
    	fwrite (&ivfpq.coa_centroidsd, sizeof(int), 1, arq);
    	fwrite (&ivfpq.pq.centroids[0], sizeof(float), ivfpq.pq.centroidsn*ivfpq.pq.centroidsn, arq2);
    	fwrite (&ivfpq.coa_centroids[0], sizeof(float), ivfpq.coa_centroidsn*ivfpq.coa_centroidsn, arq3);

    	fclose(arq);
    	fclose(arq2);
    	fclose(arq3);

    	free(v.train.mat);
    	free(ivfpq.pq.centroids);
		free(ivfpq.coa_centroids);

	#else	

    	ivf_t *ivf;
    	int my_rank;

		MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

		v = pq_test_load_vectors(dataset, tam, my_rank);

		if (my_rank==0){

			#ifdef READ_TRAIN

				FILE *arq, *arq2, *arq3;

				arq = fopen(file, "rb");
				arq2 = fopen(file2, "rb");
				arq3 = fopen(file3, "rb");

				if (arq == NULL){
    				printf("Problemas na abertura do arquivo\n");
    				return;
				}

				fread (&ivfpq.pq.nsq, sizeof(int), 1, arq);
    			fread (&ivfpq.pq.ks, sizeof(int), 1, arq);
    			fread (&ivfpq.pq.ds, sizeof(int), 1, arq);
    			fread (&ivfpq.pq.centroidsn, sizeof(int), 1, arq);
    			fread (&ivfpq.pq.centroidsd, sizeof(int), 1, arq);
    			fread (&ivfpq.coarsek, sizeof(int), 1, arq);
    			fread (&ivfpq.coa_centroidsn, sizeof(int), 1, arq);
    			fread (&ivfpq.coa_centroidsd, sizeof(int), 1, arq);
    			ivfpq.pq.centroids = (float *) malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsn);
    			fread (&ivfpq.pq.centroids[0], sizeof(float), ivfpq.pq.centroidsn*ivfpq.pq.centroidsn, arq2);
    			ivfpq.coa_centroids = (float *) malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsn);
    			fread (&ivfpq.coa_centroids[0], sizeof(float), ivfpq.coa_centroidsn*ivfpq.coa_centroidsn, arq3);

    			fclose(arq);
    			fclose(arq2);
    			fclose(arq3);

			#else

				//Cria centroides a partir dos vetores de treinamento
				ivfpq = ivfpq_new(coarsek, nsq, v.train);

			#endif
			
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
	
		ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);
	
		for(int i=0; i<coarsek; i++){
			ivf[i].ids = (int*)malloc(sizeof(int));
			ivf[i].idstam = 0;
			ivf[i].codes.mat = (int*)malloc(sizeof(int));
			ivf[i].codes.n = 0;
			ivf[i].codes.d = nsq;
		}
	

		for(int i=0; i<=(tam/1000000); i++){
			ivf_t *ivf3;
			int aux;
		
		
			ivf3 = (ivf_t *)malloc(sizeof(ivf_t)*coarsek);
			if(tam%1000000==0 && i==(tam/1000000))break;
			v.base = pq_test_load_base(dataset, my_rank, last_assign, i);
			
			//Cria a lista invertida

			ivfpq_assign(ivfpq, v.base, ivf3);
		
			for(int j=0; j<coarsek; j++){
				for(int l=0; l<ivf3[j].idstam; l++){
					ivf3[j].ids[l]+=1000000*i;
				}
				aux = ivf[j].idstam;
				ivf[j].idstam += ivf3[j].idstam;
				ivf[j].ids = (int*)realloc(ivf[j].ids,sizeof(int)*ivf[j].idstam);
				memcpy (ivf[j].ids+aux, ivf3[j].ids, sizeof(int)*ivf3[j].idstam);
				ivf[j].codes.n += ivf3[j].codes.n;
				ivf[j].codes.mat = (int*)realloc(ivf[j].codes.mat,sizeof(int)*ivf[j].codes.n*ivf[j].codes.d);
				memcpy (ivf[j].codes.mat+aux*ivf[i].codes.d, ivf3[j].codes.mat, sizeof(int)*ivf3[j].codes.n*ivf3[j].codes.d);
				free(ivf3[j].ids);
				free(ivf3[j].codes.mat);
			}
		
			free(v.base.mat);
			free(ivf3);
		}	
	
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
		
			//Envia trechos da lista invertida assinalados a cada processo de busca
			for(int i=0; i<ivfpq.coarsek; i++){
				int idstam = ivf[i].idstam;
				int codesn = ivf[i].codes.n;
			
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
				for(int j=last_assign+1; j<=last_search; j++){
					MPI_Send(&ivf[i], sizeof(ivf_t), MPI_BYTE, j, TRAINER, MPI_COMM_WORLD);
					MPI_Send(&ivf[i].ids[0], ivf[i].idstam, MPI_INT, j, TRAINER, MPI_COMM_WORLD);
					MPI_Send(&ivf[i].codes.mat[0], ivf[i].codes.d*ivf[i].codes.n, MPI_INT, j, TRAINER, MPI_COMM_WORLD);
				}
				free(ivf[i].ids);
				free(ivf[i].codes.mat);
			}
			//Envia o ids_gnd para o agregador calcular as estatisticas da busca
			for(int i=last_search+1; i<=last_aggregator; i++){
				MPI_Send(&v.ids_gnd, sizeof(matI), MPI_BYTE, i, TRAINER, MPI_COMM_WORLD);
				MPI_Send(&v.ids_gnd.mat[0], v.ids_gnd.d*v.ids_gnd.n, MPI_INT, i, TRAINER, MPI_COMM_WORLD);
			}
		}	
		free(ivf);
	
	#endif

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

			MPI_Ssend(&coaidx[id], 1, MPI_INT, dest, ASSIGN, MPI_COMM_WORLD);
			MPI_Ssend(&id, 1, MPI_INT, dest, ASSIGN, MPI_COMM_WORLD);
			MPI_Ssend(&residual.mat[0], residual.d, MPI_FLOAT, dest, ASSIGN, MPI_COMM_WORLD);
		}
	}
	

	char finish='s';
	for(int i=last_assign+1; i<=last_search; i++){
		for(int j=0; j<threads; j++){
			MPI_Send(&finish, 1, MPI_CHAR, i, FINISH+j, MPI_COMM_WORLD);
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
	ivf_threads.ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivf_threads.ivfpq.coarsek);
	for(int i=0;i<ivf_threads.ivfpq.coarsek;i++){
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
	dis_t *q;
	matI ids_gnd;
	float *dis2, *dis;
	int *coaidx, *ids2, *ids, rest, source, queryn, tam, n, l=0, id, *part, ktmp;
	double start=0, end;

	dis2 = (float*)malloc(sizeof(float)*k);
	ids2 = (int*)malloc(sizeof(int)*k);

	//Recebe o vetor contendo os indices da lista invertida
	MPI_Recv(&queryn, 1, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	coaidx = (int*)malloc(sizeof(int)*queryn*w);
	MPI_Recv(&coaidx[0], queryn*w, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
			MPI_Recv(&id, 1, MPI_INT, MPI_ANY_SOURCE, THREAD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			rest = coaidx[id]%(last_search-last_assign);
			source=rest+last_assign+1;
			
			MPI_Recv(&tam, 1, MPI_INT, source, THREAD+1+id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			MPI_Recv(&q[id/w].idx.mat[q[id/w].idx.n], tam, MPI_INT, source, THREAD+1+id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			MPI_Recv(&q[id/w].dis.mat[q[id/w].dis.n], tam, MPI_FLOAT, source, THREAD+1+id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
			q[id/w].dis.n += tam;
			q[id/w].idx.n += tam;
			part[id/w]++;

			if (part[l]==w){
				ktmp = min(q[l].idx.n, k);
	
				my_k_min(q[l], ktmp, dis2, ids2);
	
				memcpy(dis + l*k, dis2, sizeof(float)*k);
				memcpy(ids + l*k, ids2, sizeof(float)*k);

				/*for(int s=0; s<q[l].dis.n; s++){
					printf("%g ", q[l].dis.mat[s]);
				}
				printf("\n");*/

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
	free(ids);
	
}

void *search_threads(void *ivf_threads_recv){
	ivf_threads_t *ivf_threads;
	char finish;
	int coaidx, id, flag, flag2, ktmp,my_rank;
	dis_t q;
	mat residual;
	MPI_Status status, status2;
	MPI_Request request, request2;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	ivf_threads = (ivf_threads_t*)ivf_threads_recv;

	int ids[ivf_threads->k];
	float dis[ivf_threads->k];

	finish = 'n';

	residual.d = ivf_threads->residuald;
	residual.n=1;
	
	residual.mat=(float*)malloc(sizeof(float)*residual.d);
	
	MPI_Irecv(&finish, 1, MPI_CHAR, last_assign, FINISH+ivf_threads->thread, MPI_COMM_WORLD, &request2);
	while(1){
		
		pthread_mutex_lock(&lock);
		MPI_Irecv(&coaidx, 1, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, &request);
		do{
			
			MPI_Test(&request, &flag, &status);
			
			MPI_Test(&request2, &flag2, &status2);
			
		}while(flag !=1 && flag2 !=1);
		
		if (finish=='s' && flag==0){
			pthread_mutex_unlock(&lock);
			break;
		}
		
		MPI_Recv(&id, 1, MPI_INT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		MPI_Recv(&residual.mat[0], residual.d, MPI_FLOAT, last_assign, ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		pthread_mutex_unlock(&lock);		

		//Faz a busca no vetor assinalado e envia o resultado ao agregador

		q=ivfpq_search(ivf_threads->ivf, residual, ivf_threads->ivfpq.pq, coaidx);
		
		ktmp = min(q.idx.n, ivf_threads->k);
		
		my_k_min(q, ktmp, &dis[0], &ids[0]);
		
		free(q.dis.mat);
		free(q.idx.mat);
		
		pthread_mutex_lock(&lock);
		MPI_Send(&id, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD, MPI_COMM_WORLD);
		
		MPI_Send(&ktmp, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+1+id, MPI_COMM_WORLD);
		
		MPI_Send(&ids[0], ktmp, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+1+id, MPI_COMM_WORLD);
		
		MPI_Send(&dis[0], ktmp, MPI_FLOAT, last_search+1+(coaidx%(last_aggregator-last_search)), THREAD+1+id, MPI_COMM_WORLD);
		
		pthread_mutex_unlock(&lock);
	}
	free(residual.mat);
	return NULL;
}
