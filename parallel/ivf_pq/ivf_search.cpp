#include "ivf_search.h"

static int last_assign, last_search, last_aggregator;
dis_t q_total;
pthread_mutex_t lock;

void parallel_search (int nsq, int my_rank, int k, int comm_sz, int threads){

	ivf_threads_t ivf_threads, *ivf_threads_v;
	pthread_t *thread_handles;
	mat residual;
	MPI_Status status, status2;
	MPI_Request request, request2;
	int tam, ids[k], coaidx, id, flag, flag2, ktmp;
	float dis[k];
	char finish;

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe os centroides
	MPI_Recv(&ivf_threads.ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivf_threads.ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivf_threads.ivfpq.pq.centroidsn*ivf_threads.ivfpq.pq.centroidsd);
	MPI_Recv(&ivf_threads.ivfpq.pq.centroids[0], ivf_threads.ivfpq.pq.centroidsn*ivf_threads.ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivf_threads.ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivf_threads.ivfpq.coa_centroidsd*ivf_threads.ivfpq.coa_centroidsn);
	MPI_Recv(&ivf_threads.ivfpq.coa_centroids[0], ivf_threads.ivfpq.coa_centroidsn*ivf_threads.ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	//Recebe o trecho da lista invertida assinalada ao processo
	tam = ivf_threads.ivfpq.coarsek/(last_search-last_assign);
	if(my_rank<=ivf_threads.ivfpq.coarsek%(last_search-last_assign)+last_assign)tam++;
	ivf_threads.ivf = (ivf_t*)malloc(sizeof(ivf_t)*tam);
	for(int i=0;i<tam;i++){
		MPI_Recv(&ivf_threads.ivf[i], sizeof(ivf_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf_threads.ivf[i].ids = (int*)malloc(sizeof(int)*ivf_threads.ivf[i].idstam);
		MPI_Recv(&ivf_threads.ivf[i].ids[0], ivf_threads.ivf[i].idstam, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		ivf_threads.ivf[i].codes.mat = (int*)malloc(sizeof(int)*ivf_threads.ivf[i].codes.n*ivf_threads.ivf[i].codes.d);
		MPI_Recv(&ivf_threads.ivf[i].codes.mat[0], ivf_threads.ivf[i].codes.n*ivf_threads.ivf[i].codes.d, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	finish = 'n';
	
	MPI_Recv(&residual.d, 1, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	residual.n=1;
	residual.mat=(float*)malloc(sizeof(float)*residual.d);

	MPI_Irecv(&finish, 1, MPI_CHAR, last_assign, FINISH, MPI_COMM_WORLD, &request2);
	while(1){

		MPI_Irecv(&coaidx, 1, MPI_INT, last_assign, 0, MPI_COMM_WORLD, &request);
		do{
			
			MPI_Test(&request, &flag, &status);
			
			MPI_Test(&request2, &flag2, &status2);
			
		}while(flag !=1 && flag2 !=1);
		
		if (finish=='s' && flag==0){
			break;
		}
		
		MPI_Recv(&id, 1, MPI_INT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		MPI_Recv(&residual.mat[0], residual.d, MPI_FLOAT, last_assign, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		int centroid_idx = coaidx/(last_search-last_assign);

		q_total.dis.mat = (float*)malloc(sizeof(float));
		q_total.dis.n = 0;
		q_total.dis.d = 1;
		q_total.idx.mat = (int*)malloc(sizeof(int));
		q_total.idx.n = 0;
		q_total.idx.d = 1;

		ivf_threads_v = (ivf_threads_t*)malloc(sizeof(ivf_threads_t)*threads);
		thread_handles = (pthread_t*)malloc(sizeof(pthread_t)*threads);
		for(int thread=0; thread<threads; thread++){
			ivf_threads_v[thread].thread = thread;
			ivf_threads_v[thread].threads = threads;
			ivf_threads_v[thread].ivf = &ivf_threads.ivf[centroid_idx];
			ivf_threads_v[thread].ivfpq = ivf_threads.ivfpq;
			ivf_threads_v[thread].residual = residual;

			pthread_create(&thread_handles[thread], NULL, search_threads, (void*) &ivf_threads_v[thread]);
		}
		for(int thread =0; thread<threads; thread++){
			pthread_join(thread_handles[thread], NULL);	
		}

		ktmp = min(q_total.idx.n, k);

		my_k_min(q_total, ktmp, &dis[0], &ids[0]);
		
		free(thread_handles);
		free(ivf_threads_v);
		free(q_total.dis.mat);
		free(q_total.idx.mat);

		MPI_Send(&id, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), SEARCH, MPI_COMM_WORLD);
		
		MPI_Send(&ktmp, 1, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), id, MPI_COMM_WORLD);
		
		MPI_Send(&ids[0], ktmp, MPI_INT, last_search+1+(coaidx%(last_aggregator-last_search)), id, MPI_COMM_WORLD);
		
		MPI_Send(&dis[0], ktmp, MPI_FLOAT, last_search+1+(coaidx%(last_aggregator-last_search)), id, MPI_COMM_WORLD);
	}
	free(ivf_threads.ivfpq.pq.centroids);
	free(ivf_threads.ivfpq.coa_centroids);
	free(ivf_threads.ivf);
	free(residual.mat);
}

void *search_threads(void *ivf_threads_recv){
	ivf_threads_t *ivf_threads;
	dis_t q;
	ivf_t ivf2;
	int tam;
	
	ivf_threads = (ivf_threads_t*)ivf_threads_recv;
	
	tam = ivf_threads->ivf->idstam/ivf_threads->threads;
	if(ivf_threads->thread == ivf_threads->threads-1){
		ivf2.idstam = ivf_threads->ivf->idstam - tam*(ivf_threads->threads-1);
		ivf2.codes.n = ivf_threads->ivf->idstam - tam*(ivf_threads->threads-1);
	}
	else{
		ivf2.idstam = tam;
		ivf2.codes.n = tam;
	}

	ivf2.codes.d = ivf_threads->ivf->codes.d;

	ivf2.codes.mat = ivf_threads->ivf->codes.mat + tam*ivf2.codes.d*ivf_threads->thread;
	ivf2.ids = ivf_threads->ivf->ids + tam*ivf_threads->thread;

	q=ivfpq_search(&ivf2, ivf_threads->residual, ivf_threads->ivfpq.pq, 0);
	pthread_mutex_lock(&lock);
	q_total.dis.mat = (float*)realloc(q_total.dis.mat,sizeof(float)*(q_total.dis.n+q.dis.n));
	q_total.idx.mat = (int*)realloc(q_total.idx.mat,sizeof(int)*(q_total.idx.n+q.idx.n));
	memcpy(q_total.dis.mat + q_total.dis.n, q.dis.mat, sizeof(float)*q.dis.n);
	q_total.dis.n += q.dis.n;
	memcpy(q_total.idx.mat + q_total.idx.n, q.idx.mat, sizeof(int)*q.idx.n);
	q_total.idx.n += q.idx.n;
	pthread_mutex_unlock(&lock);
		
	return NULL;
}

dis_t ivfpq_search(ivf_t *ivf, mat residual, pqtipo pq, int centroid_idx){
	dis_t q;
	int ds, ks, nsq;

	ds = pq.ds;
	ks = pq.ks;
	nsq = pq.nsq;

	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;
	
	float *distab_temp=(float*)malloc(sizeof(float)*ks);
	
	float* AUXSUMIDX;
	
	q.dis.n = ivf[centroid_idx].codes.n;
	q.dis.d = 1;
	q.dis.mat = (float*)malloc(sizeof(float)*q.dis.n);

	q.idx.n = ivf[centroid_idx].codes.n;
	q.idx.d = 1;
	q.idx.mat = (int*)malloc(sizeof(int)*q.idx.n);
	
	for (int query = 0; query < nsq; query++) {
		compute_cross_distances(ds, 1, distab.d, &residual.mat[query*ds], &pq.centroids[query*ks*ds], distab_temp);
		memcpy(distab.mat+query*ks, distab_temp, sizeof(float)*ks);
	}
	
	AUXSUMIDX = sumidxtab2(distab, ivf[centroid_idx].codes, 0);

	memcpy(q.idx.mat, ivf[centroid_idx].ids,  sizeof(int)*ivf[centroid_idx].idstam);
	memcpy(q.dis.mat, AUXSUMIDX, sizeof(float)*ivf[centroid_idx].codes.n);
	
	free (AUXSUMIDX);
	free (distab_temp);
	free (distab.mat);
	return q;
}

int min(int a, int b){
	if(a>b){
		return b;
	}
	else
		return a;
}

float * sumidxtab2(mat D, matI ids, int offset){
	//aloca o vetor a ser retornado
	float *dis = (float*)malloc(sizeof(float)*ids.n);
	int i, j;

	//soma as distancias para cada vetor
	for (i = 0; i < ids.n ; i++) {
		float dis_tmp = 0;
		for(j=0; j<D.n; j++){
			dis_tmp += D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d];
		}
		dis[i]=dis_tmp;
	}

	return dis;
}

int* imat_new_transp (const int *a, int ncol, int nrow){
 	int i,j;
	int *vt=(int*)malloc(sizeof(int)*ncol*nrow);

 	for(i=0;i<ncol;i++)
    	for(j=0;j<nrow;j++)
   			vt[i*nrow+j]=a[j*ncol+i];

  	return vt;
}