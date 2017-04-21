#include "ivf_search.h"

static int last_assign, last_search, last_aggregator;

void parallel_search (int nsq, int k, int comm_sz, int threads, int tam, MPI_Comm search_comm, char *dataset, int w){

	ivfpq_t ivfpq;
	ivf_t *ivf;
	mat residual;
	int *coaidx, my_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	printf("\nSearch ");
	
	ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

	for(int i=0; i<ivfpq.coarsek; i++){
		ivf[i].ids = (int*)malloc(sizeof(int));
		ivf[i].idstam = 0;
		ivf[i].codes.mat = (int*)malloc(sizeof(int));
		ivf[i].codes.n = 0;
		ivf[i].codes.d = nsq;
	}
	printf(".");
	# pragma omp parallel for num_threads(threads) schedule(dynamic)
	for(int i=0; i<=(tam/1000000); i++){
		
		if(tam%1000000==0 && i==(tam/1000000));
		else{
			ivf_t *ivf2;
                	int aux;
                	mat vbase;
			ivf2 = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);
			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);
	
			//Cria a lista invertida
			#pragma omp critical		
			ivfpq_assign(ivfpq, vbase, ivf2);
			
			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf2[j].idstam; l++){
					ivf2[j].ids[l]+=1000000*i+tam*(my_rank-last_assign-1);
				}
				aux = ivf[j].idstam;
				
				#pragma omp critical
				{
					ivf[j].idstam += ivf2[j].idstam;
					ivf[j].ids = (int*)realloc(ivf[j].ids,sizeof(int)*ivf[j].idstam);
					memcpy (ivf[j].ids+aux, ivf2[j].ids, sizeof(int)*ivf2[j].idstam);
					ivf[j].codes.n += ivf2[j].codes.n;
					ivf[j].codes.mat = (int*)realloc(ivf[j].codes.mat,sizeof(int)*ivf[j].codes.n*ivf[j].codes.d);
					memcpy (ivf[j].codes.mat+aux*ivf[i].codes.d, ivf2[j].codes.mat, sizeof(int)*ivf2[j].codes.n*ivf2[j].codes.d);
				}
				free(ivf2[j].ids);
				free(ivf2[j].codes.mat);
			}
		free(vbase.mat);
                free(ivf2);
		}
		
	}
	
	printf(".");

	int queryn;
	
	MPI_Barrier(search_comm);
	MPI_Bcast(&queryn, 1, MPI_INT, 0, search_comm);		
	MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);
    residual.n=queryn/1;
	residual.mat = (float*)malloc(sizeof(float)*residual.n*residual.d);
		
	MPI_Bcast(&residual.mat[0], residual.d*residual.n, MPI_FLOAT, 0, search_comm);
	coaidx = (int*)malloc(sizeof(int)*residual.n);
	MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);
	
	//std::list<query_id_t> fila;
	printf(".");
	
	double start=MPI_Wtime();
	# pragma omp parallel for num_threads(threads) schedule(dynamic)	
		for(int i=0; i<queryn/w; i++){
			
			int omp_rank2 = omp_get_thread_num();
			printf("a%d ", omp_rank2);
			
			float *dis;
			int ktmp, *ids;
			dis_t qt;
			dis_t q;	
			double a1 = MPI_Wtime();
			qt.idx.mat = (int*) malloc(sizeof(int));
        		qt.dis.mat = (float*) malloc(sizeof(float));
			qt.idx.n=0;
			qt.idx.d=1;
			qt.dis.n=0;
			qt.dis.d=1;	
			double a2=MPI_Wtime();
			printf("b ");
			
			for(int j=0; j<w; j++){
				printf("c ");	
				q = ivfpq_search(ivf, &residual.mat[0]+(i*w+j)*residual.d, ivfpq.pq, coaidx[i*w+j]);
				printf("d ");	
				qt.idx.mat = (int*) realloc(qt.idx.mat, sizeof(int)*(qt.idx.n+q.idx.n));
				qt.dis.mat = (float*) realloc(qt.dis.mat, sizeof(float)*(qt.dis.n+q.dis.n));
				printf("e ");
				memcpy(&qt.idx.mat[qt.idx.n], &q.idx.mat[0], sizeof(int)*q.idx.n);
				memcpy(&qt.dis.mat[qt.dis.n], &q.dis.mat[0], sizeof(float)*q.dis.n);
				printf("f ");
				qt.idx.n+=q.idx.n;
				qt.dis.n+=q.dis.n;
				printf("g ");
				free(q.dis.mat);
				free(q.idx.mat);					
			}
			printf("h ");
			double a3=MPI_Wtime();
			ktmp = min(qt.idx.n, k);
			

			ids = (int*) malloc(sizeof(int)*ktmp);
			dis = (float*) malloc(sizeof(float)*ktmp);
			double a4=MPI_Wtime();
				
			my_k_min(qt, ktmp, &dis[0], &ids[0]);
			double a5=MPI_Wtime();
			
			int id = my_rank*(residual.n/w)+i+1;

			#pragma omp critical
			{
				MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 100000, MPI_COMM_WORLD);
                                MPI_Send(&id, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
                                MPI_Send(&ktmp, 1, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
                                MPI_Send(&ids[0], ktmp, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
                                MPI_Send(&dis[0], ktmp, MPI_FLOAT, last_aggregator, id, MPI_COMM_WORLD);
			}	
			
			double a6=MPI_Wtime();
			free(ids);
			free(dis);
			free(qt.dis.mat);
            		free(qt.idx.mat);
			//printf("\nthread%d d1%g, d2%g, d3%g, d4%g, d5%g", omp_rank2, a2-a1, a3-a2, a4-a3, a5-a4, a6-a5);				
		}
		double end=MPI_Wtime();
		printf("time%g", end*1000-start*1000);
			
	printf(".");	
	free(residual.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	free(ivf);
}

dis_t ivfpq_search(ivf_t *ivf, float *residual, pqtipo pq, int centroid_idx){
	dis_t q;
	int ds, ks, nsq;

	ds = pq.ds;
	ks = pq.ks;
	nsq = pq.nsq;
	printf("1 ");
	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;
	printf("2 ");
	float *distab_temp=(float*)malloc(sizeof(float)*ks);
	printf("3 ");
	float* AUXSUMIDX;
	
	q.dis.n = ivf[centroid_idx].codes.n;
	q.dis.d = 1;
	q.dis.mat = (float*)malloc(sizeof(float)*q.dis.n);
	printf("4 ");
	q.idx.n = ivf[centroid_idx].codes.n;
	q.idx.d = 1;
	q.idx.mat = (int*)malloc(sizeof(int)*q.idx.n);
	printf("5 ");
	for (int query = 0; query < nsq; query++) {
		compute_cross_distances(ds, 1, distab.d, &residual[query*ds], &pq.centroids[query*ks*ds], distab_temp);
		memcpy(distab.mat+query*ks, distab_temp, sizeof(float)*ks);
	}
	
	printf("6 ");

	AUXSUMIDX = sumidxtab2(distab, ivf[centroid_idx].codes, 0);
	printf("7 ");

	memcpy(q.idx.mat, ivf[centroid_idx].ids,  sizeof(int)*ivf[centroid_idx].idstam);
	memcpy(q.dis.mat, AUXSUMIDX, sizeof(float)*ivf[centroid_idx].codes.n);
	printf("8 ");
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
	printf("10 ");
	float *dis = (float*)malloc(sizeof(float)*ids.n);
	int i, j;
	printf("11 ");
	//soma as distancias para cada vetor

	for (i = 0; i < ids.n ; i++) {
		float dis_tmp = 0;
		for(j=0; j<D.n; j++){
			dis_tmp += D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d];
		}
		dis[i]=dis_tmp;
	}
	printf("12 ");

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
