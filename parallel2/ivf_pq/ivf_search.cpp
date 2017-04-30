#include "ivf_search.h"

static int last_assign, last_search, last_aggregator;

void parallel_search (int nsq, int k, int comm_sz, int threads, int tam, MPI_Comm search_comm, char *dataset, int w){

	ivfpq_t ivfpq;
	mat residual;
	int *coaidx, my_rank, queryn;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	printf("\nSearch ");
	double start = MPI_Wtime();
	#pragma omp parallel num_threads(threads)
	{
		ivf_t *ivf;
		ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);
		for(int i=0; i<ivfpq.coarsek; i++){
			ivf[i].ids = (int*)malloc(sizeof(int));
			ivf[i].idstam = 0;
			ivf[i].codes.mat = (int*)malloc(sizeof(int));
			ivf[i].codes.n = 0;
			ivf[i].codes.d = nsq;
		}
		int my_omp_rank = omp_get_thread_num ();

		for(int i=my_omp_rank; i<=(tam/100000); i+=threads){
			if(tam%100000==0 && i==(tam/100000)) break;
			
			ivf_t *ivf2;
			int aux;
        	mat vbase;
			ivf2 = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);
			
			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);
			
			//Cria a lista invertida	
			
			ivfpq_assign(ivfpq, vbase, ivf2);
			
			
			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf2[j].idstam; l++){
					ivf2[j].ids[l]+=100000*i+tam*(my_rank-last_assign-1);
				}

				aux = ivf[j].idstam;
				
				ivf[j].idstam += ivf2[j].idstam;
				ivf[j].ids = (int*)realloc(ivf[j].ids,sizeof(int)*ivf[j].idstam);
				memcpy (ivf[j].ids+aux, ivf2[j].ids, sizeof(int)*ivf2[j].idstam);
				ivf[j].codes.n += ivf2[j].codes.n;
				ivf[j].codes.mat = (int*)realloc(ivf[j].codes.mat,sizeof(int)*ivf[j].codes.n*ivf[j].codes.d);
				memcpy (ivf[j].codes.mat+aux*ivf[i].codes.d, ivf2[j].codes.mat, sizeof(int)*ivf2[j].codes.n*ivf2[j].codes.d);

				free(ivf2[j].ids);
				free(ivf2[j].codes.mat);
			}
			
			free(vbase.mat);
           	free(ivf2);
		}
		#pragma omp barrier
		if(my_omp_rank==0){
			double end = MPI_Wtime();
			printf("\ntime%g\n", end*1000-start*1000);
			MPI_Barrier(search_comm);
			MPI_Bcast(&queryn, 1, MPI_INT, 0, search_comm);	
			MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);
    		residual.n=queryn;
			residual.mat = (float*)malloc(sizeof(float)*residual.n*residual.d);
			MPI_Bcast(&residual.mat[0], residual.d*residual.n, MPI_FLOAT, 0, search_comm);
			coaidx = (int*)malloc(sizeof(int)*residual.n);
			MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);
		}
		#pragma omp barrier
		double f1=0, f2=0, f3=0, f4=0;
		
		for(int i=0; i<queryn/w; i++){
			int ktmp, *ids;
			float *dis;
			dis_t qt;

			qt.idx.mat = (int*) malloc(sizeof(int));
			qt.dis.mat = (float*) malloc(sizeof(float));
			qt.idx.n=0;
			qt.idx.d=1;
			qt.dis.n=0;
			qt.dis.d=1;	
			double a1=MPI_Wtime();

			for(int j=0; j<w; j++){
				dis_t q;

				q = ivfpq_search(ivf, &residual.mat[0]+(i*w+j)*residual.d, ivfpq.pq, coaidx[i*w+j]);

				qt.idx.mat = (int*) realloc(qt.idx.mat, sizeof(int)*(qt.idx.n+q.idx.n));
				qt.dis.mat = (float*) realloc(qt.dis.mat, sizeof(float)*(qt.dis.n+q.dis.n));

				memcpy(&qt.idx.mat[qt.idx.n], &q.idx.mat[0], sizeof(int)*q.idx.n);
				memcpy(&qt.dis.mat[qt.dis.n], &q.dis.mat[0], sizeof(float)*q.dis.n);

				qt.idx.n+=q.idx.n;
				qt.dis.n+=q.dis.n;

				free(q.dis.mat);
				free(q.idx.mat);					
			}

			double a2=MPI_Wtime();
			ktmp = min(qt.idx.n, k);

			ids = (int*) malloc(sizeof(int)*ktmp);
			dis = (float*) malloc(sizeof(float)*ktmp);
			double a3=MPI_Wtime();
			my_k_min(qt, ktmp, &dis[0], &ids[0]);

			int id = my_rank*(residual.n/w)+i+1;

			double a4=MPI_Wtime();

			#pragma omp critical
			{
				MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 100000, MPI_COMM_WORLD);
				MPI_Send(&id, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
				MPI_Send(&ktmp, 1, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
				MPI_Send(&ids[0], ktmp, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
				MPI_Send(&dis[0], ktmp, MPI_FLOAT, last_aggregator, id, MPI_COMM_WORLD);
			}

			double a5=MPI_Wtime();
			f1 += a2*1000-a1*1000;
			f2 += a3*1000-a2*1000;
			f3 += a4*1000-a3*1000;
			f4 += a5*1000-a4*1000;

			free(ids);
			free(dis);
			free(qt.dis.mat);
			free(qt.idx.mat);
		}
		printf("a%gb%gc%gd%g\n",f1,f2,f3,f4);
		free(ivf);
	}

	printf(".");	

	free(residual.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
}

dis_t ivfpq_search(ivf_t *ivf, float *residual, pqtipo pq, int centroid_idx){
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
		compute_cross_distances(ds, 1, distab.d, &residual[query*ds], &pq.centroids[query*ks*ds], distab_temp);
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
