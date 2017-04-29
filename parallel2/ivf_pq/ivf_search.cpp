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

	double start=MPI_Wtime();
	# pragma omp parallel num_threads(1) 
	{
		int my_omp_rank = omp_get_thread_num ();
		
		for(int i=0; i<=(tam/1000000); i++){
			
			if(tam%1000000==0 && i==(tam/1000000)) break;

			ivf_t *ivf2;
			int aux;
        		mat vbase;
			ivf2 = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);
			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);
	
			//Cria a lista invertida
				
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
	double end=MPI_Wtime();
	printf("\ntime%g\n", end*1000-start*1000);
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

	int **ids = (int**)malloc(sizeof(int *)*(residual.n/w));
 	float **dis = (float**)malloc(sizeof(float *)*(residual.n/w));	
	
	printf(".");
	
	double start2=MPI_Wtime();
	int *fila_tam = (int *)calloc(residual.n/w, sizeof(int));
	int *fila_id = (int *)malloc(sizeof(int)*residual.n/w);
	for(int i=0; i<residual.n/w; i++){
		fila_id[i]=-1;
	}

	int l=0;
	printf(".");
	# pragma omp parallel num_threads(threads)
	{
		int my_omp_rank = omp_get_thread_num ();
		
		#pragma omp single nowait
		{

			query_id_t element;
                        double b1;
                        for(int i=0; i<residual.n/w; i++){

                                while(fila_id[i]==-1);
                                element.tam = fila_tam[i];
				element.id = fila_id[i];
                                if(i==0) b1=MPI_Wtime();
                                

                                int id = my_rank*(residual.n/w)+element.id+1;

                                MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 100000, MPI_COMM_WORLD);
                                MPI_Send(&id, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
                                MPI_Send(&element.tam, 1, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
                                MPI_Send(&ids[element.id][0], element.tam, MPI_INT, last_aggregator, id, MPI_COMM_WORLD);
                                MPI_Send(&dis[element.id][0], element.tam, MPI_FLOAT, last_aggregator, id, MPI_COMM_WORLD);
                       }
                       double b2=MPI_Wtime();
                       printf("z%g", b2*1000-b1*1000);

		}

		double f1=0, f2=0, f3=0, f4=0;
			
		#pragma omp for	schedule(dynamic) 
			for(int i=0; i<queryn/w; i++){
				//printf("%d\n", my_omp_rank);
				query_id_t element;
				int ktmp;
				dis_t qt;
				dis_t q;	
			
				qt.idx.mat = (int*) malloc(sizeof(int));
        			qt.dis.mat = (float*) malloc(sizeof(float));
				qt.idx.n=0;
				qt.idx.d=1;
				qt.dis.n=0;
				qt.dis.d=1;	
				double a1=MPI_Wtime();
				for(int j=0; j<w; j++){
					
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
			
				ids[i] = (int*) malloc(sizeof(int)*ktmp);
				dis[i] = (float*) malloc(sizeof(float)*ktmp);
				double a3=MPI_Wtime();
				my_k_min(qt, ktmp, &dis[i][0], &ids[i][0]);
				
				element.tam=ktmp;
				element.id=i;

				double a4=MPI_Wtime();

				#pragma omp atomic
					fila_tam[l]+=element.tam;
				#pragma omp atomic
                                        fila_id[l]+=element.id+1;
				#pragma omp atomic
					l++;
				double a5=MPI_Wtime();
				free(qt.dis.mat);
                        	free(qt.idx.mat);
				//printf("a%gb%gc%gd%g\n", a2*1000-a1*1000,a3*1000-a2*1000, a4*1000-a3*1000, a5*1000-a4*1000);

				f1 += a2*1000-a1*1000;
				f2 += a3*1000-a2*1000;
				f3 += a4*1000-a3*1000;
				f4 += a5*1000-a4*1000;
				
			}
			printf("\nfa%gfb%gfc%gfd%g\n",  f1, f2, f3, f4);
		
	}
	
	double end2=MPI_Wtime();

	printf("\ntime%g\n", end2*1000-start2*1000);
	printf(".");	
	free(ids);
	free(dis);
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
