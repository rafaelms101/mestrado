#include "ivf_training.h"

void parallel_training (char *dataset, int coarsek, int nsq, int tam, int comm_sz){
	data v;
	ivfpq_t ivfpq;
	char file[25];
	char file2[25];
	char file3[25];
	static int last_assign, last_search, last_aggregator;

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	strcpy (file,"bin/file_ivfpq.bin");
	strcpy (file2,"bin/cent_ivfpq.bin");
	strcpy (file3,"bin/coa_ivfpq.bin");

	#ifdef TRAIN

		v = pq_test_load_vectors(dataset, tam);

		ivfpq = ivfpq_new(coarsek, nsq, v.train);

		FILE *arq, *arq2, *arq3;

		arq = fopen(file, "wb");
		arq2 = fopen(file2, "wb");
		arq3 = fopen(file3, "wb");

		if (arq == NULL){
        	printf("Problemas na CRIACAO do arquivo\n");
   			return;
    	}

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

		v = pq_test_load_vectors(dataset, tam);

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
			MPI_Send(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsd*ivfpq.coa_centroidsn, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
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
			v.base = pq_test_load_base(dataset, last_assign, i);
			
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
	
		int *sendcounts, *displs;
			
		//Envia trechos da lista invertida assinalados a cada processo de busca
		for(int i=0; i<ivfpq.coarsek; i++){
			int idstam = ivf[i].idstam;
			int codesn = ivf[i].codes.n;

			int itr=0;
				
			sendcounts = (int*) calloc(last_search-last_assign, sizeof(int));
			displs = (int*) malloc(sizeof(int)*(last_search-last_assign));
				
			for(int j=0;j<ivf[i].idstam; j++){
				sendcounts[itr]++;
				
				itr++;
				if(itr>=last_search-last_assign){
					itr=0;
				}
			}
				
			displs[0]=0;
				
			for(int j=1; j<(last_search-last_assign); j++){
					
				displs[j]= displs[j-1]+sendcounts[j-1];
			}
				
			
			for(int j=last_assign+1; j<=last_search; j++){
					
				MPI_Send(&ivf[i].codes.d, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
				MPI_Send(&sendcounts[j-last_assign-1], 1, MPI_INT, j, 0, MPI_COMM_WORLD);
				MPI_Send(&ivf[i].ids[displs[j-last_assign-1]], sendcounts[j-last_assign-1], MPI_INT, j, 0, MPI_COMM_WORLD);
				MPI_Send(&ivf[i].codes.mat[displs[j-last_assign-1]*ivf[i].codes.d], sendcounts[j-last_assign-1]*ivf[i].codes.d, MPI_INT, j, 0, MPI_COMM_WORLD);
			}
			free(ivf[i].ids);
			free(ivf[i].codes.mat);
		}
			
		//Envia o ids_gnd para o agregador calcular as estatisticas da busca
		for(int i=last_search+1; i<=last_aggregator; i++){
			MPI_Send(&v.ids_gnd, sizeof(matI), MPI_BYTE, i, 0, MPI_COMM_WORLD);
			MPI_Send(&v.ids_gnd.mat[0], v.ids_gnd.d*v.ids_gnd.n, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
			
		free(ivf);
	
	#endif

	free(v.ids_gnd.mat);
}

ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain){

	ivfpq_t ivfpq;
	ivfpq.coarsek = ivfpq.coa_centroidsn = coarsek;
	ivfpq.coa_centroidsd = vtrain.d;
	ivfpq.coa_centroids = (float*)malloc(sizeof(float)*ivfpq.coa_centroidsn*ivfpq.coa_centroidsd);

	float * dis = (float*)malloc(sizeof(float)*vtrain.n);

	//definicao de variaveis
	int flags = 0;
	flags = flags | KMEANS_INIT_BERKELEY;
	flags |= 1;
	flags |= KMEANS_QUIET;
	int* assign = (int*)malloc(sizeof(int)*vtrain.n);

	kmeans(vtrain.d, vtrain.n, coarsek, 50, vtrain.mat, flags, 2, 1, ivfpq.coa_centroids, NULL, NULL, NULL);

	//calculo do vetores residuais
	knn_full(L2, vtrain.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, 1, ivfpq.coa_centroids, vtrain.mat, NULL, assign, dis);
	subtract(vtrain, ivfpq.coa_centroids, assign, ivfpq.coa_centroidsd);

	//aprendizagem do produto residual
	ivfpq.pq = pq_new(nsq, vtrain);

	free(assign);
	free(dis);

	return ivfpq;
}

void subtract(mat v, float* v2, int* idx, int c_d){
	for (int i = 0; i < v.d; i++) {
		for (int j = 0; j < v.n; j++) {
			v.mat[j*v.d + i] = v.mat[j*v.d + i] - v2[idx[j]*c_d + i];
		}
	}
}

void copySubVectorsI(int* qcoaidx, int* coaidx, int query, int nq, int w){
	for (int i = 0; i < w; i++) {
		qcoaidx[i] = coaidx[query*w + i];
	}
}

void copySubVectors2(float* vout, float* vin, int dim, int nvec, int subn){
	for (int i = 0; i < dim; i++) {
			vout[i] = vin[nvec*dim + i + subn*dim];
	}
}

void ivfpq_assign(ivfpq_t ivfpq, mat vbase, ivf_t *ivf){
	int k = 1;

	
	int *assign = (int*)malloc(sizeof(int)*vbase.n);
	float *dis = (float*)malloc(sizeof(float)*vbase.n);
	
	//acha os indices para o coarse quantizer
	knn_full(2, vbase.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, k,
					 ivfpq.coa_centroids, vbase.mat, NULL, assign, dis);
	
	//residuos
	subtract(vbase, ivfpq.coa_centroids, assign, ivfpq.coa_centroidsd);
	
	matI codebook = pq_assign(ivfpq.pq, vbase);
	
	int *codeaux = (int*)malloc(sizeof(int)*codebook.d*codebook.n);
	memcpy(codeaux, codebook.mat, sizeof(int)*codebook.n*codebook.d);
		
	int * hist = (int*) calloc(ivfpq.coarsek,sizeof(int)); 
	histogram(assign, vbase.n ,ivfpq.coarsek, hist);
	
	// -- Sort on assign, new codebook with sorted ids as identifiers for codebook
	int* ids = (int*)malloc(sizeof(int)*vbase.n);
	ivec_sort_index(assign, vbase.n, ids);
	
	for(int i=0; i<codebook.n; i++){
		memcpy(codebook.mat+i*codebook.d, codeaux+codebook.d*ids[i], sizeof(int)*codebook.d);
	}
	
	int pos = 0;
	for (int i = 0; i < ivfpq.coarsek; i++) {
		int nextpos;
		
		ivf[i].ids = (int*)malloc(sizeof(int)*hist[i]);
		
		ivf[i].idstam = hist[i];
		
		ivf[i].codes.mat = (int*)malloc(sizeof(int)*hist[i]*ivfpq.pq.nsq);
		
		ivf[i].codes.n = hist[i];
		
		ivf[i].codes.d = ivfpq.pq.nsq;
		
		nextpos = pos+hist[i];
		
		memcpy(ivf[i].ids, &ids[pos], sizeof(int)*hist[i]);
		
		for (int p = pos; p < nextpos; p++) {
			copySubVectorsI(ivf[i].codes.mat + (p-pos)*ivf[i].codes.d, codebook.mat, p, 0,  ivf[i].codes.d);
		
		}
		
		pos += hist[i];
	}
	
	free(codebook.mat);
	
	free(codeaux);
	
	free(ids);
	
	free(hist);

	free(assign);
	
	free(dis);
}

void histogram(const int* vec, int n, int range, int *hist){
	
	for(int j=0; j<n; j++){
		hist[vec[j]]++;
	}
}