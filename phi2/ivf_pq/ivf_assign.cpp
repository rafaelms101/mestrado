#include "ivf_assign.h"

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
