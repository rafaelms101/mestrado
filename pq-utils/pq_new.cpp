#include "pq_new.h"

// nsq: numero de subquantizadores (m no paper)
// vtrain: estrutura de vetores, contendo os dados de treinamento

pqtipo pq_new(int nsq, mat vtrain){

	// ds: dimensao dos subvetores
	// ks: numero de centroides por subquantizador
	// flags: modo de inicializacao do kmeans
	// assign: vetor que guarda os indices dos centroides
	// seed: semente do kmeans (qualquer valor diferente de 0)
	// centroids_tmp: vetor que guarda temporariamente os centroides de um subvetor
	// dis: vetor que guarda as distancias entre um centroide e o vetor assinalado por ele
	// vs: vetor que guarda temporariamente cada subvetor
	// pq: estrutura do quantizador

	int	i,
		ds,
		ks,
		flags=0,
		*assign,
		seed=2;

	float	*centroids_tmp,
			*dis,
			*vs;

	pqtipo pq;

	// definicao de variaveis

	flags = flags | KMEANS_INIT_BERKELEY;
	flags |= 1;
	flags |= KMEANS_QUIET;
	ds=vtrain.d/nsq;
	ks=pow(2,nsq);
	pq.nsq = nsq;
	pq.ks = ks;
	pq.ds = ds;
	pq.centroidsn=ks*ds;
	pq.centroidsd=nsq;
	pq.centroids=(float **) malloc(sizeof(float*)*nsq);
	for (i=0; i<nsq; i++){
		pq.centroids[i]=(float *) malloc(sizeof(float)*ks*ds);
	}

	// alocacao de memoria

	centroids_tmp= fvec_new(ks*ds);
	dis = fvec_new(vtrain.n);
	assign= ivec_new(vtrain.n);
	vs=fmat_new (ds, vtrain.n);

	// criacao dos centroides

	for(i=0;i<nsq;i++){
		copySubVectors(vs,vtrain,ds,i,0,vtrain.n-1);
		kmeans(ds, vtrain.n, ks, 100, vs, flags, seed, 1, centroids_tmp , dis, assign, NULL);
		memcpy(pq.centroids[i], centroids_tmp, sizeof(float)*ks*ds);
	}
	return pq;
}

void check_new(){
  cout << ":::PQ_NEW OK:::" << endl;
}

// Função que concatena dois vetores de numeros reais

void fvec_concat(float* vinout, int vinout_n, float* vin, int vin_n){
	if (vinout == NULL) {
		vinout = (float*)malloc(sizeof(float)*vin_n);
		vinout_n = 0;
		memcpy(vinout + vinout_n, vin, sizeof(float)*vin_n);
	}
	else{
		fvec_resize(vinout, vinout_n + vin_n);
		memcpy(vinout + vinout_n, vin, sizeof(float)*vin_n);
	}
}

// Função que concatena dois vetores de numeros inteiros

void ivec_concat(int* vinout, int vinout_n, int* vin, int vin_n){
	if (vinout == NULL) {
		vinout = (int*)malloc(sizeof(int)*vin_n);
		vinout_n = 0;
		memcpy(vinout + vinout_n, vin, sizeof(int)*vin_n);
	}
	else{
		ivec_resize(vinout, vinout_n + vin_n);
		memcpy(vinout + vinout_n, vin, sizeof(int)*vin_n);
	}
}

// Função que copia um intervalo de um vetor, criando um subvetor
// vout : vetor de saída
// vin : estrutura com o vetor de entrada
// ds: subdimensão
// n1row : numero do subvetor a ser copiado (ex: v1[16] -> 8 vetores v2[2]. n1row= i variando de 0 a 7)
// n1col : coluna inicial a ser copiada
// n2col : coluna final a ser copiada

void copySubVectors(float *vout, mat vin, int ds, int n1row, int n1col, int n2col) {

    for (int i = n1col; i <= n2col ; i++) {
        memcpy(vout +(i-n1col)*ds, vin.mat+vin.d*i+n1row*ds, sizeof(float)*(ds));
    }
}

// Função que concatena vetores em uma matriz transposta

void ivec_concat_transp(matI vinout, int* vin, int nsq){
	for(int i=0; i<vinout.n; i++){
		memcpy(vinout.mat + vinout.d+nsq*i, vin+i, sizeof(int));
	}

}
