#include "ivf_search.h"

#define L2 2

void Y(ivfpq_t ivfpq, ivf_t* ivf, int ds, int d, int nsq, int* qcoaidx, int id, int w ,float* y);
void printCodes(ivf_t ivf);
void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, mat vbase, int k ,int kl, int w, int* ids, float* dis){

    int nq, d,ds, ks, nsq, nextdis, nextidx;
	nq = vquery.n;        //number of queries
	d = vquery.d;         //Dimension
	ds = ivfpq.pq.ds;     //Subdimension dimensions
	ks = ivfpq.pq.ks;     //number of ...
	nsq =  ivfpq.pq.nsq;  //Subdimension number
	mat v;

    int counter=0;

    //tabulated distances
	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;

	int* coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	float* coadis = (float*)malloc(sizeof(float)*vquery.n*w);

	float* distab_temp = (float*)malloc(sizeof(float)*ks);

	int* qcoaidx = (int*)malloc(sizeof(int)*w);

	//find the w vizinhos mais prximos
	// knn_full(2, nq, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w,
	//          ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
					 w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);

	float* AUXSUMIDX;

    //it stores the indexes(or indentifier) of vectors in database
	matI qidx;
    //it stores the distances between thos vectors and the query
	mat qdis;
	qdis.mat = (float*)malloc(sizeof(float));
	qidx.mat = (int*)malloc(sizeof(int));

	float* y = (float *) malloc(sizeof(float)*d);
	//nq =1 ;
    //for each query search the closest vectors in database
    for (int query = 0; query < nq; query++) {
        copySubVectorsI(qcoaidx, coaidx, query, nq, w);

        //compute the w residual vectors
        v = bsxfunMINUS(vquery, ivfpq.coa_centroids, vquery.d, query, qcoaidx, w);

        nextidx = 0;
        nextdis = 0;
        qidx.n = 0;
        qidx.d = 1;
        qdis.n = 0;
        qdis.d = 1;

        for (int a = 0; a < w; a++) {
            qdis.n += ivf[qcoaidx[a]].codes.n;
            qidx.n += ivf[qcoaidx[a]].idstam;
        }

        qdis.mat = (float*)malloc(sizeof(float)*qdis.n);
        qidx.mat = (int*)malloc(sizeof(int)*qidx.n);

        //compute the distancee between all neighbours inside w indexes in ivf

        for (int j = 0; j < w; j++) {
            for (int q = 0; q < nsq; q++) {
                compute_cross_distances(ds, 1, ks, v.mat + j*v.d + q*ds, ivfpq.pq.centroids[q], distab_temp);

                memcpy(distab.mat + q*ks, distab_temp, sizeof(float)*ks);
            }
            AUXSUMIDX = sumidxtab2(distab, ivf[qcoaidx[j]].codes, 0);
            memcpy(qidx.mat + nextidx, ivf[qcoaidx[j]].ids,  sizeof(int)*ivf[qcoaidx[j]].idstam);
            memcpy(qdis.mat + nextdis, AUXSUMIDX, sizeof(float)*ivf[qcoaidx[j]].codes.n);

            nextidx += ivf[qcoaidx[j]].idstam;
            nextdis += ivf[qcoaidx[j]].codes.n;

            free(AUXSUMIDX);
        }

        //get the neighbours with smaller distances() where its indexes will be
        //stored in ids1, and its distances in dis1
        int ktmp = min(qidx.n, kl);

        float* dis1 = (float*)malloc(sizeof(float)*ktmp);
    	int* ids1 = (int*)malloc(sizeof(int)*ktmp);

        k_min(qdis, ktmp, dis1, ids1);

        //RERANK

        int *ids_base = (int*)malloc(sizeof(int)*ktmp);
        float *nearest = (float*)malloc(sizeof(float)*vbase.d*ktmp);

        for(int b = 0; b < ktmp ; b++){
            ids_base[b] = qidx.mat[ids1[b]-1];
            memcpy(&nearest[b*vbase.d], &vbase.mat[ids_base[b]*vbase.d],  sizeof(float)*vbase.d);
        }

        float *lquery = (float*)malloc(sizeof(float)*vquery.d);
        memcpy(lquery, &vquery.mat[query*vquery.d],  sizeof(float)*vquery.d);

        int* ids2 = (int*)malloc(sizeof(int)*k);
        float* dis2 = (float*)malloc(sizeof(float)*k);


        knn_full(2, 1, ktmp, vbase.d, k, nearest, lquery, NULL , ids2 , dis2);

        for(int c=0; c<k; c++){
            ids[query*k+c] = ids_base[ids2[c]];
        }

        free(dis1);
    	free(ids1);
        free(dis2);
    	free(ids2);
        free(ids_base);
        free(lquery);
        free(qidx.mat);
      	free(qdis.mat);

    }

	free(qcoaidx);
	free(y);
	free(v.mat);
}

mat bsxfunMINUS(mat vin, float* vin2, int dim, int nq, int* qcoaidx, int ncoaidx){

	static mat mout;
	mout.mat = (float*)malloc(sizeof(float)*dim*ncoaidx);
	mout.d = dim;
	mout.n = ncoaidx;

	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < ncoaidx; j++) {
				//printf("vin[%d] = %f\n",vin.d*nq + j, vin.mat[vin.d*nq+j]);
				//printf("centroid(%d, %d) %f\n", qcoaidx[i], j ,vin2[qcoaidx[i]*dim+j]);
				mout.mat[j*dim+i] = vin.mat[(vin.d*nq) + i] - vin2[(qcoaidx[j]*dim)+i];
				//getchar();
		}
	}

	return mout;
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
	float dis_tmp = 0;
	int i, j, idsN = 0;

	//soma as distancias para cada vetor
	for (i = 0; i < ids.n ; i++) {
		dis_tmp = 0;
    //printf("subID = %d\n", i);
		for(j=0; j<D.n; j++){
			// printf("i = %d   j = %d\n", i, j);
			// printf("ids = %d\n", ids.mat[i*ids.d+j]);
			// printf("D.mat = %f\n", D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d]);
			dis_tmp += D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d];
      //printf("conteudo = %d\n", ids.mat[i*ids.d+j]);
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


//  yi = qc(yi) + qr(yi);
//  qc = ivfpq.coa_centroids
//  qr = ivfpq.pq.centroids[0 ... nsq][]

// w -> numero de vizinhos do coarse a serem visitador
//ds ->subdimensao
//d ->dimensao,
//nsq -> m
//id, ->indice correspondente a lista ja calculada
void Y(ivfpq_t ivfpq, ivf_t* ivf, int ds ,int d, int nsq, int* qcoaidx, int id, int w ,float* y){

  //find in wich neighbour of ivf the id is
	int qcoaj = 0;
	for (int i = 0; i < w; i++) {
		if (id - ivf[qcoaidx[i]].codes.n < 0) {
			break;
		}
		else{
			id -= ivf[qcoaidx[i]].codes.n;
			qcoaj++;
		}
	}

  for (int i = 0; i < nsq; i++) {
    //find the quantization code from the vector in base
		int subID = ivf[qcoaidx[qcoaj]].codes.mat[id*nsq + i];

    //sum each subdimension of qr into qc
		for (int j = 0; j < ds; j++) {
			y[i*ds +j] = ivfpq.coa_centroids[qcoaidx[qcoaj]*ivfpq.coa_centroidsd + i*ds +j] + ivfpq.pq.centroids[i][subID*ds + j];
		}
	}
}

void printCodes(ivf_t ivf){

  for (size_t i = 0; i < ivf.codes.n; i++) {
    printf("id = %d\n", ivf.ids[i]);
    for (size_t j = 0; j < ivf.codes.d; j++) {
      printf(" === %d\n", ivf.codes.mat[j + i*ivf.codes.d]);
    }
  }

}
