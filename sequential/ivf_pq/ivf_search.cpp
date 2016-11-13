#include "ivf_search.h"

#define L2 2

void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, int k, int w, int* ids, float* dis, mat vbase){

	int nq, d,ds, ks, nsq, nextdis, nextidx;
	nq = vquery.n;
	d = vquery.d;
	ds = ivfpq.pq.ds;
	ks = ivfpq.pq.ks;
	nsq =  ivfpq.pq.nsq;

	mat v;
	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;

	// dis = (float*)malloc(sizeof(float)*nq*k);
	// ids = (int*)malloc(sizeof(int)*nq*k);

	int* coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	float* coadis = (float*)malloc(sizeof(float)*vquery.n*w);

	int* coaidx2 = (int*)malloc(sizeof(int)*vbase.n);
	float* coadis2 = (float*)malloc(sizeof(float)*vbase.n);

	float* distab_temp = (float*)malloc(sizeof(float)*ks);

	int* qcoaidx = (int*)malloc(sizeof(int)*w);

	//find the w vizinhos mais prximos
	// knn_full(2, nq, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w,
	//          ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
					 w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);


	knn_full(2, vbase.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
				 	 1, ivfpq.coa_centroids, vbase.mat, NULL, coaidx2, coadis2);

	//printMatI(coaidx2, vbase.n, 1);
	//printMatI(coaidx, vquery.n, w);
	//getchar();

	float* AUXSUMIDX;

	 matI qidx;
	 mat qdis;
	 qdis.mat = (float*)malloc(sizeof(float));
	 qidx.mat = (int*)malloc(sizeof(int));

	 float* dis1 = (float*)malloc(sizeof(float)*k);
	 int* ids1 = (int*)malloc(sizeof(int)*k);

	 int* ids2 = (int*)malloc(sizeof(int)*(k/2));

	 float** y = (float **) malloc(sizeof(float*)*nsq);
	 for (int i=0; i < nsq; i++){
		 y[i]=(float *) malloc(sizeof(float)*ds);
	 }

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

			qdis.mat = (float*)realloc(qdis.mat, sizeof(float)*qdis.n);
			qidx.mat = (int*)realloc(qidx.mat, sizeof(int)*qidx.n);

			for (int j = 0; j < w; j++) {
				//printf("j = %d\n", j);
				for (int q = 0; q < nsq; q++) {
					compute_cross_distances(ds, 1, ks, v.mat + j*v.d + q*ds, ivfpq.pq.centroids[q], distab_temp);

					memcpy(distab.mat+q*ks, distab_temp, sizeof(float)*ks);
				}

				AUXSUMIDX = sumidxtab2(distab, ivf[qcoaidx[j]].codes, 0);
				// printf("A=SUMIDXTAB = \n");
				// fvec_print(AUXSUMIDX, ivf[qcoaidx[j]].codes.n);
				// getchar();
				memcpy(qidx.mat + nextidx, ivf[qcoaidx[j]].ids,  sizeof(int)*ivf[qcoaidx[j]].idstam);
				memcpy(qdis.mat + nextdis, AUXSUMIDX, sizeof(float)*ivf[qcoaidx[j]].codes.n);

				nextidx += ivf[qcoaidx[j]].idstam;
				nextdis += ivf[qcoaidx[j]].codes.n;

				free(AUXSUMIDX);
			}

			int ktmp = min(qidx.n, k);
			k_min(qdis, ktmp, dis1, ids1);



			//RERANKING
			nextdis = 0;
			qdis.n = 0;
			qdis.d = 1;
			float *distances = (float*)calloc(ktmp, sizeof(float));
			for (int i = 0; i < ktmp; i++) {
					distances[i] = 0.0;
					//printf(" i = %d coaidx2[ids1[i]] = %d\n",i , coaidx2[ids1[i]]);
					//v = bsxfunMINUS(vquery, ivfpq.coa_centroids, vquery.d, query, &coaidx2[qidx.mat[ids1[i]-1]], 1);

					fillY(y, coaidx2[qidx.mat[ids1[i]-1]], vquery.mat +  vquery.d*query, ivfpq, ivf, k);

					float distance = 0.0;
					for (int q = 0; q < nsq; q++) {
						compute_cross_distances(ds, 1, 1, vquery.mat +  vquery.d*query, y[q], &distance);
						//printf("compute_cross_distances = %f\n", distance);
						//L2_distance(vquery.mat +  vquery.d*query + q*ds, y[q], ds, &distance);
						//printf("L2 = %f\n", distance);

						distances[i] += distance;
					}

			}
			fvec_k_min(distances, ktmp, ids2, k/2);


			memcpy(dis + query*k, distances, sizeof(float)*k);
			for(int b = 0; b < k ; b++){
				//memcpy(ids + query*k + b, qidx.mat + (ids1[b]-1)*qidx.d, sizeof(int));
				if(b >= ktmp){
					ids[query*k + b] = -1;
				}
				else{
					//printf("ids2[b] = %d ids1[ids2[b]]-1 = %d\n", ids2[b], ids1[ids2[b]]);
					ids[query*k + b] = qidx.mat[ids1[b]-1];
				}
			}

			// FOR RERANKING
			// memcpy(dis + query*k/2, distances, sizeof(float)*k/2);
			// for(int b = 0; b < k/2 ; b++){
			// 	//memcpy(ids + query*k/2 + b, qidx.mat + (ids1[b]-1)*qidx.d, sizeof(int));
			// 	if(b >= ktmp){
			// 		ids[query*k/2 + b] = -1;
			// 	}
			// 	else{
			// 		//printf("ids2[b] = %d ids1[ids2[b]]-1 = %d\n", ids2[b], ids1[ids2[b]]);
			// 		ids[query*k/2 + b] = qidx.mat[ids1[ids2[b]-1]-1];
			// 	}
			// }

	}
	free(qcoaidx);
	free(qidx.mat);
	free(qdis.mat);
	free(dis1);
	free(ids1);
	//free(distances);
	free(ids2);
	for (size_t a = 0; a < nsq; a++) {
		free(y[a]);
	}
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
		for(j=0; j<D.n; j++){
			// printf("i = %d   j = %d\n", i, j);
			// printf("ids = %d\n", ids.mat[i*ids.d+j]);
			// printf("D.mat = %f\n", D.mat[ids.mat[i*ids.d+j]+ offset + j*D.d]);
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


//  yi = qc(yi) + qr(yi);
//  qc = ivfpq.coa_centroids
//  qr = ivfpq.pq.centorids[0 ... nsq][]
void fillY(float** y, int ids, float* v , ivfpq_t ivfpq, ivf_t *ivf, int k){
	int ds = ivfpq.pq.ds;
	int nsq = ivfpq.pq.nsq;
	int cd = ivfpq.coa_centroidsd;

	int index = ids;
	// int ridx[1];
	// float dis[1];
	for (int q = 0; q < nsq; q++) {
		for (int j = 0; j < ds; j++) {
				//knn_full(2, 1, ivfpq.pq.ks, ds, 1, v + q*ds ,ivfpq.pq.centroids[q], NULL,ridx, dis);
				//myKnn_single(v+q*ds, ivfpq.pq.centroids[q], ivfpq.pq.ks, ds, &ridx);
				y[q][j] = ivfpq.pq.centroids[q][(index)*ds + j] + ivfpq.coa_centroids[index*cd + q*ds + j];

		}
	}

}


void myKnn_single(float* v, float *base, int basen, int ds , int *idxsaida){
	float min = 100000000.0;

	float dis;
	for (int i = 0; i < basen; i++) {
		L2_distance(v, base + i*ds, ds, &dis);
		if(dis < min){
			*idxsaida = i;
			min = dis;
		}
	}
}

void L2_distance(float* vin, float* vin2, int n, float* dis){
	float aux;
	float res = 0.0;
	for (int i = 0; i < n; i++) {
		aux = vin[i] - vin2[i];
		res += (aux * aux);
	}
	*dis = res;

}
