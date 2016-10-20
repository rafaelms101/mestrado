#include "ivf_search.h"

#define L2 2

float * sumidxtab2(mat D, matI ids, int offset);

void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, int k, int w, int* ids, float* dis){

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

	dis = (float*)malloc(sizeof(float)*nq*k);
	ids = (int*)malloc(sizeof(int)*nq*k);

	int* coaidx = (int*)malloc(sizeof(int)*vquery.n*w);
	float* coadis = (float*)malloc(sizeof(float)*vquery.n*w);

	mat vsub;
	vsub.mat = (float*)malloc(sizeof(float)*ds);
	vsub.n = 1;
	vsub.d = ds;

	float* distab_temp = (float*)malloc(sizeof(float)*ks);

	int* qcoaidx = (int*)malloc(sizeof(int)*w);

	//find the w vizinhos mais prximos
	// knn_full(2, nq, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w,
	//          ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
					 w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);

	// printMatI(coaidx, vquery.n, w);
	// getchar();

	float* AUXSUMIDX;

	 matI qidx;
	 mat qdis;
	 qdis.mat = (float*)malloc(sizeof(float));
	 qidx.mat = (int*)malloc(sizeof(int));

	 float* dis1 = (float*)malloc(sizeof(float)*k);
	 int* ids1 = (int*)malloc(sizeof(int)*k);

	for (int query = 0; query < nq; query++) {
			copySubVectorsI(qcoaidx, coaidx, query, nq, w);
			//printf("qcoaidx = \n");
			//ivec_print(qcoaidx, w);
			//getchar();

			//compute the w residual vectors
			v = bsxfunMINUS(vquery, ivfpq.coa_centroids, vquery.d, query, qcoaidx, w);

			// printf("V = \n");
			// printMat(v.mat, v.n, v.d);
			// getchar();

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
					//printf("q = %d\n", q);
					copySubVectors2(vsub.mat, v.mat, ds, j, q);
					//printf("vsub =\n");
					//fvec_print(vsub.mat, vsub.d);
					compute_cross_distances(vsub.d, vsub.n, ks, vsub.mat, ivfpq.pq.centroids[q], distab_temp);
					//fvec_print(distab_temp, ks);
					memcpy(distab.mat+q*ks, distab_temp, sizeof(float)*ks);
				}
				// printMat(distab.mat, distab.n, distab.d);
				// getchar();
				//qidx.mat = (int*)realloc(qidx.mat, qidx.n + ivf[qcoaidx[j]].idstam);
				//qdis.mat = (float*)realloc(qdis.mat, qdis.n + ivf[qcoaidx[j]].codes.n);


				//printMatI(ivf[qcoaidx[j]].codes.mat, ivf[qcoaidx[j]].codes.n, ivf[qcoaidx[j]].codes.d);

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

			// printf("qdis = \n");
			// fvec_print(qdis.mat, qdis.n);
			// getchar();

			int ktmp = min(qidx.n, k);
			k_min(qdis, ktmp, dis1, ids1);

			//printMatI(ids1, 1 ,k);

			memcpy(dis + query*k, dis1, sizeof(float)*k);
			memcpy(ids + query*k, ids1, sizeof(int)*k);

	}
	free(qcoaidx);
	free(qidx.mat);
	free(qdis.mat);
	free(dis1);
	free(ids1);
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
