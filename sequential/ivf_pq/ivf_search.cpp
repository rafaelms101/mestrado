#include "ivf_search.h"

#define L2 2

 void Y(ivfpq_t ivfpq, ivf_t* ivf, int ds ,float* coarse, int d, int nsq, int* qcoaidx, int id, int w ,float* y);

void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, int k, int kl ,int w, int* ids, float* dis, mat vbase){


	//printf("codes.n = %d, codes.d = %d\n", ivf[0].codes.n, ivf[0].codes.d);

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

	float* distab_temp = (float*)malloc(sizeof(float)*ks);

	int* qcoaidx = (int*)malloc(sizeof(int)*w);

	//find the w vizinhos mais prximos
	// knn_full(2, nq, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd, w,
	//          ivfpq.coa_centroids, vquery.mat, NULL , coaidx, coadis);
	knn_full(2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
					 w, ivfpq.coa_centroids, vquery.mat, NULL, coaidx, coadis);

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

	 int* ids2 = (int*)malloc(sizeof(int));

	 float* y = (float *) malloc(sizeof(float)*d);
	//  for (int i=0; i < nsq; i++){
	// 	 y[i]=(float *) malloc(sizeof(float)*ds);
	//  }
//	nq =1 ;
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

					memcpy(distab.mat + q*ks, distab_temp, sizeof(float)*ks);
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

      // printf("QUERY == %d IDS1 = \n", query);
      // printMatI(ids1, ktmp, 1);
      // printf("DIS1 == \n");
      // printMat(dis1, ktmp, 1);

      //RE-RANKING
			ids2 = (int*)realloc(ids2, sizeof(int)*kl);
			float distance;
      float dis2[ktmp];
			for (int i = 0; i < ktmp; i++) {

				Y(ivfpq, ivf, ds, ivfpq.coa_centroids + qcoaidx[0]*d, d, nsq, qcoaidx, ids1[i]-1, w, y);
				compute_cross_distances(d, 1, 1, vquery.mat + query*vquery.d,
																y, &distance);
				dis2[i] = distance;
			}

			fvec_k_min(dis2, ktmp, ids2, kl);

      // printf("IDS2 = \n");
      // printMatI(ids2, kl, 1);
      // printf("DIS2 == \n");
      // printMat(dis2, ktmp, 1);

      memcpy(dis + query*kl, dis1, sizeof(float)*kl);
      for(int b = 0; b < kl ; b++){
        //memcpy(ids + query*kl + b, qidx.mat + (ids1[b]-1)*qidx.d, sizeof(int));
        if(b >= kl){
          ids[query*kl + b] = -1;
        }
        else{
          //printf("ids2[b] = %d ids1[ids2[b]]-1 = %d\n", ids2[b], ids1[ids2[b]]);
          ids[query*kl + b] = qidx.mat[ids1[ids2[b]]-1];
          //printf("qidx.mat[ids1[b]-1] = %d\n", qidx.mat[ids1[b]-1]);
	       }
      }

      //RE-RANKING END

			// memcpy(dis + query*k, dis1, sizeof(float)*k);
			// for(int b = 0; b < k ; b++){
			// 	if(b >= ktmp){
			// 		ids[query*k + b] = -1;
			// 	}
			// 	else{
			// 		ids[query*k + b] = qidx.mat[ids1[b]]-1];
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
	free(y);
	// for (size_t a = 0; a < nsq; a++) {
	// 	free(y[a]);
	// }
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

// w -> numero de vizinhos do coarse a serem visitador
//ds ->subdimensao
//d ->dimensao,
//nsq -> m
//id, ->indice correspondente a lista ja calculada
void Y(ivfpq_t ivfpq, ivf_t* ivf, int ds ,float* coarse, int d, int nsq, int* qcoaidx, int id, int w ,float* y){

  //printf("id = %d\n", id);
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
  //printf("qcoaj = %dqcoaidx[qcoaj] = %d\n", qcoaj, qcoaidx[qcoaj]);
	for (int i = 0; i < nsq; i++) {
		int subID = ivf[qcoaidx[qcoaj]].codes.mat[id*nsq + i];
    //printf("subID = %d\n", subID);
		for (int j = 0; j < ds; j++) {
			y[i*ds +j] = coarse[i*ds +j] + ivfpq.pq.centroids[i][subID*ds + j];
      //printf("%.3f = %.3f + %.3f\n", y[i*ds +j], coarse[i*ds +j], ivfpq.pq.centroids[i][subID*ds + j]);
		}
	}
}
