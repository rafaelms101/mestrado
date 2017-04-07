#include "ivf_search.h"

#define L2 2


__declspec(target(mic)) float * sumidxtab2(mat D, matI ids, int offset);

__declspec(target(mic)) dis_t ivfpq_search(ivf_t *ivf, mat residual, pqtipo pq, int centroid_idx){
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
		compute_cross_distances(ds, 1, distab.d, &residual.mat[query*ds], &pq.centroids[query*ks*ds], distab_temp);
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

void bsxfunMINUS(mat mout, mat vin, float* vin2, int nq, int* qcoaidx, int ncoaidx){
	for (int i = 0; i < vin.d; i++) {
		for (int j = 0; j < ncoaidx; j++) {
			mout.mat[j*vin.d+i] = vin.mat[(vin.d*nq) + i] - vin2[(qcoaidx[j]*vin.d)+i];
		}
	}
}

__declspec(target(mic)) int min(int a, int b){
	if(a>b){
		return b;
	}
	else
		return a;
}

__declspec(target(mic)) float * sumidxtab2(mat D, matI ids, int offset){
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
