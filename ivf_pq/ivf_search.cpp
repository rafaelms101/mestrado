#include "ivf_search.h"

#define L2 2

void ivfpq_search(ivfpq_t ivfpq, ivf_t *ivf, mat vquery, int k, int w, int* ids, float* dis){
  // fmat_print (vquery.mat, vquery.n, vquery.d);
  // getchar();
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
  knn_full(L2, vquery.n, ivfpq.coa_centroidsn, ivfpq.coa_centroidsd,
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
      // ivec_print(qcoaidx, w);
      // getchar();

      //compute the w residual vectors
      v = bsxfunMINUS(vquery, ivfpq.coa_centroids, vquery.d, query, qcoaidx, w);

      //printMat(v.mat, v.n, v.d);
      //getchar();

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
        printf("j = %d\n", j);
        for (int q = 0; q < nsq; q++) {
          printf("q = %d\n", q);
          copySubVectors2(vsub.mat, v.mat, ds, j, q);
    			compute_cross_distances(vsub.d, vsub.n, ks, vsub.mat, ivfpq.pq.centroids[q], distab_temp);
          //fvec_print(distab_temp, ks);
    			memcpy(distab.mat+q*ks, distab_temp, sizeof(float)*ks);
        }
        printMat(distab.mat, distab.n, distab.d);
        //qidx.mat = (int*)realloc(qidx.mat, qidx.n + ivf[qcoaidx[j]].idstam);
        //qdis.mat = (float*)realloc(qdis.mat, qdis.n + ivf[qcoaidx[j]].codes.n);


        AUXSUMIDX = sumidxtab(distab, ivf[qcoaidx[j]].codes);
        fvec_print(AUXSUMIDX, ivf[qcoaidx[j]].codes.n);
        getchar();
        memcpy(qidx.mat + nextidx, ivf[qcoaidx[j]].ids,  sizeof(int)*ivf[qcoaidx[j]].idstam);
        memcpy(qdis.mat + nextdis, AUXSUMIDX, sizeof(float)*ivf[qcoaidx[j]].codes.n);

        nextidx += ivf[qcoaidx[j]].idstam;
        nextdis += ivf[qcoaidx[j]].codes.n;

        //free(AUXSUMIDX);
      }

      int ktmp = min(qidx.n, k);
      k_min(qdis, ktmp, dis1, ids1);

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
