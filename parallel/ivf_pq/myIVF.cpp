#include "myIVF.h"

void my_k_min(dis_t q, int ktmp, float *dis, int *ids){
	int f, *qidx, j=1;
	float *qdis;

	if (q.dis.n == 0 || ktmp == 0)
    return;
	
	qdis = (float*)malloc(sizeof(float)*q.dis.n);
	qidx = (int*)malloc(sizeof(int)*q.idx.n);
	
	memcpy(qdis, q.dis.mat, sizeof(float)*q.dis.n);
	memcpy(qidx, q.idx.mat, sizeof(int)*q.idx.n);
	
	constroiHeap(q.dis.n, qdis, qidx);	
	
	dis[0] = qdis[0];
    qdis[0] = qdis[q.dis.n-1];
    ids[0] = qidx[0];
    qidx[0] = qidx[q.dis.n-1];
    
	for (int i = q.dis.n-1; i > q.dis.n-ktmp; i--){
	
		trocarRaiz(q.dis.n, qdis, qidx);
		
		dis[j] = qdis[0];
        qdis[0] = qdis[i-1];
        ids[j] = qidx[0];
        qidx[0] = qidx[i-1];
        
        j++;
	}
	
	free(qdis);
	free(qidx);
	
}

static void constroiHeap (int n, float *qdis, int *qidx){
	int k, aux;
	float aux2; 
   
	for (k = 1; k < n; k++) {                   
		int f = k+1;
		while (f > 1 && qdis[(f/2)-1] > qdis[f-1]) {  
        	aux2 = qdis[(f/2)-1];
        	qdis[(f/2)-1] = qdis[f-1];
        	qdis[f-1] = aux2;
        	aux = qidx[(f/2)-1];
        	qidx[(f/2)-1] = qidx[f-1];
        	qidx[f-1] = aux;
			f /= 2;                        
		}
	}
}

static void trocarRaiz (int n, float *qdis, int *qidx){ 
	int p = 1, f = 2, aux2 = qidx[0];
	float aux = qdis[0];
	while (f < n) {
		if (f < n-1 && qdis[f-1] > qdis[f])  f++;
		if (aux <= qdis[f]) break;
		qdis[p-1] = qdis[f-1];
		qidx[p-1] = qidx[f-1];
		p = f;
		f = 2*p;
	}
	qdis[p-1] = aux;
	qidx[p-1] = aux2;
}