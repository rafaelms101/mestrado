#include "pq_search.h"

void pq_search(pqtipo pq, int *cbase, mat vquery, int k){
	int i,
		j,
		*ids;

	float 	*distab,
			*dis;

	mat vsub;		

	for (i=0;i<vquery.n;i++){

		for (j=0;j<pq.nsq;j++){
			copySubVectors(vsub.mat,vquery.mat,pq.ds,j,i,i);
			compute_cross_distances (vsub.d, vsub.n, pq.centroids.n, vsub.mat, pq.centroids.mat, distab);
			//criar sumidxtab
		}


	}
}	