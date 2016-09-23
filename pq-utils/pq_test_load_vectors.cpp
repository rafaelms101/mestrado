#include "pq_test_load_vectors.h"

data pq_test_load_vectors(char* dataset){

	data v;    

	if(strcmp(dataset, "random")==0){

		v.base.n=1000000;
		v.base.d=16;
		v.base.mat= fmat_new (v.base.d, v.base.n);

		v.query.n=1000;
		v.query.d=16;
		v.query.mat= fmat_new (v.query.d, v.query.n);

		v.train.n=10000;
		v.train.d=16;
		v.train.mat= fmat_new (v.train.d, v.train.n);

		v.ids_gnd.n=1000;
		v.ids_gnd.d=100;
		v.ids_gnd.mat= ivec_new (v.ids_gnd.d*v.ids_gnd.n);
		float *dis_gnd= fmat_new (v.ids_gnd.d,v.ids_gnd.n);

		srand (time(NULL));

		///inicializa com valores aleatórios

		load_random(v.train.mat, v.train.n, v.train.d);
		load_random(v.base.mat, v.base.n, v.base.d);
		load_random(v.query.mat, v.query.n, v.query.d);

		knn_full(2, v.query.n, v.base.n, v.base.d, 1 , v.base.mat, v.query.mat, NULL , v.ids_gnd.mat , dis_gnd);

	}
	else {
		namefile f;

		f.base= (char*)malloc(sizeof(char)*30);
		f.query= (char*) malloc(sizeof(char)*30);
		f.train= (char*) malloc(sizeof(char)*30);
		f.groundtruth= (char*) malloc(sizeof(char)*30);

		if(strcmp(dataset, "siftsmall")==0){
			strcpy (f.base,"./siftsmall/siftsmall_base.fvecs");
			v.base.n=10000;
			v.base.d=128;
			v.base.mat= fmat_new (v.base.d, v.base.n);
			strcpy (f.query,"./siftsmall/siftsmall_query.fvecs");
			v.query.n=100;
			v.query.d=128;
			v.query.mat= fmat_new (v.query.d, v.query.n);
			strcpy (f.train,"./siftsmall/siftsmall_learn.fvecs");
			v.train.n=25000;
			v.train.d=128;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"./siftsmall/siftsmall_groundtruth.ivecs");
			v.ids_gnd.n=100;
			v.ids_gnd.d=100;
			v.ids_gnd.mat= ivec_new (v.ids_gnd.d*v.ids_gnd.n);
		}
		else if(strcmp(dataset, "sift")==0){
			strcpy (f.base,"./sift/sift_base.fvecs");
			v.base.n=1000000;
			v.base.d=128;
			v.base.mat= fmat_new (v.base.d, v.base.n);
			strcpy (f.query,"./sift/sift_query.fvecs");
			v.query.n=10000;
			v.query.d=128;
			v.query.mat= fmat_new (v.query.d, v.query.n);
			strcpy (f.train,"./sift/sift_learn.fvecs");
			v.train.n=100000;
			v.train.d=128;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"./sift/sift_groundtruth.ivecs");
			v.ids_gnd.n=10000;
			v.ids_gnd.d=100;
			v.ids_gnd.mat= ivec_new (v.ids_gnd.d*v.ids_gnd.n);
		}
		else if(strcmp(dataset, "gist")==0){
			strcpy (f.base,"./gist/gist_base.fvecs");
			v.base.n=1000000;
			v.base.d=960;
			v.base.mat= fmat_new (v.base.d, v.base.n);
			strcpy (f.query,"./gist/gist_query.fvecs");
			v.query.n=1000;
			v.query.d=960;
			v.query.mat= fmat_new (v.query.d, v.query.n);
			strcpy (f.train,"./gist/gist_learn.fvecs");
			v.train.n=500000;
			v.train.d=960;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"./gist/gist_groundtruth.ivecs");
			v.ids_gnd.n=1000;
			v.ids_gnd.d=100;
			v.ids_gnd.mat= ivec_new (v.ids_gnd.d*v.ids_gnd.n);
		}
		fvecs_read (f.base, v.base.d, v.base.n, v.base.mat);
		fvecs_read (f.query, v.query.d, v.query.n, v.query.mat);
		fvecs_read (f.train, v.train.d, v.train.n, v.train.mat);
		ivecs_read (f.groundtruth, v.ids_gnd.d, v.ids_gnd.n, v.ids_gnd.mat);
	}	
	return v;
}

void load_random (float *v, int n, int d){
	int i;

	for(i=0;i<n*d;i++){
		v[i]= (float) rand()/RAND_MAX;
	}
}

int ivecs_read (const char *fname, int d, int n, int *a){
	FILE *f = fopen (fname, "r");
	
	if (!f) {
		fprintf (stderr, "ivecs_read: could not open %s\n", fname);
		perror ("");
		return -1;
	}

	long i;
	for (i = 0; i < n; i++) {
		int new_d;

		if (fread (&new_d, sizeof (int), 1, f) != 1) {
			if (feof (f))break;
			else {
				perror ("fvecs_read error 1");
				fclose(f);
				return -1;
			}
		}

		if (new_d != d) {
			fprintf (stderr, "fvecs_read error 2: unexpected vector dimension\n");
			fclose(f);
			return -1;
		}

		if (fread (a + d * (long) i, sizeof (int), d, f) != d) {
			fprintf (stderr, "fvecs_read error 3\n");
			fclose(f);
			return -1;
		}
	}
	fclose (f);

	return i;
}