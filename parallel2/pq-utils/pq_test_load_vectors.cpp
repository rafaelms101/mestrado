#include "pq_test_load_vectors.h"

data pq_test_load_vectors(char* dataset, int tam){

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
		v.ids_gnd.d=1;
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
		int *ids_gnd;
		namefile f;

		f.train= (char*) malloc(sizeof(char)*51);
		f.groundtruth= (char*) malloc(sizeof(char)*51);

		if(strcmp(dataset, "siftsmall")==0){
			strcpy (f.train,"../siftsmall/siftsmall_learn.fvecs");
			v.train.n=25000;
			v.train.d=128;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"../siftsmall/siftsmall_groundtruth.ivecs");
			v.ids_gnd.n=100;
			v.ids_gnd.d=100;
			ids_gnd= ivec_new (v.ids_gnd.n*v.ids_gnd.d);
		}
		else if(strcmp(dataset, "sift")==0){
			strcpy (f.train,"../sift/sift_learn.fvecs");
			v.train.n=100000;
			v.train.d=128;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"../sift/sift_groundtruth.ivecs");
			v.ids_gnd.n=10000;
			v.ids_gnd.d=100;
			ids_gnd= ivec_new (v.ids_gnd.n*v.ids_gnd.d);
		}
		else if(strcmp(dataset, "siftbig")==0 ){
			if(tam==200000000){
				strcpy (f.groundtruth,"/scratch/04596/tg838951/siftbig_groundtruth_200M.ivecs");
			}
			else if(tam==500000000){
				strcpy (f.groundtruth,"/scratch/04596/tg838951/siftbig_groundtruth_500M.ivecs");
			}
			else if(tam==1000000000){
				strcpy (f.groundtruth,"/scratch/04596/tg838951/siftbig_groundtruth_1000M.ivecs");
			}
			else{
				strcpy (f.groundtruth,"/scratch/04596/tg838951/siftbig_groundtruth_100M.ivecs");
			}
			strcpy (f.train,"/scratch/04596/tg838951/siftbig_learn.bvecs");
			v.train.n=tam/100;
			v.train.d=128;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			v.ids_gnd.n=10000;
			v.ids_gnd.d=1000;
			ids_gnd= ivec_new (v.ids_gnd.n*v.ids_gnd.d);
		}
		else if(strcmp(dataset, "gist")==0){
			strcpy (f.train,"../gist/gist_learn.fvecs");
			v.train.n=500000;
			v.train.d=960;
			v.train.mat= fmat_new (v.train.d, v.train.n);
			strcpy (f.groundtruth,"../gist/gist_groundtruth.ivecs");
			v.ids_gnd.n=1000;
			v.ids_gnd.d=100;
			ids_gnd= ivec_new (v.ids_gnd.n*v.ids_gnd.d);
		}

		if(strcmp(dataset, "siftbig")!=0){
			fvecs_read (f.train, v.train.d, v.train.n, v.train.mat);
			ivecs_read (f.groundtruth, v.ids_gnd.d , v.ids_gnd.n, ids_gnd);
		}
		else{
			b2fvecs_read (f.train, v.train.d, v.train.n, v.train.mat);
			ivecs_read (f.groundtruth, v.ids_gnd.d , v.ids_gnd.n, ids_gnd);
		}

		v.ids_gnd.mat= ivec_new (v.ids_gnd.n);
		for(int i=0; i<v.ids_gnd.n;i++){
			v.ids_gnd.mat[i]=ids_gnd[i*v.ids_gnd.d];
		}
		v.ids_gnd.d=1;
		free(ids_gnd);
		free(f.train);
		free(f.groundtruth);
	}
	return v;
}

mat pq_test_load_query(char* dataset){

	mat vquery;

	if(strcmp(dataset, "random")==0){

		vquery.n=1000;
		vquery.d=16;
		vquery.mat= fmat_new (vquery.d, vquery.n);

		srand (time(NULL));

		///inicializa com valores aleatórios

		load_random(vquery.mat, vquery.n, vquery.d);
	}
	else {
		char *fquery;

		fquery= (char*) malloc(sizeof(char)*51);

		if(strcmp(dataset, "siftsmall")==0){
			strcpy (fquery,"../siftsmall/siftsmall_query.fvecs");
			vquery.n=100;
			vquery.d=128;
			vquery.mat= fmat_new (vquery.d, vquery.n);
		}
		else if(strcmp(dataset, "sift")==0){
			strcpy (fquery,"../sift/sift_query.fvecs");
			vquery.n=10000;
			vquery.d=128;
			vquery.mat= fmat_new (vquery.d, vquery.n);
		}
		else if(strcmp(dataset, "siftbig")==0){
			strcpy (fquery,"/scratch/04596/tg838951/siftbig_query.bvecs");
			vquery.n=10000;
			vquery.d=128;
			vquery.mat= fmat_new (vquery.d, vquery.n);
		}
		else if(strcmp(dataset, "gist")==0){
			strcpy (fquery,"/scratch/04596/tg838951/gist_query.fvecs");
			vquery.n=1000;
			vquery.d=960;
			vquery.mat= fmat_new (vquery.d, vquery.n);
		}
		if(strcmp(dataset, "siftbig")!=0){
			fvecs_read (fquery, vquery.d, vquery.n, vquery.mat);
		}
		else{
			b2fvecs_read (fquery, vquery.d, vquery.n, vquery.mat);
		}
		free(fquery);
	}
	return vquery;
}

mat pq_test_load_base(char* dataset, int offset, int my_rank){

	mat vbase;
	char srank[4];

	sprintf (srank, "%d",my_rank);

	if(strcmp(dataset, "random")==0){

		vbase.n=1000000;
		vbase.d=16;
		vbase.mat= fmat_new (vbase.d, vbase.n);

		srand (time(NULL));

		///inicializa com valores aleatórios

		load_random(vbase.mat, vbase.n, vbase.d);
	}
	else {
		char *fbase;

		fbase= (char*) malloc(sizeof(char)*51);

		if(strcmp(dataset, "siftsmall")==0){
			strcpy (fbase,"../siftsmall/siftsmall_base.fvecs");
			vbase.n=1000;
			vbase.d=128;
			vbase.mat= fmat_new (vbase.d, vbase.n);
		}
		else if(strcmp(dataset, "sift")==0){
			strcpy (fbase,"../sift/sift_base.fvecs");
			vbase.n=1000000;
			vbase.d=128;
			vbase.mat= fmat_new (vbase.d, vbase.n);
		}
		else if(strcmp(dataset, "siftbig")==0 ){
			strcpy (fbase,"/scratch/04596/tg838951/siftbig_base.bvecs");
			vbase.n=1000000;
			vbase.d=128;
			vbase.mat= (float*) malloc(sizeof(float)*vbase.d*vbase.n);
		}
		else if(strcmp(dataset, "gist")==0){
			strcpy (fbase,"/scratch/04596/tg838951/gist_base.fvecs");
			vbase.n=1000000;
			vbase.d=960;
			vbase.mat= fmat_new (vbase.d, vbase.n);
		}
		strcat(fbase, srank);
		if(strcmp(dataset, "siftbig")!=0){
			fvecs_read (fbase, vbase.d, vbase.n, vbase.mat);
		}
		else{
			my_bvecs_read (offset, fbase, vbase.d, vbase.n, vbase.mat);
		}
		free(fbase);
	}
	return vbase;
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
				perror ("ivecs_read error 1");
				fclose(f);
				return -1;
			}
		}

		if (new_d != d) {
			fprintf (stderr, "ivecs_read error 2: unexpected vector dimension\n");
			fclose(f);
			return -1;
		}

		if (fread (a + d * (long) i, sizeof (int), d, f) != d) {
			fprintf (stderr, "ivecs_read error 3\n");
			fclose(f);
			return -1;
		}
	}
	fclose (f);

	return i;
}

int my_bvecs_read (int offset, const char *fname, int d, int n, float *a){
	FILE *f = fopen (fname, "r");

	if (!f) {
		fprintf (stderr, "my_bvecs_read: could not open %s\n", fname);
		perror ("");
		return -1;
	}

	unsigned long long b;
	b=(unsigned long long) offset*n*(d+4);

	fseek (f, b, SEEK_SET);
	long i;
	
	for (i = 0; i < n; i++) {
		int new_d;

		if (fread (&new_d, sizeof (int), 1, f) != 1) {
			if (feof (f))break;
			else {
				perror ("my_bvecs_read error 1");
				fclose(f);
				return -1;
			}
		}

		if (new_d != d) {
			printf("d%dnd%d",d,new_d);
			fprintf (stderr, "my_bvecs_read error 2: unexpected vector dimension\n");
			fclose(f);
			return -1;
		}

		unsigned char * vb = (unsigned char *) malloc (sizeof (*vb) * d);

		if (fread (vb, sizeof (*vb), d, f) != d) {
			fprintf (stderr, "my_bvecs_read error 3\n");
			fclose(f);
			return -1;
		}

		for (int j = 0 ; j < d ; j++){
    		a[i*d+j] = vb[j];
		}
  		free (vb);
	}
	fclose (f);

	return i;
}
