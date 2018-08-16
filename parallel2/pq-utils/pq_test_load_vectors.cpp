#include "pq_test_load_vectors.h"

#include "../ivf_pq/debug.h"

mat pq_test_load_train(char* dataset, int tam){
	mat vtrain;
	vtrain.n = tam / 10;
	vtrain.d = 128;
	vtrain.mat = fmat_new(vtrain.d, vtrain.n);
	
	char ftrain[100];
	strcpy(ftrain, BASE_DIR);
	strcat(ftrain, "/");
	strcat(ftrain, dataset);
	strcat(ftrain,"/learn");
	
	
	if(! strcmp(dataset, "siftbig")){
		b2fvecs_read (ftrain, vtrain.d, vtrain.n, vtrain.mat);
	}
	else {
		fvecs_read (ftrain, vtrain.d, vtrain.n, vtrain.mat);
	}
	
	return vtrain;
}

matI pq_test_load_gdn(char* dataset, int tam, int nqueries){
	matI vids_gnd;
	int *ids_gnd;
	char fgroundtruth[100];

	strcpy(fgroundtruth, BASE_DIR);
	strcat(fgroundtruth, "/");
	strcat(fgroundtruth, dataset);
	strcat(fgroundtruth,"/");
	
	if (! strcmp(dataset, "siftbig")) {
		if (tam == 1000000)
			strcat(fgroundtruth, "gnd/idx_1M.ivecs");
		else if (tam == 2000000)
			strcat(fgroundtruth, "gnd/idx_2M.ivecs");
		else if (tam == 5000000)
			strcat(fgroundtruth, "gnd/idx_5M.ivecs");
		else if (tam == 10000000)
			strcat(fgroundtruth, "gnd/idx_10M.ivecs");
		else if (tam == 20000000)
			strcat(fgroundtruth, "gnd/idx_20M.ivecs");
		else if (tam == 50000000)
			strcat(fgroundtruth, "gnd/idx_50M.ivecs");
		else if (tam == 100000000)
			strcat(fgroundtruth, "gnd/idx_100M.ivecs");
		else if (tam == 200000000)
			strcat(fgroundtruth, "gnd/idx_200M.ivecs");
		else if (tam == 500000000)
			strcat(fgroundtruth, "gnd/idx_500M.ivecs");
		else if (tam == 1000000000)
			strcat(fgroundtruth, "gnd/idx_1000M.ivecs");
		else {
			std::printf("Wrong database size supplied\n");
			exit(0);
		}
		
		vids_gnd.d = 1000;
	} else {
		strcat(fgroundtruth,"gnd.ivecs");
		vids_gnd.d = 100;	
	}

	vids_gnd.n = nqueries;
	ids_gnd = ivec_new(vids_gnd.n * vids_gnd.d);
	ivecs_read(fgroundtruth, vids_gnd.d, vids_gnd.n, ids_gnd);
	vids_gnd.mat = ivec_new (vids_gnd.n);
	
	for(int i=0; i < vids_gnd.n; i++){
		vids_gnd.mat[i] = ids_gnd[i * vids_gnd.d];
	}
	
	vids_gnd.d = 1;
	
	free(ids_gnd);
	return vids_gnd;
}

mat pq_test_load_query(char* dataset, int threads, int nqueries){
	mat vquery;
	char srank[4];
	char fquery[100];
	
	strcpy(fquery, BASE_DIR);
	strcat(fquery, "/");
	strcat(fquery, dataset);
	strcat(fquery, "/query");
	vquery.n = nqueries;
	vquery.d = 128;
	vquery.mat = fmat_new(vquery.d, vquery.n);
	
	if (! strcmp(dataset, "siftbig")) b2fvecs_read(fquery, vquery.d, vquery.n, vquery.mat);
	else fvecs_read(fquery, vquery.d, vquery.n, vquery.mat);


	return vquery;
}

mat pq_test_load_base(char* dataset, int offset, int tam){
	mat vbase;
	char fbase[100];
	strcpy(fbase, BASE_DIR);
	strcat(fbase, "/");
	strcat(fbase, dataset);
	strcat(fbase, "/base");
	vbase.n = tam;
	vbase.d = 128;
	

	if (! strcmp(dataset, "siftbig")) {
		vbase.mat = (float*) malloc(sizeof(float) * vbase.d * vbase.n);
		my_bvecs_read(offset, fbase, vbase.d, vbase.n, vbase.mat);
	} else { 
		vbase.mat= fmat_new (vbase.d, vbase.n);
		fvecs_read(fbase, vbase.d, vbase.n, vbase.mat);
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
