#include <boost/filesystem.hpp>
#include "pq-utils/pq_test_load_vectors.h"
#include "ivf_pq/ivf_training.h"
#include <mpi.h>

void write_ivf(ivfpq_t ivfpq, int threads, int tam, int nsq, char* dataset, char* ivf_path);

int main(int argc, char **argv) {
	if (argc != 6) {
		std::cout
				<< "Usage: genivf <database> <tam> <coarsek> <nsq> <threads>"
				<< endl;
		return -1;
	}

	char* dataset = argv[1];
	int tam = atoi(argv[2]);
	int coarsek = atoi(argv[3]);
	int nsq = atoi(argv[4]);
	int threads = atoi(argv[5]);

	char* ivf_path;
	asprintf(&ivf_path, "%s/%s/ivf/%d_%d_%d", BASE_DIR, dataset, tam, coarsek, nsq);
	
	if (boost::filesystem::exists(ivf_path)) {
		boost::filesystem::remove_all(ivf_path);
	}
	
	boost::filesystem::create_directories(ivf_path);

	char* train_path;
	asprintf(&train_path, "%s/%s/train/%d_%d_%d", BASE_DIR, dataset, tam, coarsek, nsq);

	if (! boost::filesystem::exists(train_path)) {
		std::printf("You have to train first\n");
		exit(-1);
	}
	
	char* header;
	asprintf(&header, "%s/header", train_path);
	char* cent;
	asprintf(&cent, "%s/pq_centroids", train_path);
	char* coa;
	asprintf(&coa, "%s/coa_centroids", train_path);

	ivfpq_t ivfpq;
	read_cent(header, cent, coa, &ivfpq);
	write_ivf(ivfpq, threads, tam, nsq, dataset, ivf_path);
	
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	
	free(ivf_path);
	free(train_path);
	free(header);
	free(cent);
	free(coa);
}

//TODO: using a limit of 1.000.000 while loading all the data in parallel doesn't make sense if the objective was to put an upper limit on memory usage
//TODO: redo this when I need to load more memory than we have available
void write_ivf(ivfpq_t ivfpq, int threads, int tam, int nsq, char* dataset, char* ivf_path) {
	char filename[100];
	
	ivf_t* ivf = (ivf_t*) malloc(sizeof(ivf_t) * ivfpq.coarsek);
	mat vbase = pq_test_load_base(dataset, tam);
	ivfpq_assign(ivfpq, vbase, ivf);
	free(vbase.mat);
	
	for (int j = 0; j < ivfpq.coarsek; j++) {
		sprintf(filename, "%s/%d", ivf_path, j);
		FILE* fp = fopen(filename, "ab");
		fwrite(&ivfpq.coarsek, sizeof(int), 1, fp);
		fwrite(&ivf[j].idstam, sizeof(int), 1, fp);
		fwrite(&ivf[j].ids[0], sizeof(int), ivf[j].idstam, fp);
		fwrite(&ivf[j].codes.n, sizeof(int), 1, fp);
		fwrite(&ivf[j].codes.d, sizeof(int), 1, fp);
		fwrite(&ivf[j].codes.mat[0], sizeof(int), ivf[j].codes.n * ivf[j].codes.d, fp);
		fclose(fp);
		free(ivf[j].ids);
		free(ivf[j].codes.mat);
	}
	
	
	free(ivf);
}
