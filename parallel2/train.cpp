#include <boost/filesystem.hpp>
#include "pq-utils/pq_test_load_vectors.h"
#include "ivf_pq/ivf_training.h"

int main(int argc, char **argv) {
	if (argc != 6) {
		std::cout
				<< "Usage: train <database> <tam> <coarsek> <nsq> <threads>"
				<< endl;
		return -1;
	}

	char* dataset = argv[1];
	int tam = atoi(argv[2]);
	int coarsek = atoi(argv[3]);
	int nsq = atoi(argv[4]);
	int threads = atoi(argv[5]);

	char* path;
	asprintf(&path, "%s/%s/train/%d_%d", BASE_DIR, dataset, coarsek, nsq);

	if (boost::filesystem::exists(path)) return 0;
	boost::filesystem::create_directories(path);
	
	char* header;
	asprintf(&header, "%s/header", path);
	char* cent;
	asprintf(&cent, "%s/pq_centroids", path);
	char* coa;
	asprintf(&coa, "%s/coa_centroids", path);

	// Cria os centroides baseado em uma base de treinamento e os armazena em arquivos
	mat vtrain = pq_test_load_train(dataset, tam);
	ivfpq_t ivfpq = ivfpq_new(coarsek, nsq, vtrain, threads);

	write_cent(header, cent, coa, ivfpq);

	free(vtrain.mat);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	
	free(path);
	free(header);
	free(cent);
	free(coa);
}
