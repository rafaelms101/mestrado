#include <boost/filesystem.hpp>
#include "pq-utils/pq_test_load_vectors.h"
#include "ivf_pq/ivf_training.h"
#include <mpi.h>

void write_ivf(ivfpq_t ivfpq, int threads, int tam, int my_rank, int nsq, char* dataset, char* ivf_path);

int main(int argc, char **argv) {
	int provided;
	int my_rank;

	MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	std::printf("rank: %d\n", my_rank);

	
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
	write_ivf(ivfpq, threads, tam, my_rank, nsq, dataset, ivf_path);
	
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	
	free(ivf_path);
	free(train_path);
	free(header);
	free(cent);
	free(coa);
	
	MPI_Finalize();
}

void write_ivf(ivfpq_t ivfpq, int threads, int tam, int my_rank, int nsq, char* dataset, char* ivf_path) {
	char filename[100];

	int lim = tam / 1000000;
	if (tam % 1000000 != 0) lim++;

	//Cria a lista invertida correspondente ao trecho da base assinalado a esse processo
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
	for (int i = 0; i < lim; i++) {
		ivf_t* ivf = (ivf_t*) malloc(sizeof(ivf_t) * ivfpq.coarsek);
		mat vbase = pq_test_load_base(dataset, i, tam);

		ivfpq_assign(ivfpq, vbase, ivf);

		free(vbase.mat);

		for (int j = 0; j < ivfpq.coarsek; j++) {
			for (int l = 0; l < ivf[j].idstam; l++)
				ivf[j].ids[l] += 1000000 * i + tam * my_rank;

			#pragma omp critical
			{
				sprintf(filename, "%s/%d", ivf_path, j);
				FILE* fp = fopen(filename, "ab");
				fwrite(&ivfpq.coarsek, sizeof(int), 1, fp);
				fwrite(&ivf[j].idstam, sizeof(int), 1, fp);
				fwrite(&ivf[j].ids[0], sizeof(int), ivf[j].idstam, fp);
				fwrite(&ivf[j].codes.n, sizeof(int), 1, fp);
				fwrite(&ivf[j].codes.d, sizeof(int), 1, fp);
				fwrite(&ivf[j].codes.mat[0], sizeof(int), ivf[j].codes.n * ivf[j].codes.d, fp);
				fclose(fp);
			}

			free(ivf[j].ids);
			free(ivf[j].codes.mat);
		}
		free(ivf);
	}
}
