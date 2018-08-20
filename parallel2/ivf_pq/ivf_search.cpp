#include "ivf_search.h"

#include <sys/time.h>
#include <set>
#include <cstdio>
#include <queue>
#include <ctime>

#include "mycuda.h"

#include <cmath>

#include "pqueue.h"

#include "debug.h"


static sem_t sem;


//TODO: dont try this at home
time_t start, end;
double micro;


struct timeval tv;
struct timeval start_tv;
struct timeval old_start_tv;

#ifdef DEBUG
#define sw(call)  old_start_tv = start_tv; \
				  gettimeofday(&start_tv, NULL); \
				  call ; \
				  gettimeofday(&tv, NULL); \
				  micro = (tv.tv_sec - start_tv.tv_sec) * 1000000 + (tv.tv_usec - start_tv.tv_usec); \
				  printf ("Elapsed: %.2lf seconds on call: %s\n", micro / 1000000, #call); \
				  start_tv = old_start_tv;
#else
#define sw(call) call;
#endif

struct timeval total_tv;
struct timeval total_start_tv;

//TODO: refactor variable names, they are terrible
//TODO: comment the code
//TODO: think about how to parallelize the other stages and/or if it would be worthwhile

float dist2(float* a, float* b, int size) {
	float d = 0;
	for (int i = 0; i < size; i++) {
		float diff = a[i] - b[i];
		d += diff * diff;
	}

	return d;
}

//TODO: we dont need k, maybe we shouldnt require that core_cpu and core_gpu have the same interface. Or maybe we should create some sort of structure that represents the context info
//TODO: receive the number of threads from outside sources
void core_cpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* rid_to_ivf, int* qid_to_starting_outid, matI idxs, mat dists, int k, int w) {
	const int threads = 8;
	debug("Executing %d threads", threads);

	#pragma omp parallel num_threads(threads)
	{
		int nthreads = omp_get_num_threads();
		int tid = omp_get_thread_num();

		for (int qid = tid; qid < residual.n / w; qid += nthreads) {
			int rid = qid * w;
			
			std::pair<float, int> heap[k];
			pqueue pq(heap, k);
			
			for (int i = 0; i < w; i++, rid++) {
				float* current_residual = residual.mat + rid * PQ.nsq * PQ.ds;

				mat distab;
				distab.mat = new float[PQ.ks * PQ.nsq];
				distab.n = PQ.nsq;
				distab.d = PQ.ks;

				for (int d = 0; d < PQ.nsq; d++) {
					for (int k = 0; k < PQ.ks; k++) {
						float dist = dist2(current_residual + d * PQ.ds, PQ.centroids + d * PQ.ks * PQ.ds + k * PQ.ds, PQ.ds);
						distab.mat[PQ.ks * d + k] = dist;
					}
				}

				int ivf_id = rid_to_ivf[rid];
				ivf_t entry = ivf[ivf_id];


				for (int j = 0; j < entry.idstam; j++) {
					float dist = 0;

					for (int d = 0; d < PQ.nsq; d++) {
						dist += distab.mat[PQ.ks * d + entry.codes.mat[PQ.nsq * j + d]];
					}

					pq.add(dist, entry.ids[j]);
				}

				delete[] distab.mat;
			}
			
			int outid = qid_to_starting_outid[qid];
			
			for (int i = 0; i < pq.size; i++) {
				dists.mat[outid] = pq.base[i].first;
				idxs.mat[outid] = pq.base[i].second;
				outid++;
			}
		}
	}
}

//TODO: try to refactor the code so that we don't need to have these "partial" arrays, which are basically a copy of the full array
void do_on(void (*target)(pqtipo, mat, ivf_t*, int, int*, int*, matI, mat, int, int),
		ivfpq_t PQ, std::list<int>& queries, mat residual, int* coaidx, ivf_t* ivf, query_id_t*& elements, matI& idxs, mat& dists, int k, int w) {
	if (queries.size() == 0) return;
	
	elements = new query_id_t[queries.size()];

	int qid_to_starting_outid[queries.size()];
	int rid = 0;
	int outid = 0;
	for (int qid = 0; qid < queries.size(); qid++) {
		int numImgs = 0;

		for (int i = 0; i < w; i++, rid++) {
			numImgs += ivf[coaidx[rid]].idstam;
		}

		int size = min(numImgs, k);
		elements[qid].id = qid;
		elements[qid].tam = size;
		qid_to_starting_outid[qid] = outid;
		outid += size;
	}

	idxs.mat = new int[outid];
	idxs.n = outid;
	dists.mat = new float[outid];
	dists.n = outid;


	sw((*target)(PQ.pq, residual, ivf, PQ.coa_centroidsn, coaidx, qid_to_starting_outid, idxs, dists, k, w));
}


void do_cpu(ivfpq_t PQ, std::list<int>& to_cpu, mat residual, int* coaidx, ivf_t* ivf, query_id_t*& elements, matI& idxs, mat& dists, int k, int w) {
	do_on(&core_cpu, PQ, to_cpu, residual, coaidx, ivf, elements, idxs, dists, k, w);
}

void do_gpu(ivfpq_t PQ, std::list<int>& to_gpu, mat residual, int* coaidx, ivf_t* ivf,  query_id_t*& elements, matI& idxs, mat& dists, int k, int w) {
	do_on(&core_gpu, PQ, to_gpu, residual, coaidx, ivf, elements, idxs, dists, k, w);
}

void send_results(int nqueries, query_id_t* elements, matI idxs, mat dists, int aggregator_id) {
	int counter = 0;

	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	MPI_Send(&my_rank, 1, MPI_INT, aggregator_id, 1, MPI_COMM_WORLD);
	MPI_Send(&nqueries, 1, MPI_INT, aggregator_id, 0, MPI_COMM_WORLD);
	MPI_Send(elements, sizeof(query_id_t) * nqueries, MPI_BYTE, aggregator_id, 0, MPI_COMM_WORLD);

	MPI_Send(idxs.mat, idxs.n, MPI_INT, aggregator_id, 0, MPI_COMM_WORLD);
	MPI_Send(dists.mat, dists.n, MPI_FLOAT, aggregator_id, 0, MPI_COMM_WORLD);
}



void parallel_search (int nsq, int k, int threads, int tam, int aggregator_id, MPI_Comm search_comm, char *dataset, int w, char* train_path, char* ivf_path, bool gpu){
	mat residual;
	int *coaidx, my_rank;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	char* header;
	asprintf(&header, "%s/header", train_path);
	char* cent;
	asprintf(&cent, "%s/pq_centroids", train_path);
	char* coa;
	asprintf(&coa, "%s/coa_centroids", train_path);

	ivfpq_t ivfpq;
	read_cent(header, cent, coa, &ivfpq);
	
	std::cout << "number of coarse centroids: " << ivfpq.coa_centroidsn << "\n";
	std::cout << "number of product centroids per dimension: " << ivfpq.pq.centroidsn << "\n";
	std::cout << "number of product centroids dimensions: " << ivfpq.pq.centroidsd << "\n";


	ivf_t *ivf, *ivf2;
	ivf = read_ivf(ivfpq, tam, my_rank, ivf_path);

	float **dis;
	int **ids;

	int count = 0;


	int base_id = 0; // corresponds to the query_id

	MPI_Barrier(search_comm);

	sem_init(&sem, 0, 1);
	
	preallocate_gpu_mem(ivfpq.pq, ivf, ivfpq.coa_centroidsn);

	

//
	MPI_Bcast(&residual.n, 1, MPI_INT, 0, search_comm);
	MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);
//
	residual.mat = (float*) malloc(sizeof(float) * residual.n * residual.d);

	MPI_Bcast(&residual.mat[0], residual.d * residual.n, MPI_FLOAT, 0, search_comm);

	coaidx = (int*) malloc(sizeof(int) * residual.n);

	MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);
	
	gettimeofday(&total_start_tv, NULL);

	dis = (float**) malloc(sizeof(float *) * (residual.n / w));
	ids = (int**) malloc(sizeof(int *) * (residual.n / w));

	std::list<int> to_gpu;
	std::list<int> to_cpu;

	debug("residual.n=%d", residual.n);

	for (int qid = 0; qid < residual.n / w; qid++) {
		if (gpu) to_gpu.push_back(qid);
		else to_cpu.push_back(qid);
	}

	debug("EXECUTING ON THE %s", to_cpu.size() == 0 ? "gpu" : "cpu");

	time_t start, end;
	time(&start);

	debug("PQ.ks=%d and k=%d", ivfpq.pq.ks, k);

	//GPU PART
	query_id_t* elements;
	matI idxs;
	mat dists;

	if (to_gpu.size() != 0) {
		sw(do_gpu(ivfpq, to_gpu, residual, coaidx, ivf, elements, idxs, dists, k, w));
	} else {
		sw(do_cpu(ivfpq, to_cpu, residual, coaidx, ivf, elements, idxs, dists, k, w));
	}

	debug("Before sending results");
	sw(send_results(to_cpu.size() + to_gpu.size(), elements, idxs, dists, aggregator_id));
	debug("after sending results");

	delete[] elements;
	delete[] idxs.mat;
	delete[] dists.mat;

	base_id += residual.n / w;


	gettimeofday(&total_tv, NULL);
	micro = (total_tv.tv_sec - total_start_tv.tv_sec) * 1000000 + (total_tv.tv_usec - total_start_tv.tv_usec); \
	printf("\nElapsed: %.2lf seconds on TOTAL\n", micro / 1000000);

	deallocate_gpu_mem();
	
	debug("GOT OUT OF HERE");
	
	sem_destroy(&sem);
	free(ivf);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);

	debug("FINISHED THE SEARCH");
}


//TODO: make these paths available in a config file
//TODO: make the decision of wheter reading or writing the IVF a runtime option
//TODO: currently it assumes that we have only one search node. NEED TO FIX THIS ASAP
ivf_t* read_ivf(ivfpq_t ivfpq, int tam, int my_rank, char* ivf_path){
	FILE *fp;
	char name_arq[100];

	ivf_t* ivf = (ivf_t*) malloc(sizeof(ivf_t) * ivfpq.coarsek);

	for(int i = 0; i < ivfpq.coarsek; i++) {
		sprintf(name_arq, "%s/%d", ivf_path, i);

		fp = fopen(name_arq,"rb");

		int coarsek;
		fread(&coarsek, sizeof(int), 1, fp);
		
		fread(&ivf[i].idstam, sizeof(int), 1, fp);
		ivf[i].ids = (int*) malloc(sizeof(int) * ivf[i].idstam);
		fread(&ivf[i].ids[0], sizeof(int), ivf[i].idstam, fp);


		fread(&ivf[i].codes.n, sizeof(int), 1, fp);
		fread(&ivf[i].codes.d, sizeof(int), 1, fp);
		
		ivf[i].codes.mat = (int*) malloc(sizeof(int) * ivf[i].codes.n * ivf[i].codes.d);
		fread(&ivf[i].codes.mat[0], sizeof(int), ivf[i].codes.n * ivf[i].codes.d, fp);
		fclose(fp);
	}

	return ivf;
}

int min(int a, int b){
	if(a>b){
		return b;
	}
	else
		return a;
}
