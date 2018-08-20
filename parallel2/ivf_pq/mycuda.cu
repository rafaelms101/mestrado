#include <cuda_runtime.h>

#include "debug.h"
#include "mycuda.h"
#include "topk.cu"

#include "helper_cuda.h"

#include <cstdio>

#define safe_call(call) if (cudaSuccess != call) { err = cudaGetLastError(); \
													fprintf(stderr, "Failed call: %s\nError: %s\n", \
															#call, cudaGetErrorString(err)); \
													exit(EXIT_FAILURE); }


extern __shared__ char shared_memory[];

#define ACTIVE_BLOCKS 32
//
static pqtipo gpu_PQ;
static ivf_t* gpu_ivf;


cudaError_t err = cudaSuccess; //TODO: find a way to not have a global

//TODO: preload PQ.centroids and ivf
__global__ void compute_dists(const pqtipo PQ, ivf_t* ivf, const mat residual, 
		const int* rid_to_ivf, const int* const qid_to_starting_outid,
		Img* distance_buffer, int block_buffer_size, matI idxs, mat dists, const int k, const int w) {
	auto tid = threadIdx.x;
	auto nthreads = blockDim.x;
	auto bid = blockIdx.x;
	auto numBlocks = gridDim.x;
	
	float* distab = (float*) shared_memory;
	
	for (int qid = bid; qid < residual.n / w; qid += numBlocks) {
		Img* block_input = distance_buffer + bid * block_buffer_size;
		Img* current_block_input = block_input;
		int numDists = 0;
		
		for (int current_w = 0; current_w < w; current_w++) {
			//computing disttab
			int rid = qid * w + current_w;
			float* current_residual = residual.mat + rid * PQ.nsq * PQ.ds;

			int step_size = (PQ.ks * PQ.nsq + nthreads - 1) / nthreads;

			int begin_i = tid * step_size;
			int end_i = min(begin_i + step_size, PQ.ks * PQ.nsq) - 1;


			for (int i = begin_i; i <= end_i; i++) {
				float* centroid = PQ.centroids + i * PQ.ds;
				int d = i / PQ.ks;

				float* sub_residual = current_residual + d * PQ.ds;
				float dist = 0;

				for (int j = 0; j < PQ.ds; j++) {
					float diff = sub_residual[j] - centroid[j];
					dist += diff * diff;
				}

				distab[i] = dist;
			}

			__syncthreads();

			//computing the distances to the vectors
			ivf_t entry = ivf[rid_to_ivf[rid]];
			

			for (int i = tid; i < entry.idstam; i += nthreads) {
				float dist = 0;

				for (int s = 0; s < PQ.nsq; s++) {
					dist += distab[PQ.ks * s + entry.codes.mat[PQ.nsq * i + s]];
				}

				current_block_input[i].dist = dist;
				current_block_input[i].imgid = entry.ids[i];
			}
			
			current_block_input = current_block_input + entry.idstam;
			numDists += entry.idstam;
			
			__syncthreads();
		}
		

		//choosing the top k
		// selecting num_heaps
		auto shared_memory_size = 48 << 10; //TODO: there might be some function to obtain the shared memory size from the environment
		auto heap_size = k * sizeof(Entry<Img>);
		auto max_heaps = shared_memory_size / heap_size;
		auto num_subheaps = max_heaps - 1;

		if (num_subheaps > blockDim.x) num_subheaps = blockDim.x;

		if (num_subheaps * 2 * k > numDists) {
			num_subheaps = numDists / (2 * k);
		}

		if (num_subheaps == 0) num_subheaps = 1;

		topk(qid, num_subheaps, k, block_input, numDists, qid_to_starting_outid, dists.mat, idxs.mat);

		__syncthreads();
	}
}

cudaError_t alloc(void **devPtr, size_t size) {
	return cudaMalloc(devPtr, size);
}


void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* rid_to_ivf, int* qid_to_starting_outid, matI idxs, mat dists, int k, int w) {
	//TODO: implement / redo the error handling so that we have less code duplication
	mat gpu_residual = residual;
	debug("Allocating %d MB for residuals\n",  sizeof(float) * residual.n * residual.d / 1024 / 1024);
	safe_call(alloc((void **) &gpu_residual.mat, sizeof(float) * residual.n * residual.d));
	safe_call(cudaMemcpy(gpu_residual.mat, residual.mat, sizeof(float) * residual.n * residual.d, cudaMemcpyHostToDevice));


	int biggest_idstam = 0;
	for (int i = 0; i < ivf_size; i++) {
		if (ivf[i].idstam > biggest_idstam) biggest_idstam = ivf[i].idstam;
	}

	int* gpu_rid_to_ivf;
	debug("Allocating %d MB for rid_to_ivf\n",  sizeof(int) * residual.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_rid_to_ivf, sizeof(int) * residual.n));
	safe_call(cudaMemcpy(gpu_rid_to_ivf, rid_to_ivf, sizeof(int) * residual.n, cudaMemcpyHostToDevice));


	matI gpu_idxs = idxs;
	debug("Allocating %d MB for idxs\n", sizeof(int) * idxs.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_idxs.mat, sizeof(int) * idxs.n));

	mat gpu_dists = dists;
	debug("Allocating %d MB for dists\n", sizeof(float) * dists.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_dists.mat, sizeof(float) * dists.n));

	//allocating the input buffer
	int* gpu_qid_to_starting_outid;
	debug("Allocating %d MB for qid_to_starting_outid\n", sizeof(int) * residual.n / w / 1024 / 1024);
	safe_call(alloc((void **) &gpu_qid_to_starting_outid, sizeof(int) * residual.n / w));
	safe_call(cudaMemcpy(gpu_qid_to_starting_outid, qid_to_starting_outid, sizeof(int) * residual.n / w, cudaMemcpyHostToDevice));

	debug("Allocating %d MB for the distance buffer\n",  sizeof(Img) * biggest_idstam * w * ACTIVE_BLOCKS / 1024 / 1024);
	Img* gpu_distance_buffer;
	safe_call(alloc((void **) &gpu_distance_buffer, sizeof(Img) * biggest_idstam * w * ACTIVE_BLOCKS)); //TODO: its possible to save some memory if I compute the biggest AGGREGATED idstam (as in, the sum of all w idstam)


	dim3 block(1024, 1, 1);
	dim3 grid(ACTIVE_BLOCKS, 1, 1);

	int sm_size = 48 << 10;

	debug("Trying to allocate: %dKB in shared memory\n", 48 << 10 / 1024);
	debug("distab needs: %dKB in shared memory\n",  PQ.ks * PQ.nsq * sizeof(float) / 1024);

	compute_dists<<<grid, block,  sm_size>>>(gpu_PQ, gpu_ivf, gpu_residual, gpu_rid_to_ivf, gpu_qid_to_starting_outid, gpu_distance_buffer, biggest_idstam * w, gpu_idxs, gpu_dists, k, w);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch compute_dists kernel.\nError: %s\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	} 

	debug("After calling the kernel\n");


	safe_call(cudaMemcpy(idxs.mat, gpu_idxs.mat , sizeof(int) * idxs.n, cudaMemcpyDeviceToHost));
	safe_call(cudaMemcpy(dists.mat, gpu_dists.mat, sizeof(float) * dists.n, cudaMemcpyDeviceToHost));

	//FREEING MEMORY
	cudaFree(gpu_residual.mat);
	cudaFree(gpu_rid_to_ivf);
	cudaFree(gpu_idxs.mat);
	cudaFree(gpu_dists.mat);
	cudaFree(gpu_qid_to_starting_outid);
	cudaFree(gpu_distance_buffer);
}

ivf_t* tmp_ivf;
int tmp_size;

void preallocate_gpu_mem(pqtipo host_PQ, ivf_t* host_ivf, int ivf_size) {
	long ivf_mem_size = 0;
	safe_call(alloc((void **) &gpu_ivf, sizeof(ivf_t) * ivf_size));	
	
	ivf_mem_size += sizeof(ivf_t) * ivf_size;
	tmp_ivf = new ivf_t[ivf_size];
	tmp_size = ivf_size;
	
	for (int i = 0; i < ivf_size; i++) {
		tmp_ivf[i].idstam = host_ivf[i].idstam;
		tmp_ivf[i].codes = host_ivf[i].codes;

		ivf_mem_size += sizeof(int) * tmp_ivf[i].idstam;
		safe_call(alloc((void ** ) &tmp_ivf[i].ids, sizeof(int) * tmp_ivf[i].idstam));
		safe_call(cudaMemcpy(tmp_ivf[i].ids, host_ivf[i].ids, sizeof(int) * host_ivf[i].idstam, cudaMemcpyHostToDevice));
		ivf_mem_size += sizeof(int) * tmp_ivf[i].codes.n * tmp_ivf[i].codes.d;
		safe_call(alloc((void ** ) &tmp_ivf[i].codes.mat, sizeof(int) * tmp_ivf[i].codes.n * tmp_ivf[i].codes.d));
		safe_call(cudaMemcpy(tmp_ivf[i].codes.mat, host_ivf[i].codes.mat, sizeof(int) * host_ivf[i].codes.n * host_ivf[i].codes.d, cudaMemcpyHostToDevice));
	}

	debug("Allocating %d MB for IVF\n",  ivf_mem_size / 1024 / 1024);
	safe_call(cudaMemcpy(gpu_ivf, tmp_ivf, sizeof(ivf_t) * ivf_size, cudaMemcpyHostToDevice));
	
	//centroids
	gpu_PQ = host_PQ;
	debug("Allocating %d MB for centroids\n",  sizeof(float) * host_PQ.centroidsd * host_PQ.centroidsn / 1024 / 1024);
	safe_call(alloc((void **) &gpu_PQ.centroids, sizeof(float) * host_PQ.centroidsd * host_PQ.centroidsn));
	safe_call(cudaMemcpy(gpu_PQ.centroids, host_PQ.centroids, sizeof(float) * host_PQ.centroidsd * host_PQ.centroidsn, cudaMemcpyHostToDevice));
}

void deallocate_gpu_mem() {
	for (int i = 0; i < tmp_size; i++) {
		cudaFree(tmp_ivf[i].ids);
		cudaFree(tmp_ivf[i].codes.mat);
		
	}
	
	cudaFree(gpu_ivf);
	cudaFree(gpu_PQ.centroids);
	
	delete tmp_ivf;
}
