#include <cuda_runtime.h>

#include "mycuda.h"
#include "topk.cu"

#include "helper_cuda.h"

#include <cstdio>

#define safe_call(call) if (cudaSuccess != call) { err = cudaGetLastError(); \
													fprintf(stderr, "Failed call: %s\nError: %s\n", \
															#call, cudaGetErrorString(err)); \
													exit(EXIT_FAILURE); }


extern __shared__ char shared_memory[];

#define ACTIVE_BLOCKS 10


//TODO: make the merge of the w query results in the GPU (?)
__global__ void compute_dists(pqtipo PQ, mat residual, ivf_t* ivf,
		int* entry_map, int* starting_imgid, int* starting_inputid,
		Img* full_input, matI idxs, mat dists, int k) {
	int tid = threadIdx.x;
	int nthreads = blockDim.x;
	int bid = blockIdx.x;
	int numBlocks = gridDim.x;

	float* distab = (float*) shared_memory;

	for (int qid = bid; qid < residual.n; qid += numBlocks) {
		//computing disttab
		float* current_residual = residual.mat + qid * PQ.nsq * PQ.ds;
		int step_size = (PQ.ks * PQ.nsq + nthreads - 1) / nthreads;

		int begin_i = tid * step_size;
		int end_i = min(begin_i + step_size, PQ.ks * PQ.nsq) - 1;
		float* centroid = PQ.centroids + begin_i * PQ.ds;

		for (int i = begin_i; i <= end_i; i++) {
			int d = i / PQ.ks;

			float* sub_residual = current_residual + d * PQ.ds;
			float dist = 0;

			for (int j = 0; j < PQ.ds; j++, centroid++) {
				float diff = sub_residual[j] - *centroid;
				dist += diff * diff;
			}

			distab[i] = dist;
		}

		__syncthreads();

		//computing the distances to the vectors
		ivf_t entry = ivf[entry_map[qid]];
		Img* input = full_input + starting_inputid[qid];

		for (int i = tid; i < entry.idstam; i += nthreads) {
			float dist = 0;

			for (int s = 0; s < PQ.nsq; s++) {
				dist += distab[PQ.ks * s + entry.codes.mat[PQ.nsq * i + s]];
			}

			input[i] = {dist, entry.ids[i]};
		}

		__syncthreads();
		//choosing the top k


		//TODO: remember to analyze the case where size < k or size < 2k
		// selecting num_heaps
		auto shared_memory_size = 48 << 10; //TODO: there might be some function to obtain the shared memory size from the environment
		auto heap_size = k * sizeof(Entry<Img>);
		auto max_heaps = shared_memory_size / heap_size;
		auto num_subheaps = max_heaps - 1;

		if (num_subheaps > blockDim.x) num_subheaps = blockDim.x;

		if (num_subheaps * 2 * k > entry.idstam) {
			num_subheaps = entry.idstam / (2 * k);
		}

		if (num_subheaps == 0) num_subheaps = 1;

		topk(qid, num_subheaps, k, full_input, starting_inputid, dists.mat,
				idxs.mat);

		__syncthreads();
	}
}

cudaError_t alloc(void **devPtr, size_t size) {
	return cudaMalloc(devPtr, size);
}



void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* entry_map, int* starting_imgid,  int* starting_inputid,  int num_imgs, matI idxs, mat dists, int k) {
//	int numBlocks;
//	cudaOccupancyMaxActiveBlocksPerMultiprocessor (&numBlocks, compute_dists, 1024, 48 << 10);
//	std::printf("Maximum number of blocks: %d\n", numBlocks);


	//TODO: implement / redo the error handling so that we have less code duplication
	cudaError_t err = cudaSuccess;

	pqtipo gpu_PQ = PQ;

	std::printf("Allocating %d MB for centroids\n",  sizeof(float) * PQ.centroidsd * PQ.centroidsn / 1024 / 1024);
	safe_call(alloc((void **) &gpu_PQ.centroids, sizeof(float) * PQ.centroidsd * PQ.centroidsn));
	safe_call(cudaMemcpy(gpu_PQ.centroids, PQ.centroids, sizeof(float) * PQ.centroidsd * PQ.centroidsn, cudaMemcpyHostToDevice));

	mat gpu_residual = residual;
	std::printf("Allocating %d MB for residuals\n",  sizeof(float) * residual.n * residual.d / 1024 / 1024);
	safe_call(alloc((void **) &gpu_residual.mat, sizeof(float) * residual.n * residual.d));
	safe_call(cudaMemcpy(gpu_residual.mat, residual.mat, sizeof(float) * residual.n * residual.d, cudaMemcpyHostToDevice));


	long ivf_mem_size = 0;
	ivf_t* gpu_ivf;

	ivf_mem_size += sizeof(ivf_t) * ivf_size;
	std::printf("IVF memory size up to now: %d MB\n",  ivf_mem_size / 1024 / 1024);
	safe_call(alloc((void **) &gpu_ivf, sizeof(ivf_t) * ivf_size));


	ivf_t* tmp_ivf = new ivf_t[ivf_size];

	for (int i = 0; i < ivf_size; i++) {
		tmp_ivf[i].idstam = ivf[i].idstam;
		tmp_ivf[i].codes = ivf[i].codes;

		ivf_mem_size += sizeof(int) * tmp_ivf[i].idstam;
		std::printf("IVF memory size up to now: %d MB\n",  ivf_mem_size / 1024 / 1024);
		safe_call(alloc((void **) &tmp_ivf[i].ids, sizeof(int) * tmp_ivf[i].idstam));
		safe_call(cudaMemcpy(tmp_ivf[i].ids, ivf[i].ids, sizeof(int) * ivf[i].idstam, cudaMemcpyHostToDevice));

		ivf_mem_size += sizeof(int) * tmp_ivf[i].codes.n * tmp_ivf[i].codes.d;
		std::printf("IVF memory size up to now: %d MB\n",  ivf_mem_size / 1024 / 1024);
		safe_call(alloc((void **) &tmp_ivf[i].codes.mat, sizeof(int) * tmp_ivf[i].codes.n * tmp_ivf[i].codes.d));
		std::printf("entry=%d, idstam=%d, codes.n=%d, codes.d=%d\n", i, tmp_ivf[i].idstam, tmp_ivf[i].codes.n, tmp_ivf[i].codes.d);
		safe_call(cudaMemcpy(tmp_ivf[i].codes.mat, ivf[i].codes.mat, sizeof(int) * ivf[i].codes.n * ivf[i].codes.d, cudaMemcpyHostToDevice));
	}

	std::printf("Allocating %d MB for IVF\n",  ivf_mem_size / 1024 / 1024);
	safe_call(cudaMemcpy(gpu_ivf, tmp_ivf, sizeof(ivf_t) * ivf_size, cudaMemcpyHostToDevice));

	int* gpu_entry_map;
	std::printf("Allocating %d MB for entry map\n",  sizeof(int) * residual.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_entry_map, sizeof(int) * residual.n));
	safe_call(cudaMemcpy(gpu_entry_map, entry_map, sizeof(int) * residual.n, cudaMemcpyHostToDevice));

	int* gpu_starting_imgid;
	std::printf("Allocating %d MB for starting img id\n",  sizeof(int) * residual.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_starting_imgid, sizeof(int) * residual.n));
	safe_call(cudaMemcpy(gpu_starting_imgid, starting_imgid, sizeof(int) * residual.n, cudaMemcpyHostToDevice));

	//query_id_t* gpu_elements;
	//safe_call(cudaMalloc((void **) &gpu_elements, sizeof(query_id_t) * residual.n)); //TODO: I dont know if this is truly needed
	//safe_call(cudaMemcpy(gpu_elements, elements, sizeof(query_id_t) * residual.n, cudaMemcpyHostToDevice));// TODO: need to rethink this

	matI gpu_idxs = idxs;

	std::printf("Allocating %d MB for idxs\n", sizeof(int) * idxs.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_idxs.mat, sizeof(int) * idxs.n));

	mat gpu_dists = dists;
	std::printf("Allocating %d MB for dists\n", sizeof(float) * dists.n / 1024 / 1024);
	safe_call(alloc((void **) &gpu_dists.mat, sizeof(float) * dists.n));

	//allocating the input buffer
	int* gpu_starting_inputid;
	std::printf("Allocating %d MB for input buffer\n", sizeof(int) * (residual.n + 1) / 1024 / 1024);
	safe_call(alloc((void **) &gpu_starting_inputid, sizeof(int) * (residual.n + 1)));
	safe_call(cudaMemcpy(gpu_starting_inputid, starting_inputid, sizeof(int) * (residual.n + 1), cudaMemcpyHostToDevice));

	Img* gpu_input;
	std::printf("Number of images: %d\n", num_imgs);
	std::printf("Image: %d\n", sizeof(Img));
	std::printf("Allocating %d MB for images\n",  sizeof(Img) * num_imgs / 1024 / 1024);
	safe_call(alloc((void **) &gpu_input, sizeof(Img) * num_imgs));

	dim3 block(1024, 1, 1);
	dim3 grid(ACTIVE_BLOCKS, 1, 1);

	//find biggest ivf entry
	int biggest = 0;
	for (int i = 0; i < ivf_size; i++ ) if (ivf[i].idstam > biggest) biggest = ivf[i].idstam;

	int sm_size = 48 << 10;

	std::printf("Trying to allocate: %dKB in shared memory\n", 48 << 10 / 1024);

	compute_dists<<<grid, block,  sm_size>>>(gpu_PQ, gpu_residual, gpu_ivf, gpu_entry_map, gpu_starting_imgid, gpu_starting_inputid, gpu_input, gpu_idxs, gpu_dists, k);

	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch compute_dists kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	} else std::printf("SUCESSS!\n");

	std::printf("After calling the kernel\n");

	//exit(0);

	//RECEIVING DATA FROM GPU
	//safe_call(cudaMemcpy(elements, 0, sizeof(query_id_t) * residual.n, cudaMemcpyDeviceToHost));

	safe_call(cudaMemcpy(idxs.mat, gpu_idxs.mat , sizeof(int) * idxs.n, cudaMemcpyDeviceToHost));
	safe_call(cudaMemcpy(dists.mat, gpu_dists.mat, sizeof(float) * dists.n, cudaMemcpyDeviceToHost));

	//FREEING MEMORY
	cudaFree(gpu_PQ.centroids);
	cudaFree(gpu_residual.mat);
	cudaFree(gpu_ivf);

	for (int i = 0; i < ivf_size; i++) {
		cudaFree(tmp_ivf[i].ids);
		cudaFree(tmp_ivf[i].codes.mat);
	}

	cudaFree(gpu_entry_map);
	cudaFree(gpu_starting_imgid);
	cudaFree(gpu_idxs.mat);
	cudaFree(gpu_dists.mat);
	cudaFree(gpu_starting_inputid);
	cudaFree(gpu_input);

	delete[] tmp_ivf;

}
