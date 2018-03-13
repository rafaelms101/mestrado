#include <cuda_runtime.h>

#include "mycuda.h"

#include "helper_cuda.h"

#include <cstdio>

#define safe_call(call) if (cudaSuccess != call) { err = cudaGetLastError(); \
													fprintf(stderr, "Failed: call (error code %s)!\n", \
															cudaGetErrorString(err)); \
													exit(EXIT_FAILURE); }

//TODO: remember to not execute queries that dont correspond to an entry on the problem
__global__ void compute_dists(pqtipo PQ, mat residual, ivf_t* ivf, int* entry_map, int* starting_imgid, query_id_t* elements, matI idxs, mat dists) {
	int d = threadIdx.x; 
	int k = threadIdx.y;
	int tid = d * PQ.ks + k;
	int qid = blockIdx.x;
	
	extern __shared__ float distab[];
	
	float* centroid = PQ.centroids + (d * PQ.ks + k) * PQ.ds;
	float* sub_residual = residual.mat + qid * PQ.nsq * PQ.ds + d * PQ.ds;
	float dist = 0;
	
	for (int i = 0; i < PQ.ds; i++) {
		float diff = sub_residual[i] - centroid[i];
		dist += diff * diff;
	}
	
	distab[PQ.ks * d + k] = dist;

	if (tid < residual.n) { //TODO: its very likely that this is unneeded
		ivf_t entry = ivf[entry_map[qid]];
		
		if (threadIdx.x == 0 && threadIdx.y == 0) { //only one thread per block should do this, since they all refer to the same query
			atomicAdd(&elements[qid].tam, entry.idstam); //atomic because we will have up to w threads trying to increase this at the same time
		}
		
		int block_size = blockDim.x * blockDim.y;

		for (int i = tid; i < entry.idstam; i += block_size) {
			float dist = 0;

			for (int s = 0; s < PQ.nsq; s++) {
				dist += distab[PQ.ks * s + entry.codes.mat[PQ.nsq * i + s]];
			}

			dists.mat[starting_imgid[qid] + i] = dist;
			idxs.mat[starting_imgid[qid] + i] = entry.ids[i];
		}
	}	
}

void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* entry_map, int* starting_imgid, query_id_t* elements, matI idxs, mat dists) {
	//TODO: implement / redo the error handling so that we have less code duplication
	cudaError_t err = cudaSuccess;
	
	pqtipo gpu_PQ = PQ;
	safe_call(cudaMalloc((void **) &gpu_PQ.centroids, sizeof(float) * PQ.centroidsd * PQ.centroidsn));
	safe_call(cudaMemcpy(gpu_PQ.centroids, PQ.centroids, sizeof(float) * PQ.centroidsd * PQ.centroidsn, cudaMemcpyHostToDevice));
	
	mat gpu_residual = residual;
	safe_call(cudaMalloc((void **) &gpu_residual.mat, sizeof(float) * residual.n * residual.d));
	safe_call(cudaMemcpy(gpu_residual.mat, residual.mat, sizeof(float) * residual.n * residual.d, cudaMemcpyHostToDevice));
	
	std::printf("residual.n=%d and residual.d=%d\n", residual.n, residual.d);
	
	ivf_t* gpu_ivf;
	safe_call(cudaMalloc((void **) &gpu_ivf, sizeof(ivf_t) * ivf_size));
	
	ivf_t* tmp_ivf = new ivf_t[ivf_size];
	
	for (int i = 0; i < ivf_size; i++) {
		tmp_ivf[i].idstam = ivf[i].idstam; 
		tmp_ivf[i].codes = ivf[i].codes;
		
		safe_call(cudaMalloc((void **) &tmp_ivf[i].ids, sizeof(int) * tmp_ivf[i].idstam));
		safe_call(cudaMemcpy(tmp_ivf[i].ids, ivf[i].ids, sizeof(int) * ivf[i].idstam, cudaMemcpyHostToDevice));
		
		safe_call(cudaMalloc((void **) &tmp_ivf[i].codes.mat, sizeof(int) * tmp_ivf[i].codes.n * tmp_ivf[i].codes.d));
		safe_call(cudaMemcpy(tmp_ivf[i].codes.mat, ivf[i].codes.mat, sizeof(int) * ivf[i].codes.n * ivf[i].codes.d, cudaMemcpyHostToDevice));
	}
	
	safe_call(cudaMemcpy(gpu_ivf, tmp_ivf, sizeof(ivf_t) * ivf_size, cudaMemcpyHostToDevice));
	
	int* gpu_entry_map;
	safe_call(cudaMalloc((void **) &gpu_entry_map, sizeof(int) * residual.n));
	safe_call(cudaMemcpy(gpu_entry_map, entry_map, sizeof(int) * residual.n, cudaMemcpyHostToDevice));
	
	int* gpu_starting_imgid;
	safe_call(cudaMalloc((void **) &gpu_starting_imgid, sizeof(int) * residual.n));
	safe_call(cudaMemcpy(gpu_starting_imgid, starting_imgid, sizeof(int) * residual.n, cudaMemcpyHostToDevice));
	
	query_id_t* gpu_elements;
	safe_call(cudaMalloc((void **) &gpu_elements, sizeof(query_id_t) * residual.n)); //TODO: I dont know if this is truly needed
	safe_call(cudaMemcpy(gpu_elements, elements, sizeof(query_id_t) * residual.n, cudaMemcpyHostToDevice));// TODO: need to rethink this
	
	matI gpu_idxs = idxs;
	safe_call(cudaMalloc((void **) &gpu_idxs.mat, sizeof(int) * idxs.n));
	
	mat gpu_dists = dists;
	safe_call(cudaMalloc((void **) &gpu_dists.mat, sizeof(float) * dists.n));

	dim3 block(PQ.nsq, PQ.ks, 1);
	dim3 grid(residual.n, 1, 1);
	
	std::printf("Before calling the kernel\n");
	compute_dists<<<grid, block, sizeof(float) * PQ.ks * PQ.nsq>>>(gpu_PQ, gpu_residual, gpu_ivf, gpu_entry_map, gpu_starting_imgid, gpu_elements, gpu_idxs, gpu_dists); 
	
	err = cudaGetLastError();

	if (err != cudaSuccess) {
		fprintf(stderr, "Failed to launch compute_dists kernel (error code %s)!\n",
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	} else std::printf("SUCESSS!\n");
	
	std::printf("After calling the kernel\n");
	
	//RECEIVING DATA FROM GPU
	safe_call(cudaMemcpy(elements, gpu_elements, sizeof(query_id_t) * residual.n, cudaMemcpyDeviceToHost));
	
	safe_call(cudaMemcpy(idxs.mat, gpu_idxs.mat , sizeof(int) * idxs.n, cudaMemcpyDeviceToHost));
	
	safe_call(cudaMemcpy(dists.mat, gpu_dists.mat, sizeof(int) * dists.n, cudaMemcpyDeviceToHost));
	
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
	cudaFree(gpu_elements);
	cudaFree(gpu_idxs.mat);
	cudaFree(gpu_dists.mat);
	
	delete[] tmp_ivf;
	
	for (int i = 0; i < residual.n; i++) {
		std::printf("element[%d].id=%d and element[%d].tam=%d\n", i, elements[i].id, i, elements[i].tam);
	}
	
	//std::exit(0);
	std::printf("EXITING CORE_GPU\n");
}
