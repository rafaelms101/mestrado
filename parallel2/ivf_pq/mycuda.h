#ifndef IVF_PQ_MYCUDA_H_
#define IVF_PQ_MYCUDA_H_

#include "../pq-utils/pq_test_load_vectors.h"
#include "myIVF.h"

void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* rid_to_ivf,  int* qid_to_starting_outid, matI idxs, mat dists, int k, int w);
void preallocate_gpu_mem(pqtipo PQ, ivf_t* ivf, int ivf_size);
void deallocate_gpu_mem();


#endif /* IVF_PQ_MYCUDA_H_ */
