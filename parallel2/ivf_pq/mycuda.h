#ifndef IVF_PQ_MYCUDA_H_
#define IVF_PQ_MYCUDA_H_

#include "../pq-utils/pq_test_load_vectors.h"
#include "myIVF.h"

void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* entry_map, int* starting_imgid,  int* starting_inputid,  int num_imgs, matI idxs, mat dists, int k);


#endif /* IVF_PQ_MYCUDA_H_ */
