#include <stdio.h>
#include <math.h>
#include "pq-utils/pq_new.h"
#include "pq-utils/pq_test_load_vectors.h"
// extern "C" {
// #include "yael/vector.h"
// #include "yael/kmeans.h"
// }

#include "ivf_pq/ivf_assign.h"
#include "ivf_pq/ivf_new.h"
//#include "ivf_pq/ivf_search.h"

int main(int argv, char **argc){
  int coarsek = 256;
  int nsq = 8;
  char* dataset = argc[1];

  data v;

  v = pq_test_load_vectors(dataset);

  ivfpq_t ivfpq = ivfpq_new(coarsek, nsq, v.train);


//  ivf_t *ivf = ivfpq_assign(ivfpq, v.base);


  return 0;
}
