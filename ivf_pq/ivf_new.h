#ifndef H_IVFNEW
#define H_IVFNEW

  #include<stdio.h>
  #include<stdlib.h>
  extern "C"{
  #include "../yael/vector.h"
  #include "../yael/nn.h"
  #include "../yael/kmeans.h"
  }
  #include "../pq-utils/pq_test_load_vectors.h"
  #include "../pq-utils/pq_new.h"
  #include "myIVF.h"

  void subtract(mat v, float* v2, int* idx);
  ivfpq_t ivfpq_new(int coarsek, int nsq, mat vtrain);


#endif
