#ifndef H_IVFASSIGN
#define H_IVFASSIGN

  #include<stdio.h>
  #include<stdlib.h>
  extern "C"{
  #include "../yael/vector.h"
  #include "../yael/nn.h"
  }
  #include "../pq-utils/pq_test_load_vectors.h"
  #include "../pq-utils/pq_new.h"
  #include "../pq-utils/pq_assign.h"
  #include "myIVF.h"
  #include "ivf_new.h"

  ivf* ivfpq_assign(ivfpq_t ivfpq, mat vbase);



#endif
