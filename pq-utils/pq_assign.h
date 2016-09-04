#ifndef H_ASSIGN
#define H_ASSIGN

#include <iostream>
#include <math.h>
extern "C" {
#include "../yael/vector.h"
#include "../yael/nn.h"
}

#include "pq_assign.h"
#include "pq_new.h"
#include "pq_test_load_vectors.h"

using namespace std;

#define L2 2

void check_assign();

void copySubVectors(float *vout, mat vin, int ini, int fim);
mat pq_assign (pqtipo pq, mat v, int n);


#endif
