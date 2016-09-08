#ifndef H_SEARCH
#define H_SEARCH

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "pq_test_load_vectors.h"
#include "pq_new.h"
extern "C" {
#include "../yael/kmeans.h"
#include "../yael/nn.h"
#include "../yael/vector.h"
#include "../yael/matrix.h"
}


void pq_search(pqtipo pq, int *cbase, mat vquery, int k);
float* sumidxtab(float* D, mat v, int offset);


#endif
