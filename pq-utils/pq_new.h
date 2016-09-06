#ifndef H_NEW
#define H_NEW

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "pq_test_load_vectors.h"
extern "C" {
#include "../yael/kmeans.h"
#include "../yael/vector.h"
#include "../yael/matrix.h"
}

/* flags para o kmeans */
#define KMEANS_QUIET                    0x10000
#define KMEANS_INIT_BERKELEY            0x20000
#define KMEANS_NORMALIZE_CENTS          0x40000
#define KMEANS_INIT_RANDOM              0x80000
#define KMEANS_INIT_USER               	0x100000
#define KMEANS_L1                       0x200000
#define KMEANS_CHI2                     0x400000

using namespace std;

typedef struct pqtipo{
	int nsq;
	int ks;
	int ds;
	mat centroids;
} pqtipo;

pqtipo pq_new(int, mat);
void check_new();

/* Concatena dois vetores, caso um não exista ele é criado exatamente igual ao vin
 * vinout : vetor a ser aumentado com os valores de vin
 * vinout_n : tamanho do vetor
 * vin : vetor a ser copiado
 *  ...
 */
void fvec_concat(float* vinout, int vinout_n, float* vin, int vin_n);

void ivec_concat(int* vinout, int vinout_n, int* vin, int vin_n);

#endif
