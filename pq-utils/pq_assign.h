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

typedef struct{
	int *mat;
	int n,
		d;
}matI;

void check_assign();

/*
	Copia os subvetores determinados pelo inicio e o fim de  cada vetor
	para o vout
	vout : vetor de saida
	v : int vetor de entrada
	ini : inicio do indice de cada subvetor
	fim : indice do final de cada subvetor
*/
void copySubVectors(float *vout, mat vin, int ini, int fim);
matI pq_assign (pqtipo, mat);


#endif
