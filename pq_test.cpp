#include <stdio.h>
#include <math.h>
#include "pq-utils/pq_assign.h"
#include "pq-utils/pq_new.h"
extern "C" {
#include "yael/vector.h"
#include "yael/kmeans.h"
#include "yael/ivf.h"
}

int main(){
	check_new();
	check_assign();

	int *vec = ivec_new_set(2, 0);
	int *vec2 = ivec_new_set(2, 1);

	printf("vec = ");
	ivec_print(vec, 2);

	printf("vec2 = ");
	ivec_print(vec2, 2);

	ivec_concat(vec, 2, 1, vec2, 2, 1);

	printf("vec = ");
	ivec_print(vec, 4);

	return 0;
}
