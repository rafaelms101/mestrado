#include <stdio.h>
#include <math.h>
extern "C" {
#include "yael/vector.h"
#include "yael/kmeans.h"
#include "yael/ivf.h"
}

int main(){
	float* vec= fvec_new_rand(10);
	printf("HELLO\n");

	fvec_print(vec, 10);

	return 0;
}
