

#ifndef K_MIN
#define K_MIN

// A C++ program to find k'th smallest element using min heap
#include<iostream>
#include<climits>
#include <list>
#include "../pq-utils/pq_new.h"
#include "../pq-utils/pq_test_load_vectors.h"
#include "../pq-utils/pq_test_compute_stats.h"
#include "ivf_assign.h"
#include "ivf_new.h"
#include "ivf_search.h"
#include "myIVF.h"

using namespace std;

typedef std::pair<double, int> element;

// A class for Min Heap
class MinHeap{
    element *harr; // pointer to array of elements in heap
    int capacity; // maximum possible size of min heap
    int heap_size; // Current number of elements in min heap
public:
    MinHeap(element a[], int size); // Constructor
    void MinHeapify(int i);  //To minheapify subtree rooted with index i
    int parent(int i) { return (i-1)/2; }
    int left(int i) { return (2*i + 1); }
    int right(int i) { return (2*i + 2); }

    element extractMin();  // extracts root (minimum) element
    element getMin() { return harr[0]; } // Returns minimum
};

// Prototype of a utility function to swap two integers
void swap(element *x, element *y);
void k_min_qsort (mat disquerybase, int k, float *dis, int *ids);
void k_min_stack (mat disquerybase, int k, float *dis, int *ids);

void my_k_min(dis_t q, int ktmp, float *dis, int *ids);
static void constroiHeap (int n, float *qdis, int *qidx);
static void trocarRaiz (int n, float *qdis, int *qidx);

#endif
