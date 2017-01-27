

#ifndef K_MIN
#define K_MIN

// A C++ program to find k'th smallest element using min heap
#include<iostream>
#include<climits>
#include <list>


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

    int extractMin();  // extracts root (minimum) element
    element getMin() { return harr[0]; } // Returns minimum
};

// Prototype of a utility function to swap two integers
void swap(int *x, int *y);
void k_min_qsort (mat disquerybase, int k, float *dis, int *ids);
void k_min_stack (mat disquerybase, int k, float *dis, int *ids);

#endif
