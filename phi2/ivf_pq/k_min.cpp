#include "k_min.h"

MinHeap::MinHeap(element a[], int size){
    heap_size = size;
    harr = a;  // store address of array
    int i = (heap_size - 1)/2;
    while (i >= 0)
    {
        MinHeapify(i);
        i--;
    }
}

// Method to remove minimum element (or root) from min heap
element MinHeap::extractMin(){
    if (heap_size == 0)
        return make_pair(INT_MAX, -1);

    // Store the minimum vakue.
    element root = harr[0];

    // If there are more than 1 items, move the last item to root
    // and call heapify.
    if (heap_size > 1)
    {
        harr[0] = harr[heap_size-1];
        MinHeapify(0);
    }
    heap_size--;

    return root;
}

// A recursive method to heapify a subtree with root at given index
// This method assumes that the subtrees are already heapified
void MinHeap::MinHeapify(int i){
    int l = left(i);
    int r = right(i);
    int smallest = i;
    if (l < heap_size && harr[l].first < harr[i].first)
        smallest = l;
    if (r < heap_size && harr[r].first < harr[smallest].first)
        smallest = r;
    if (smallest != i)
    {
        swap(&harr[i], &harr[smallest]);
        MinHeapify(smallest);
    }
}

// A utility function to swap two elements
void swap(element *x, element *y){
    element temp = *x;
    *x = *y;
    *y = temp;
}

// Function to return k'th smallest element in a given array
element kthSmallest(element arr[], int n, int k){
    // Build a heap of n elements: O(n) time
    MinHeap mh(arr, n);

    // Do extract min (k-1) times
    for (int i=0; i<k-1; i++)
        mh.extractMin();

    // Return root
    return mh.getMin();
}

//k_min for parallel purpose

//using qsort --- TODO
void k_min_qsort (mat disquerybase, int k, float *dis, int *ids){
	int i, j, d, n;

	if(disquerybase.d==1 && disquerybase.n>1){
		d=disquerybase.n;
		n=1;
	}

	for (i=0; i<n; i++){
		fvec_k_min(disquerybase.mat, d, ids, k);
		for(j=0; j<k; j++){
			dis[j] = disquerybase.mat[ids[j]];
			ids[j]++;
		}
		ids += k;
		dis += k;
		disquerybase.mat += d;
	}
}

//using a stack --- CHECK IF IT IS WORKING
void k_min_stack (mat disquerybase, int k, float *dis, int *ids){
	int i, j, d, n;
	list< std::pair<double, int> > k_mins;

	if(disquerybase.d==1 && disquerybase.n>1){
		d=disquerybase.n;
		n=1;
	}

	for(i = 0; i < n; i++) {
		//finding the k smallest distances
		k_mins.push_front( std::make_pair(disquerybase.mat[i*d], i*d+1) );
		for (j = 1; j < d; j++) {
			if(k_mins.size() <= k){
				k_mins.push_front(std::make_pair(disquerybase.mat[i*d +j], i*d +j+1));
			}
			// else if(k_mins.size() > k){
			// 	//create a new compare function for pairs, sorting them increasing order on the first item(double)
			// 	k_mins.sort(compare_function());
			// }
			else if (disquerybase.mat[i*d +j] > k_mins.back().first) {
				//find where to insert the element in the list to maintain sorted
				for (list< std::pair<double, int> >::iterator it=k_mins.begin(); it != k_mins.end(); ++it){
					std::pair<double, int> & elem(*it);
					//replace the element
					if(it->first > disquerybase.mat[i*d +j]){
						elem = make_pair(disquerybase.mat[i*d +j], i*d +j+1);
						break;
					}
				}
			}
			else if(dis[i*d +j] < k_mins.back().first){
				k_mins.pop_back();
				k_mins.push_front(std::make_pair(disquerybase.mat[i*d +j], i*d+j+1));
			}
		}

    j = 0;
		for (list< std::pair<double, int> >::iterator it=k_mins.begin(); it != k_mins.end(); ++it){
			//pass the result to dis, and ids
			ids[j] = it->second;
			dis[j] = disquerybase.mat[ids[j]];
      j++;
		}
		ids += k;
		dis += k;
		k_mins.clear();
	}

	// for (i=0; i<n; i++){
	// 	fvec_k_min(disquerybase.mat, d, ids, k);
	// 	for(j=0; j<k; j++){
	// 		dis[j] = disquerybase.mat[ids[j]];
	// 		ids[j]++;
	// 	}
	// 	ids += k;
	// 	dis += k;
	// 	disquerybase.mat += d;
	// }
}


__declspec(target(mic)) void my_k_min(dis_t q, int ktmp, float *dis, int *ids){
	int *qidx = (int*)malloc(sizeof(int)*q.idx.n), j=1;
	float *qdis = (float*)malloc(sizeof(float)*q.dis.n);

	if (q.dis.n == 0 || ktmp == 0)return;

	memcpy(&qdis[0],&q.dis.mat[0], sizeof(float)*q.dis.n);

	memcpy(&qidx[0],&q.idx.mat[0], sizeof(int)*q.idx.n);

	constroiHeap(q.dis.n, qdis, qidx);

	dis[0] = qdis[0];

	qdis[0] = qdis[q.dis.n-1];

	ids[0] = qidx[0];

	qidx[0] = qidx[q.dis.n-1];

	for (int i = q.dis.n-1; i > q.dis.n-ktmp; i--){

		trocarRaiz(q.dis.n, qdis, qidx);

		dis[j] = qdis[0];

		qdis[0] = qdis[i-1];

		ids[j] = qidx[0];

		qidx[0] = qidx[i-1];

        	j++;
	}
	free(qidx);
	free(qdis);

}


__declspec(target(mic)) static void constroiHeap (int n, float *qdis, int *qidx){
	int k, aux;
	float aux2;

	for (k = 1; k < n; k++) {
		int f = k+1;
		while (f > 1 && qdis[(f/2)-1] > qdis[f-1]) {
        	aux2 = qdis[(f/2)-1];
        	qdis[(f/2)-1] = qdis[f-1];
        	qdis[f-1] = aux2;
        	aux = qidx[(f/2)-1];
        	qidx[(f/2)-1] = qidx[f-1];
        	qidx[f-1] = aux;
			f /= 2;
		}
	}
}


__declspec(target(mic)) static void trocarRaiz (int n, float *qdis, int *qidx){
	int p = 1, f = 2, aux2 = qidx[0];
	float aux = qdis[0];
	while (f < n) {
		if (f < n-1 && qdis[f-1] > qdis[f])  f++;
		if (aux <= qdis[f]) break;
		qdis[p-1] = qdis[f-1];
		qidx[p-1] = qidx[f-1];
		p = f;
		f = 2*p;
	}
	qdis[p-1] = aux;
	qidx[p-1] = aux2;
}
