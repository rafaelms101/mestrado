#include <cuda_runtime.h>
#include <cstdio>
#include <utility>

#include <cmath>
#include <vector>

#include <cstdio>
#include <utility>

enum class HeapType {
	kMinHeap, kMaxHeap
};
enum class PreferIndices {
	kLower, kHigher
};

struct Img {
	float dist;
	int imgid;

	__device__ bool operator==(const Img& r) const {
		return r.imgid == imgid;
	}

	__device__ bool operator<(const Img& r) const {
		return dist > r.dist || (dist == r.dist && imgid > r.imgid);
	}

	__device__ bool operator>(const Img& r) const {
		return dist < r.dist || (dist == r.dist && imgid < r.imgid);
	}
};

template<typename T>
struct Entry {
	int index;
	T value;

	// Test-only.
	static bool greater(const Entry<T>& a, const Entry<T>& b) {
		if (a.value == b.value) {
			return a.index < b.index;
		}
		return a.value > b.value;
	}
};

template<typename T>
struct LinearData {
	typedef Entry<T> Entry;

	__device__ Entry& operator[](std::size_t index) const {
		return data[index];
	}

	__device__ int get_index(int i) const {
		return data[i].index;
	}
	__device__ T get_value(int i) const {
		return data[i].value;
	}

	Entry* const data;
};

template<typename T>
struct IndirectLinearData {
	typedef Entry<T> Entry;

	__device__ Entry& operator[](std::size_t index) const {
		return data[index];
	}

	__device__ int get_index(int i) const {
		return backing_data[data[i].index].index;
	}
	__device__ T get_value(int i) const {
		return data[i].value;
	}

	Entry* const data;
	Entry* const backing_data;
};

template<typename T>
struct StridedData {
	typedef Entry<T> Entry;

	__device__ Entry& operator[](std::size_t index) const {
		return data[index * num_subheaps + threadIdx.x];
	}

	__device__ int get_index(int i) const {
		return (*this)[i].index;
	}
	
	__device__ T get_value(int i) const {
		return (*this)[i].value;
	}

	Entry* const data;
	int num_subheaps;
};

// A heap of Entry<T> that can either work as a min-heap or as a max-heap.
template<HeapType heapType, PreferIndices preferIndices,
		template<typename > class Data, typename T>
struct IndexedHeap {
	typedef typename Data<T>::Entry Entry;
	const Data<T> data;

	__device__ bool is_above(int left, int right) {
		T left_value = data.get_value(left);
		T right_value = data.get_value(right);
		if (left_value == right_value) {
			if (preferIndices == PreferIndices::kLower) {
				return data.get_index(left) < data.get_index(right);
			} else {
				return data.get_index(left) > data.get_index(right);
			}
		}
		if (heapType == HeapType::kMinHeap) {
			return left_value < right_value;
		} else {
			return left_value > right_value;
		}
	}

	__device__ void assign(int i, const Entry& entry) {
		data[i] = entry;
	}

	__device__ void push_up(int i) {
		int child = i;
		int parent;
		for (; child > 0; child = parent) {
			parent = (child - 1) / 2;
			if (!is_above(child, parent)) {
				// Heap property satisfied.
				break;
			}
			swap(child, parent);
		}
	}

	__device__ void swap(int a, int b) {
		auto tmp = data[b];
		data[b] = data[a];
		data[a] = tmp;
	}

	__device__ void push_root_down(int k) {
		push_down(0, k);
	}

	// MAX-HEAPIFY in Cormen
	__device__ void push_down(int node, int k) {
		while (true) {
			const int left = 2 * node + 1;
			const int right = left + 1;
			int smallest = node;
			if (left < k && is_above(left, smallest)) {
				smallest = left;
			}
			if (right < k && is_above(right, smallest)) {
				smallest = right;
			}
			if (smallest == node) {
				break;
			}
			swap(smallest, node);
			node = smallest;
		}
	}

	// BUILD-MAX-HEAPIFY in Cormen
	__device__ void build(int k) {
		for (int node = (k - 1) / 2; node >= 0; node--) {
			push_down(node, k);
		}
	}

	// HEAP-EXTRACT-MAX in Cormen
	__device__ void remove_root(int k) {
		data[0] = data[k - 1];
		push_root_down(k - 1);
	}

	// in-place HEAPSORT in Cormen
	// This method destroys the heap property.
	__device__ void sort(int k) {
		for (int slot = k - 1; slot > 0; slot--) {
			// This is like remove_root but we insert the element at the end.
			swap(slot, 0);
			// Heap is now an element smaller.
			push_root_down(/*k=*/slot);
		}
	}

	__device__ void replace_root(const Entry& entry, int k) {
		data[0] = entry;
		push_root_down(k);
	}

	__device__ const Entry& root() {
		return data[0];
	}
};

template<HeapType heapType, PreferIndices preferIndices,
		template<typename > class Data, typename T>
__device__ IndexedHeap<heapType, preferIndices, Data, T> make_indexed_heap(
		typename Data<T>::Entry* data, int num_shards) {
	return IndexedHeap<heapType, preferIndices, Data, T> { Data<T> { data, num_shards } };
}

// heapTopK walks over [input, input+length) with `step_size` stride starting at
// `start_index`.
// It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
// access elements in `heap_entries`. If sorted=true, the elements will be
// sorted at the end.
template<typename T, template<typename > class Data = LinearData>
__device__ void heapTopK(const T* __restrict__ block_input, int length, int k,
		Entry<T>* __restrict__ shared, int num_subheaps, bool sorted = false,
		int start_index = 0, int step_size = 1) {

	auto heap = make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher,
			Data, T>(shared, num_subheaps);

	int heap_end_index = start_index + k * step_size;
	if (heap_end_index > length) {
		heap_end_index = length;
	}
	// Initialize the min-heap.
	int slot = 0;
	for (int index = start_index; index < heap_end_index; index += step_size, slot++) {
		heap.assign(slot, { index, block_input[index] });
	}

	heap.build(slot); //TODO: [before it was heap.build(k)] verify if the heap building function works when you havent assigned all the elements

	// Now iterate over the remaining items.
	// If an item is smaller than the min element, it is not amongst the top k.
	// Otherwise, replace the min element with it and push upwards.
	for (int index = heap_end_index; index < length; index += step_size) {
		// We prefer elements with lower indices. This is given here.
		// Later elements automatically have higher indices, so can be discarded.
		if (block_input[index] > heap.root().value) {
			// This element should replace the min.
			heap.replace_root( { index, block_input[index] }, k);
		}
	}

	// Sort if wanted.
	if (sorted) {
		heap.sort(k);
	}
}

// mergeShards performs a top-k merge on `num_shards` many sorted streams that
// are sorted and stored in `entries` in a strided way:
// |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
// The overall top k elements are written to `top_k_values` and their indices
// to top_k_indices.
// `top_k_heap` is used as temporary storage for the merge heap.
__device__ void mergeShards(int num_shards, int k,
		Entry<Img>* __restrict__ entries, Entry<Img>* __restrict__ top_k_heap,
		float* top_k_values, int* top_k_indices) {
	// If k < num_shards, we can use a min-heap with k elements to get the top k
	// of the sorted blocks.
	// If k > num_shards, we can initialize a min-heap with the top element from
	// each sorted block.
	const int heap_size = k < num_shards ? k : num_shards;

	// Min-heap part.
	{
		auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
				IndirectLinearData, Img> { IndirectLinearData<Img> { top_k_heap,
				entries } };
		// Initialize the heap as a min-heap.
		for (int slot = 0; slot < heap_size; slot++) {
			min_heap.assign(slot, { slot, entries[slot].value });
		}
		min_heap.build(heap_size);

		// Now perform top k with the remaining shards (if num_shards > heap_size).
		for (int shard = heap_size; shard < num_shards; shard++) {
			const auto entry = entries[shard];
			const auto root = min_heap.root();
			if (entry.value < root.value) {
				continue;
			}
			if (entry.value == root.value
					&& entry.index > entries[root.index].index) {
				continue;
			}
			// This element should replace the min.
			min_heap.replace_root( { shard, entry.value }, heap_size);
		}
	}

	// Max-part.
	{
		// Turn the min-heap into a max-heap in-place.
		auto max_heap = IndexedHeap<HeapType::kMaxHeap, PreferIndices::kLower,
				IndirectLinearData, Img> { IndirectLinearData<Img> { top_k_heap,
				entries } };
		// Heapify into a max heap.
		max_heap.build(heap_size);

		// Now extract the minimum k-1 times.
		// k is treated specially.
		const int last_k = k - 1;
		for (int rank = 0; rank < last_k; rank++) {
			const Entry<Img>& max_element = max_heap.root();
			top_k_values[rank] = max_element.value.dist;

			int shard_index = max_element.index;
			top_k_indices[rank] = entries[shard_index].value.imgid;

			int next_shard_index = shard_index + num_shards;
			// For rank < k-1, each top k heap still contains at least 1 element,
			// so we can draw a replacement.
			max_heap.replace_root(
					{ next_shard_index, entries[next_shard_index].value },
					heap_size);
		}

		// rank == last_k.
		const Entry<Img>& max_element = max_heap.root();
		top_k_values[last_k] = max_element.value.dist;

		int shard_index = max_element.index;
		top_k_indices[last_k] = entries[shard_index].value.imgid;
	}
}

extern __shared__ char shared_memory[];

__device__ void TopKKernel(const int qid, const int num_subheaps, const Img* input,
		const int* const starting_inputid, const int k, const bool sorted, float* output,
		int* indices) {
	const Img* block_input = input + starting_inputid[qid];
	auto tid = threadIdx.x;

	Entry<Img>* shared = (Entry<Img>*) shared_memory;

	
	int length = starting_inputid[qid + 1] - starting_inputid[qid]; //TODO: find a better solution for passing along the number of images
	
	if (tid < num_subheaps) {
		heapTopK<Img, StridedData>(block_input, length, k, shared, num_subheaps, true, tid,  num_subheaps);
	}
	
	__syncthreads();
	
	if (tid == 0) {
		float* block_output = output + qid * k;
		int* batch_indices = indices + qid * k;
		Entry<Img>* top_k_heap = shared + num_subheaps  * k;

		// TODO(blackhc): Erich says: Performance can likely be improved
		// significantly by having the merge be done by multiple threads rather than
		// just one.  ModernGPU has some nice primitives that could help with this.
		mergeShards(num_subheaps, k, shared, top_k_heap, block_output,
				batch_indices);
	}
}

/*
 template <typename T>
 cudaError LaunchTopKKernel(const cudaStream_t& stream, int num_shards,
 const T* input, int batch_size, int length, int k,
 bool sorted, T* output, int* indices) {
 // This code assumes that k is small enough that the computation
 // fits inside shared memory (hard coded to 48KB).  In practice this
 // means k <= 3072 for T=float/int32 and k <= 2048 for T=double/int64.
 // The calculation is:
 //   shared_memory_size / (2 * (sizeof(int) + sizeof(T))) < k.

 // Use as many shards as possible.
 if (num_shards <= 0) {
 constexpr auto shared_memory_size = 48 << 10;  // 48 KB
 const auto heap_size = k * sizeof(Entry<T>);
 // shared_memory_size = (num_shards + 1) * heap_size <=>
 num_shards = shared_memory_size / heap_size - 1;
 if (num_shards <= 0) {
 num_shards = 1;
 }
 auto shard_size = length / num_shards;
 auto min_shard_size = 2 * k;
 if (shard_size < min_shard_size) {
 num_shards = length / min_shard_size;
 }
 if (num_shards <= 0) {
 num_shards = 1;
 } else if (num_shards > 1024) {
 num_shards = 1024;
 }
 }
 // We are limited by the amount of shared memory we have per block.
 auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T>);

 TopKKernel<<<batch_size, num_shards, shared_memory_size, stream>>>(
 input, length, k, sorted, output, indices);
 return cudaGetLastError();
 }*/

__device__ void topk(const int qid, const int num_subheaps, const int k, Img* input, const int* const starting_inputid,
		float* output, int* indexes) {
		TopKKernel(qid, num_subheaps, input, starting_inputid, k, false, output,
						indexes);
}

/*
 struct Entry {
 float value;
 int imgid;
 int index;
 };

 __device__ void sort_strided(Entry* entries, int length) {
 int tid = threadIdx.x;
 int numThreads = blockDim.x;
 
 for (int new_element_id = tid + numThreads; new_element_id < length; new_element_id += numThreads) {
 //Trying to insert element with index id in the sorted array
 
 int id;
 for (id = new_element_id - numThreads; id >= 0; id -= numThreads) {
 if (entries[new_element_id].value <= entries[id].value) break;
 }
 
 Entry to_be_inserted = entries[new_element_id];
 
 //now we shift everyone right
 for (int insertion_id = id + numThreads; insertion_id <= new_element_id; insertion_id += numThreads) {
 Entry tmp = entries[insertion_id];
 entries[insertion_id] = to_be_inserted;
 to_be_inserted = tmp;
 }
 }
 }

 __device__ void sort(Entry* entries, int length) {
 for (int new_element_id = 1; new_element_id < length; new_element_id += 1) {
 //Trying to insert element with index id in the sorted array
 
 int id;
 for (id = new_element_id - 1; id >= 0; id -= 1) {
 if (entries[new_element_id].value <= entries[id].value) break;
 }
 
 Entry to_be_inserted = entries[new_element_id];
 
 //now we shift everyone right
 for (int insertion_id = id + 1; insertion_id <= new_element_id; insertion_id += 1) {
 Entry tmp = entries[insertion_id];
 entries[insertion_id] = to_be_inserted;
 to_be_inserted = tmp;
 }
 }
 }

 __device__ void insert(Entry* entries, int new_id, int k) {
 Entry new_element = entries[new_id];

 int i;

 for (i = k - 1; i >= 0; i--) {
 if (new_element.value > entries[i].value) {
 entries[i + 1] = entries[i];
 } else break;
 }

 entries[i + 1] = new_element;
 }

 //TODO: test the case where  the number of entries is too small
 __device__ void topk(int k, Entry* entry, int length) {
 int tid = threadIdx.x;
 int stride = blockDim.x;

 sort_strided(entry, length);

 for (int id = tid; id < length; id += stride) {
 entry[id].index = id;
 }
 
 __syncthreads();
 
 if (tid == 0) {
 k = min(length, k);
 int lastk = k - 1;

 sort(entry, k);
 
 int start = k;
 int end = min(k + stride - 1, length - 1);

 Entry old_smallest = entry[lastk];
 
 
 while (true) {
 for (int id = start; id <= end; id++) {
 if (entry[id].value > entry[lastk].value) {
 int next_id = entry[id].index + stride;
 insert(entry, id, k);

 if (next_id < length) {
 entry[id] = entry[next_id];
 } else {
 entry[id] = {-1, -1};
 }
 }
 }

 if (entry[lastk].value == old_smallest.value) break;
 }	
 
 }
 }
 */
