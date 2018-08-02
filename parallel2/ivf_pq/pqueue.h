#ifndef PQUEUE_H_
#define PQUEUE_H_

#include <utility>
#include <algorithm>


bool cmpPair(const std::pair<float, int> &a, const std::pair<float, int> &b) {
	return a.first < b.first;
}

struct pqueue {
	pqueue(std::pair<float, int>* _base, int _max) {
		size = 0;
		max = _max;
		base = _base;
	}
	
	void add(float dist, int id) {
		if (size < max) {
			base[size] = std::make_pair(dist, id);
			size++;
			
			if (size == max) {
				std::make_heap(base, base + max, cmpPair);
			}
		}
		else if (dist < base[0].first) {
			std::pop_heap(base, base + max, cmpPair);
			base[max-1] = std::make_pair(dist, id);
			std::push_heap(base, base + max - 1, cmpPair);
		}
	}

	int size;
	int max;
	std::pair<float,int>* base;
};


#endif /* PQUEUE_H_ */
