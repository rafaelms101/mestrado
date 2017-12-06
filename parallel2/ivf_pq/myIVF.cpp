#include "myIVF.h"

void set_last (int comm_sz, int *last_assign, int *last_search, int *last_aggregator){

	#ifdef WRITE_IVF
		*last_assign=0;
		*last_search=comm_sz-1;
	#else
		*last_assign=1;
		*last_search=comm_sz-2;
		*last_aggregator=comm_sz-1;
	#endif
}
