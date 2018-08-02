#ifndef DEBUG_H_
#define DEBUG_H_

#define DEBUG

#ifdef DEBUG 

#define debug(...) std::printf(__VA_ARGS__); std::printf("\n");

#else
#define debug(str) ;
#endif


#endif /* DEBUG_H_ */
