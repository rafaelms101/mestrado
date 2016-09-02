.SUFFIXES: .cpp .c .o 

include makefile.inc

CC=gcc 
CXX=g++

all: pq_test

ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif

# Various  

.c.o:
	$(cc) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

.cpp.o:
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)


pq_test: pq_test.o 
	$(CXX) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)

# Dependencies  

pq_test.o: pq_test.cpp yael/vector.h yael/kmeans.h yael/ivf.h

clean:
	rm -f *.o pq_test 
