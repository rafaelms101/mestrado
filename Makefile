.SUFFIXES: .cpp .c .o

include makefile.inc

CC=gcc
CXX=g++
PQ_UTILS_DIR = ./pq-utils
OBJDIR=./obj

all: pq_test_load_vectors.o pq_new.o pq_assign.o pq_test.o pq_test

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

pq_assign.o:
$(OBJDIR)/pq_assign.o: $(PQ_UTILS_DIR)/pq_assign.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael/nn.h yael/vector.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_new.o:
$(OBJDIR)/pq_new.o: $(PQ_UTILS_DIR)/pq_new.cpp $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_load_vectors.o:
$(OBJDIR)/pq_test_load_vectors.o: $(PQ_UTILS_DIR)/pq_test_load_vectors.cpp $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test.o:
$(OBJDIR)/pq_test.o: pq_test.cpp yael/vector.h yael/kmeans.h yael/ivf.h $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test: $(OBJDIR)/pq_test.o $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h  $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h
	$(CXX) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)

# Dependencies

clean:
	rm -f $(OBJDIR)/*.o pq_test *.o
