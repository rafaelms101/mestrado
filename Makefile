.SUFFIXES: .cpp .c .o

include makefile.inc

CC=gcc -g
CXX=g++ -g
PQ_UTILS_DIR = ./pq-utils
OBJDIR=./obj
IVF_DIR = ./ivf_pq

all: pq_test_load_vectors.o pq_new.o pq_assign.o pq_search.o pq_test.o pq_test $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_new.o ivfpq_test.o ivfpq_test

ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "yes"
  EXTRAMATRIXFLAG+=-fopenmp
  EXTRAYAELLDFLAG+=-fopenmp
endif

# Various

# .c.o:
# 	$(cc) $(CFLAGS) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)
#
# .cpp.o:
# 	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_assign.o:
$(OBJDIR)/pq_assign.o: $(PQ_UTILS_DIR)/pq_assign.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael/nn.h yael/vector.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_new.o:
$(OBJDIR)/pq_new.o: $(PQ_UTILS_DIR)/pq_new.cpp $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_load_vectors.o:
$(OBJDIR)/pq_test_load_vectors.o: $(PQ_UTILS_DIR)/pq_test_load_vectors.cpp $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_compute_stats.o:
$(OBJDIR)/pq_test_compute_stats.o: $(PQ_UTILS_DIR)/pq_test_compute_stats.cpp $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)		

pq_search.o:
$(OBJDIR)/pq_search.o: $(PQ_UTILS_DIR)/pq_search.cpp $(PQ_UTILS_DIR)/pq_search.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael/matrix.h yael/kmeans.h yael/vector.h yael/nn.h
				$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test.o:
$(OBJDIR)/pq_test.o: pq_test.cpp yael/vector.h yael/kmeans.h yael/ivf.h $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test: $(OBJDIR)/pq_test.o $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h  $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h
	$(CXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)

# IVF_TEST

$(OBJDIR)/myIVF.o:  $(IVF_DIR)/myIVF.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

$(OBJDIR)/ivf_new.o: $(IVF_DIR)/ivf_new.cpp $(IVF_DIR)/ivf_new.h $(IVF_DIR)/myIVF.h  $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/nn.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

$(OBJDIR)/ivf_assign.o: $(IVF_DIR)/ivf_assign.cpp $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_new.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael/kmeans.h yael/vector.h yael/nn.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

ivfpq_test.o: ivf_test.cpp $(OBJDIR)/ivf_assign.o $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_new.o $(IVF_DIR)/ivf_new.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_assign.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

ivfpq_test: ivfpq_test.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/myIVF.o $(OBJDIR)/ivf_new.o $(IVF_DIR)/ivf_new.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_assign.h  $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h
		$(CXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)

clean:
	rm -f $(OBJDIR)/*.o pq_test *.o ivfpq_test
