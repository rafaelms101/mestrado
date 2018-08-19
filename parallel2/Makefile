.SUFFIXES: .cpp .c .o

include makefile.inc

CC=gcc -g -O3
CXX=g++  -g -O3
MF =
IVF =
MPICXX = mpic++ $(MF) $(IVF) -fopenmp  -g -no-pie
PQ_UTILS_DIR = ./pq-utils
OBJDIR=./obj
IVF_DIR = ./ivf_pq
OMP = -fopenmp
PTHREAD = -lpthread
NVCC_FLAGS = -O3

all: train genivf ivfpq_test


ifeq "$(USEARPACK)" "yes"
  EXTRAYAELLDFLAG=$(ARPACKLDFLAGS)
  EXTRAMATRIXFLAG=-DHAVE_ARPACK
endif

ifeq "$(USEOPENMP)" "no"
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
$(OBJDIR)/pq_assign.o: $(PQ_UTILS_DIR)/pq_assign.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael_needs//nn.h yael_needs//vector.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_new.o:
$(OBJDIR)/pq_new.o: $(PQ_UTILS_DIR)/pq_new.cpp $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael_needs//kmeans.h yael_needs//vector.h yael_needs//matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_load_vectors.o:
$(OBJDIR)/pq_test_load_vectors.o: $(PQ_UTILS_DIR)/pq_test_load_vectors.cpp $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael_needs//matrix.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test_compute_stats.o:
$(OBJDIR)/pq_test_compute_stats.o: $(PQ_UTILS_DIR)/pq_test_compute_stats.cpp $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
		$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_search.o:
$(OBJDIR)/pq_search.o: $(PQ_UTILS_DIR)/pq_search.cpp $(PQ_UTILS_DIR)/pq_search.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(PQ_UTILS_DIR)/pq_new.h yael_needs//matrix.h yael_needs//kmeans.h yael_needs//vector.h yael_needs//nn.h
				$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

mycuda.o:
$(OBJDIR)/mycuda.o: $(IVF_DIR)/mycuda.cu $(IVF_DIR)/topk.cu
	nvcc -c $(NVCC_FLAGS) $< -o $@

pq_test.o:
$(OBJDIR)/pq_test.o: pq_test.cpp yael_needs/vector.h yael_needs/kmeans.h $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

pq_test: $(OBJDIR)/pq_test.o $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h  $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h
	$(CXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS)


# IVF_TEST

# $(OBJDIR)/myIVF.o:  $(IVF_DIR)/myIVF.cpp $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael_needs/kmeans.h yael_needs/vector.h
# 	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_training.o: $(IVF_DIR)/ivf_training.cpp $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h  $(PQ_UTILS_DIR)/pq_assign.h $(PQ_UTILS_DIR)/pq_new.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael_needs/kmeans.h yael_needs/vector.h yael_needs/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_assign.o: $(IVF_DIR)/ivf_assign.cpp $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h  yael_needs/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_search.o: $(IVF_DIR)/ivf_search.cpp $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h yael_needs/kmeans.h yael_needs/vector.h yael_needs/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/ivf_aggregator.o: $(IVF_DIR)/ivf_aggregator.cpp $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h yael_needs/kmeans.h yael_needs/vector.h yael_needs/nn.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

$(OBJDIR)/k_min.o: $(IVF_DIR)/k_min.cpp $(IVF_DIR)/k_min.h $(IVF_DIR)/ivf_aggregator.cpp $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h yael_needs/kmeans.h yael_needs/vector.h yael_needs/nn.h
	$(CXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags)

$(OBJDIR)/ivfpq_test.o: ivf_test.cpp $(OBJDIR)/ivf_aggregator.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/ivf_training.o $(OBJDIR)/ivf_search.o $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_aggregator.h $(IVF_DIR)/ivf_assign.h $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

ivfpq_test:  $(OBJDIR)/mycuda.o $(OBJDIR)/ivfpq_test.o $(OBJDIR)/ivf_aggregator.o $(OBJDIR)/ivf_assign.o $(OBJDIR)/ivf_training.o $(OBJDIR)/ivf_search.o $(OBJDIR)/k_min.o $(IVF_DIR)/k_min.h $(IVF_DIR)/ivf_search.h $(IVF_DIR)/ivf_training.h $(IVF_DIR)/myIVF.h $(IVF_DIR)/ivf_assign.h $(IVF_DIR)/ivf_aggregator.h  $(OBJDIR)/pq_test_load_vectors.o $(PQ_UTILS_DIR)/pq_test_load_vectors.h $(OBJDIR)/pq_test_compute_stats.o $(PQ_UTILS_DIR)/pq_test_compute_stats.h $(OBJDIR)/pq_new.o $(PQ_UTILS_DIR)/pq_new.h $(OBJDIR)/pq_assign.o $(PQ_UTILS_DIR)/pq_assign.h $(OBJDIR)/pq_search.o $(PQ_UTILS_DIR)/pq_search.h
		$(MPICXX) $(cflags)  -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS) $(PTHREAD) -lcudart -L/opt/cuda/lib64 -lboost_system -lboost_filesystem

$(OBJDIR)/train.o: train.cpp $(IVF_DIR)/ivf_training.h $(PQ_UTILS_DIR)/pq_test_load_vectors.h
	$(MPICXX) $(cflags) -c train.cpp -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

train: $(OBJDIR)/train.o $(OBJDIR)/ivf_training.o $(OBJDIR)/pq_test_load_vectors.o $(OBJDIR)/pq_new.o $(OBJDIR)/pq_assign.o
	$(MPICXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS) $(PTHREAD) -lboost_system -lboost_filesystem

$(OBJDIR)/genivf.o: genivf.cpp $(IVF_DIR)/ivf_training.h $(OBJDIR)/pq_test_load_vectors.o
	$(MPICXX) $(cflags) -c $< -o $@ $(flags) $(extracflags) $(yaelcflags) $(PTHREAD)

genivf: $(OBJDIR)/genivf.o $(OBJDIR)/ivf_training.o $(OBJDIR)/pq_test_load_vectors.o $(OBJDIR)/pq_new.o $(OBJDIR)/pq_assign.o
	$(MPICXX) $(cflags) -o $@ $^ $(LDFLAGS) $(LAPACKLDFLAGS) $(THREADLDFLAGS) $(EXTRAYAELLDFLAG) $(YAELLDFLAGS) $(PTHREAD) -lboost_system -lboost_filesystem

clean:
	rm -f $(OBJDIR)/*.o pq_test *.o ivfpq_test train genivf
