#include "ivf_search.h"

#include <set>
#include <cstdio>
#include <queue>
#include <ctime>

#include "mycuda.h"

#include <cmath>


int iter;
static int last_assign, last_search, last_aggregator;
static sem_t sem;


//TODO: dont try this at home
time_t start, end;
double diff;



#define sw(call) time(&start); \
				  call ; \
				  time(&end); \
				  diff = difftime (end,start); \
				  printf ("Elapsed: %.2lf seconds on call: %s\n", diff, #call);
				  
				  

//TODO: refactor variable names, they are terrible
//TODO: comment the code
//TODO: think about how to parallelize the other stages and/or if it would be worthwhile

float dist2(float* a, float* b, int size) {
	float d = 0;
	for (int i = 0; i < size; i++) {
		float diff = a[i] - b[i];
		d += diff * diff;
	}
	
	return d;
}

void core_cpu(pqtipo PQ, mat partial_residual, ivf_t* partial_ivf, int ivf_size, int* entry_map, int* starting_imgid, int* starting_inputid, int num_imgs, matI idxs, mat dists) {
	std::printf("idxs.n=%d\n", idxs.n);
	
	int p = 0;
		
	for (int i = 0; i < partial_residual.n; i++) {
		float* residual = partial_residual.mat + i * PQ.nsq * PQ.ds;
		
		mat distab;
		distab.mat = new float[PQ.ks * PQ.nsq]; 
		distab.n = PQ.nsq;
		distab.d = PQ.ks;
		
		for (int d = 0; d < PQ.nsq; d++) {
			for (int k = 0; k < PQ.ks; k++) {
				float dist = dist2(residual + d * PQ.ds, PQ.centroids + d * PQ.ks * PQ.ds + k * PQ.ds, PQ.ds);
				distab.mat[PQ.ks * d + k] = dist;
			}
		}
		
		int ivf_id = entry_map[i];
		ivf_t entry = partial_ivf[ivf_id];
		//elements[i].tam += entry.idstam;
		
		for (int j = 0; j < entry.idstam; j++) {
			float dist = 0;
			
			for (int d = 0; d < PQ.nsq; d++) {
				dist += distab.mat[PQ.ks * d + entry.codes.mat[PQ.nsq * j + d]];
			}
			
			if (p >= dists.n)std::printf("Accessing index %d\n", p);
			dists.mat[p] = dist;
			idxs.mat[p] = entry.ids[j];
			p++;
		}
		
		delete[] distab.mat; 
	}
}

void kernel() {
	
}

//TODO: refactor and improve the whole code
//void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* entry_map, int* starting_imgid,  int* starting_inputid,  int num_imgs, query_id_t* elements, matI idxs, mat dists)
void do_on(void (*target)(pqtipo, mat, ivf_t*, int, int*, int*, int*,  int, matI, mat),
		pqtipo PQ, std::list<int>& toX, mat residual, int* coaidx, ivf_t* ivf, bool preselection, query_id_t*& elements, matI& idxs, mat& dists) {
	std::set<int> coaidPresent;

	int D = residual.d;

	mat partial_residual;
	partial_residual.n = toX.size();
	partial_residual.d = residual.d;
	partial_residual.mat = new float[partial_residual.d * partial_residual.n];

	elements = new query_id_t[toX.size()];

	int i;
	i = 0;

	for (auto it = toX.begin(); it != toX.end(); it++, i++) {
		coaidPresent.insert(coaidx[*it]);

		elements[i].id = *it; 
		elements[i].tam = 0;

		for (int d = 0; d < D; d++) {
			partial_residual.mat[i * D + d] = residual.mat[*it * D + d];
		}
	}
	
	std::cout << "ALIVE1\n";

	ivf_t partial_ivf[coaidPresent.size()];
	std::map<int, int> coaid_to_IVF;

	i = 0;
	for (auto it = coaidPresent.begin(); it != coaidPresent.end(); it++, i++) {
		partial_ivf[i].idstam = ivf[*it].idstam;
		partial_ivf[i].ids = ivf[*it].ids;
		partial_ivf[i].codes = ivf[*it].codes;
		coaid_to_IVF.insert(std::pair<int, int>(*it, i));
	}

	
	int entry_map[toX.size()];

	int starting_imgid[partial_residual.n];
	int starting_inputid[partial_residual.n + 1];
	
	int imgid = 0;
	int num_imgs = 0;
	int count = 0;
	
	i = 0;
	for (auto it = toX.begin(); it != toX.end(); ++it, i++) {
		entry_map[i] = coaid_to_IVF.find(coaidx[*it])->second;
		starting_imgid[i] = count;
		starting_inputid[i] = num_imgs;
		
		int size = partial_ivf[entry_map[i]].idstam;
		
		elements[i].tam += preselection ? min(size, PQ.ks) : size;
		count += preselection ? std::min(size, PQ.ks) : size;
		num_imgs += size;
	}
	
	
	starting_inputid[residual.n] = num_imgs;

	idxs.mat = new int[count];
	idxs.n = count;
	dists.mat = new float[count];
	dists.n = count;
	
	//void core_gpu(pqtipo PQ, mat residual, ivf_t* ivf, int ivf_size, int* entry_map, int* starting_imgid,  int* starting_inputid,  int num_imgs, query_id_t* elements, matI idxs, mat dists)
	if (partial_residual.n >= 1) (*target)(PQ, partial_residual, partial_ivf, coaidPresent.size(), entry_map, starting_imgid, starting_inputid, num_imgs, idxs, dists);
	
	delete[] partial_residual.mat;
}


void do_cpu(pqtipo PQ, std::list<int>& to_cpu, mat residual, int* coaidx, ivf_t* ivf, query_id_t*& elements, matI& idxs, mat& dists) {
	do_on(&core_cpu, PQ, to_cpu, residual, coaidx, ivf, false, elements, idxs, dists);
}

void do_gpu(pqtipo PQ, std::list<int>& to_gpu, mat residual, int* coaidx, ivf_t* ivf,  query_id_t*& elements, matI& idxs, mat& dists) {
	do_on(&core_gpu, PQ, to_gpu, residual, coaidx, ivf, true, elements, idxs, dists);
}



/*
 * One very strong assumption is that the elements that refers to the same query (but in different coarse centroids) are in order, and in gpu >>>> cpu preference.
 * Another assumption is that when we receive a (coaidx, residual) pair, that all of the w coaidx that refers to the same query are together
 */
void merge_results(int base_id, int w, int ncpu, query_id_t* cpu_elements, matI cpu_idxs, mat cpu_dists, int ngpu, query_id_t* gpu_elements, matI gpu_idxs, mat gpu_dists, query_id_t*& elements, matI& idxs, mat& dists) {
	//TODO: if we join merge_results with choose_best, we can reduce memory consumption, since we dont have to store both
	int nq = (ncpu + ngpu) / w;                   
	elements = new query_id_t[nq];
	
	for (int i = 0; i < nq; i++) {
		elements[i].tam = 0;
	}
	
	for (int i = 0; i < ngpu; i++) {
		gpu_elements[i].id = gpu_elements[i].id / w + base_id;
	}
	
	for (int i = 0; i < ncpu; i++) {
		cpu_elements[i].id = cpu_elements[i].id / w + base_id;
	}
	
	idxs.n = cpu_idxs.n + gpu_idxs.n;
	idxs.mat = new int[idxs.n];
	
	dists.n = cpu_dists.n + gpu_dists.n;
	dists.mat = new float[dists.n];
	
	int imgi = 0;
	int img_gpui = 0;
	int img_cpui = 0;
	int gpui = 0;
	int cpui = 0;

	for (int i = 0; i < nq; i++) {
		int gpu_id = gpui < ngpu ? gpu_elements[gpui].id : -1;
		int cpu_id = cpui < ncpu ? cpu_elements[cpui].id : -1;
		
		int id = gpu_id;
		if (id == -1 || cpu_id != -1 && cpu_id < gpu_id) id = cpu_id;
		
		elements[i].tam = 0;
		elements[i].id = id;
		
		while (gpui < ngpu && gpu_elements[gpui].id == id) {
			elements[i].tam += gpu_elements[gpui].tam;
			
			for (int j = 0; j < gpu_elements[gpui].tam; j++) {
				idxs.mat[imgi] = gpu_idxs.mat[img_gpui];
				dists.mat[imgi] = gpu_dists.mat[img_gpui];
				imgi++;
				img_gpui++; 
			}
			
			gpui++;
		}

		while (cpui < ncpu && cpu_elements[cpui].id == id) {
			elements[i].tam += cpu_elements[cpui].tam;

			for (int j = 0; j < cpu_elements[cpui].tam; j++) {
				idxs.mat[imgi] = cpu_idxs.mat[img_cpui];
				dists.mat[imgi] = cpu_dists.mat[img_cpui];
				imgi++;
				img_cpui++;
			}

			cpui++;
		}
	}
}


void choose_best(query_id_t* elements, int ne, matI& idxs, mat& dists, int k) {
	std::printf("k is %d\n", k);
	
	int imgi = 0;
	int wi = 0;
	
	for (int i = 0; i < ne; i++) {
		std::priority_queue<std::pair<float, int>, 
		                    std::vector<std::pair<float, int>>, 
							std::less<std::pair<float, int>>> queue;
		
		for (int j = 0; j < elements[i].tam; j++) {
			//std::printf("%d elements\n", elements[i].tam);
			
			if (queue.size() < k) queue.push(std::pair<float, int>(dists.mat[imgi], idxs.mat[imgi]));
			else if (dists.mat[imgi] < queue.top().first) {
				queue.pop();
				queue.push(std::pair<float, int>(dists.mat[imgi], idxs.mat[imgi]));
			}
			
			imgi++;
		}
		
		elements[i].tam = queue.size();
		while (queue.size() >= 1) {
			idxs.mat[wi] = queue.top().second;
			dists.mat[wi] = queue.top().first;
			queue.pop();
			wi++;
		}
	}
	
	idxs.n = wi;
	dists.n = wi;
}

void send_results(int ne, query_id_t* elements, matI idxs, mat dists, int finish) {
	int counter = 0;
	
	int my_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 1, MPI_COMM_WORLD);
	MPI_Send(&ne, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
	MPI_Send(elements, sizeof(query_id_t) * ne, MPI_BYTE, last_aggregator,
			0, MPI_COMM_WORLD);
	
	MPI_Send(idxs.mat, idxs.n, MPI_INT, last_aggregator, 0,
	MPI_COMM_WORLD);
	MPI_Send(dists.mat, dists.n, MPI_FLOAT, last_aggregator, 0,
	MPI_COMM_WORLD);
	MPI_Send(&finish, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD); 
}

void parallel_search (int nsq, int k, int comm_sz, int threads, int tam, MPI_Comm search_comm, char *dataset, int w){
	ivfpq_t ivfpq;
	mat residual;
	int *coaidx, my_rank;
	//double time;

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	set_last (comm_sz, &last_assign, &last_search, &last_aggregator);

	//Recebe os centroides
	MPI_Recv(&ivfpq, sizeof(ivfpq_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.pq.centroids = (float*)malloc(sizeof(float)*ivfpq.pq.centroidsn*ivfpq.pq.centroidsd);
	MPI_Recv(&ivfpq.pq.centroids[0], ivfpq.pq.centroidsn*ivfpq.pq.centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	ivfpq.coa_centroids=(float*)malloc(sizeof(float)*ivfpq.coa_centroidsd*ivfpq.coa_centroidsn);
	MPI_Recv(&ivfpq.coa_centroids[0], ivfpq.coa_centroidsn*ivfpq.coa_centroidsd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
	std::cout << "number of coarse centroids: " << ivfpq.coa_centroidsn << "\n";
	std::cout << "number of product centroids per dimension: " << ivfpq.pq.centroidsn << "\n";
	std::cout << "number of product centroids dimensions: " << ivfpq.pq.centroidsd << "\n";


	ivf_t *ivf, *ivf2;

	#ifdef WRITE_IVF
		write_ivf(ivfpq, threads, tam, my_rank, nsq, dataset);

		ivf = read_ivf(ivfpq, tam, my_rank);
	#else
		#ifdef READ_IVF		
			ivf = read_ivf(ivfpq, tam, my_rank);
			
		#else
			ivf = create_ivf(ivfpq, threads, tam, my_rank, nsq, dataset);
		#endif
	#endif

	float **dis;
	int **ids;
	int finish_aux=0;

	query_id_t *fila;
	int count = 0;

	
	int base_id = 0; // corresponds to the query_id
	
	MPI_Barrier(search_comm);

	sem_init(&sem, 0, 1);
	
	system("exit");

	while (1) {
		double f1 = 0, f2 = 0, f3 = 0, f4 = 0, g1 = 0, g2 = 0, g3 = 0;

		MPI_Bcast(&residual.n, 1, MPI_INT, 0, search_comm);
		MPI_Bcast(&residual.d, 1, MPI_INT, 0, search_comm);

		residual.mat = (float*) malloc(sizeof(float) * residual.n * residual.d);

		MPI_Bcast(&residual.mat[0], residual.d * residual.n, MPI_FLOAT, 0,
				search_comm);

		coaidx = (int*) malloc(sizeof(int) * residual.n);

		MPI_Bcast(&coaidx[0], residual.n, MPI_INT, 0, search_comm);
		MPI_Bcast(&finish_aux, 1, MPI_INT, 0, search_comm);

		fila = (query_id_t*) malloc(sizeof(query_id_t) * (residual.n / w));

		for (int it = 0; it < residual.n / w; it++) {
			fila[it].tam = 0;
			fila[it].id = count;
		}

		iter = 0;

		dis = (float**) malloc(sizeof(float *) * (residual.n / w));
		ids = (int**) malloc(sizeof(int *) * (residual.n / w));
		
		int maxgpu = residual.n / 2;
		
		
		std::list<int> to_gpu;
		std::list<int> to_cpu;
		
		//TODO: implement the real logic to separate gpu and cpu
		for (int i = 0; i < residual.n; i++) {
			to_gpu.push_back(i);
		}
		
		std::printf("EXECUTING ON THE %s\n", to_cpu.size() == 0 ? "gpu" : "cpu");
		
		time_t start,end;
		time (&start);
		
		
		//GPU PART
		query_id_t* gpu_elements;
		matI gpu_idxs;
		mat gpu_dists;
		sw(do_gpu(ivfpq.pq, to_gpu, residual, coaidx, ivf, gpu_elements, gpu_idxs, gpu_dists));
		
		//CPU PART
		query_id_t* cpu_elements;
		matI cpu_idxs;
		mat cpu_dists;
		std::cout << "DO_CPU started\n";
		sw(do_cpu(ivfpq.pq, to_cpu, residual, coaidx, ivf, cpu_elements, cpu_idxs, cpu_dists));
		std::cout << "DO_CPU ended\n";
		
		query_id_t* elements;
		matI idxs;
		mat dists;
		
		std::cout << "merge_results started\n";
		sw(merge_results(base_id, w, to_cpu.size(), cpu_elements, cpu_idxs, cpu_dists, to_gpu.size(), gpu_elements, gpu_idxs, gpu_dists, elements, idxs, dists));
		std::cout << "merge_results ended\n";
		
		delete[] cpu_elements;
		delete[] cpu_idxs.mat;
		delete[] cpu_dists.mat;
		
		delete[] gpu_elements;
		delete[] gpu_idxs.mat;
		delete[] gpu_dists.mat;
		
		std::cout << "choose_best started\n";
		sw(choose_best(elements, residual.n / w, idxs, dists, k));
		std::cout << "choose_best ended\n";
		
		sw(send_results((to_cpu.size() + to_gpu.size()) / w, elements, idxs, dists, finish_aux));
		
		delete[] elements;
		delete[] idxs.mat;
		delete[] dists.mat;
		
		base_id += residual.n / w;
		
		
		time (&end);
		double dif = difftime (end,start);
		printf ("Elapsed time is %.2lf seconds.", dif );
				
				
		if (finish_aux == 1) break;
		
		std::cout << "ABORTTTTTTTTTT THIS SHIT\n";
	}
	
	std::cout << "GOT OUT OF HERE\n";
	cout << "." << endl;
	sem_destroy(&sem);
	free(ivf);
	free(ivfpq.pq.centroids);
	free(ivfpq.coa_centroids);
	
	std::cout << "FINISHED THE SEARCH\n";
}

dis_t ivfpq_search(ivf_t *ivf, float *residual, pqtipo pq, int centroid_idx, double *g1, double *g2){
	dis_t q;
	int ds, ks, nsq;
	struct timeval b1, b2, b3;

	ds = pq.ds;
	ks = pq.ks;
	nsq = pq.nsq;

	mat distab;
	distab.mat = (float*)malloc(sizeof(float)*ks*nsq);
	distab.n = nsq;
	distab.d = ks;

	float *distab_temp=(float*)malloc(sizeof(float)*ks);

	float* AUXSUMIDX;

	q.dis.n = ivf[centroid_idx].codes.n;
	q.dis.d = 1;
	q.dis.mat = (float*)malloc(sizeof(float)*q.dis.n);

	q.idx.n = ivf[centroid_idx].codes.n;
	q.idx.d = 1;
	q.idx.mat = (int*)malloc(sizeof(int)*q.idx.n);

	gettimeofday(&b1, NULL);

	clock_t begin = clock();
	for (int query = 0; query < nsq; query++) {
		
		compute_cross_distances(ds, 1, distab.d, &residual[query*ds], &pq.centroids[query*ks*ds], distab_temp);
		
		memcpy(distab.mat+query*ks, distab_temp, sizeof(float)*ks);
	}

	gettimeofday(&b2, NULL);

	begin = clock();
	AUXSUMIDX = sumidxtab2(distab, ivf[centroid_idx].codes, 0);

	gettimeofday(&b3, NULL);

	memcpy(q.idx.mat, ivf[centroid_idx].ids,  sizeof(int)*ivf[centroid_idx].idstam);
	memcpy(q.dis.mat, AUXSUMIDX, sizeof(float)*ivf[centroid_idx].codes.n);

	free (AUXSUMIDX);
	free (distab_temp);
	free (distab.mat);

	*g1 += ((b2.tv_sec * 1000000 + b2.tv_usec)-(b1.tv_sec * 1000000 + b1.tv_usec));
	*g2 += ((b3.tv_sec * 1000000 + b3.tv_usec)-(b2.tv_sec * 1000000 + b2.tv_usec));

	return q;
}

void send_aggregator(int residualn, int w, query_id_t *fila, int **ids,
		float **dis, int finish_aux, int count) {
	int i = 0, num = 0, ttam = 0, *ids2, it = 0, my_rank, finish = 0;
	float *dis2;
	query_id_t *element;

	ids2 = (int*) malloc(sizeof(int));
	dis2 = (float*) malloc(sizeof(float));
	element = (query_id_t*) malloc(sizeof(query_id_t) * 10);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	while (i < residualn / w) {
		element[num] = fila[it];

		ids2 = (int*) realloc(ids2, sizeof(int) * (ttam + element[num].tam));
		dis2 = (float*) realloc(dis2,
				sizeof(float) * (ttam + element[num].tam));
		memcpy(&ids2[0] + ttam, ids[element[num].id - count],
				sizeof(int) * element[num].tam);
		memcpy(&dis2[0] + ttam, dis[element[num].id - count],
				sizeof(float) * element[num].tam);
		free(ids[element[num].id - count]);
		free(dis[element[num].id - count]);
		ttam += element[num].tam;
		num++;
		it++;
		if (num == 10 || num == residualn / w - i) {

			if (num == residualn / w - i && finish_aux == 1)
				finish = 1;

			MPI_Send(&my_rank, 1, MPI_INT, last_aggregator, 1, MPI_COMM_WORLD);
			MPI_Send(&num, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD);
			MPI_Send(&element[0], sizeof(query_id_t) * num, MPI_BYTE,
					last_aggregator, 0, MPI_COMM_WORLD);
			
			MPI_Send(&ids2[0], ttam, MPI_INT, last_aggregator, 0,
					MPI_COMM_WORLD);
			MPI_Send(&dis2[0], ttam, MPI_FLOAT, last_aggregator, 0,
					MPI_COMM_WORLD);
			MPI_Send(&finish, 1, MPI_INT, last_aggregator, 0, MPI_COMM_WORLD); //TODO: why is finish sent and not finish_aux (since it ends the loop on the main function)?

			i += num;
			num = 0;
			ttam = 0;
		}
	}
	free(element);
	free(ids2);
	free(dis2);
}

ivf_t* create_ivf(ivfpq_t ivfpq, int threads, int tam, int my_rank, int nsq, char* dataset){
	ivf_t *ivf;
	struct timeval start, end;
	double time;
	int lim;

	printf("\nIndexing\n");

	gettimeofday(&start, NULL);

	ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);
	for(int i=0; i<ivfpq.coarsek; i++){
		ivf[i].ids = (int*)malloc(sizeof(int));
		ivf[i].idstam = 0;
		ivf[i].codes.mat = (int*)malloc(sizeof(int));
		ivf[i].codes.n = 0;
		ivf[i].codes.d = nsq;
	}
	lim = tam/1000000;
	if(tam%1000000!=0){
		lim = (tam/1000000) + 1;
	}

	//Cria a lista invertida correspondente ao trecho da base assinalado a esse processo
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
		for(int i=0; i<lim; i++){

			ivf_t *ivf2;
			int aux;
				mat vbase;
			ivf2 = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);

			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);

			ivfpq_assign(ivfpq, vbase, ivf2);

			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf2[j].idstam; l++){
					ivf2[j].ids[l]+=1000000*i+tam*(my_rank-last_assign-1);
				}

				aux = ivf[j].idstam;
				#pragma omp critical
				{
					ivf[j].idstam += ivf2[j].idstam;
					ivf[j].ids = (int*)realloc(ivf[j].ids,sizeof(int)*ivf[j].idstam);
					memcpy (ivf[j].ids+aux, ivf2[j].ids, sizeof(int)*ivf2[j].idstam);
					ivf[j].codes.n += ivf2[j].codes.n;
					ivf[j].codes.mat = (int*)realloc(ivf[j].codes.mat,sizeof(int)*ivf[j].codes.n*ivf[j].codes.d);
					memcpy (ivf[j].codes.mat+aux*ivf[i].codes.d, ivf2[j].codes.mat, sizeof(int)*ivf2[j].codes.n*ivf2[j].codes.d);
				}
				free(ivf2[j].ids);
				free(ivf2[j].codes.mat);
			}
			free(vbase.mat);
			free(ivf2);
		}

	gettimeofday(&end, NULL);
	time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec))/1000;

	printf ("\nTempo de criacao da lista invertida: %g\n",time);

	return ivf;
}

void write_ivf(ivfpq_t ivfpq, int threads, int tam, int my_rank, int nsq, char* dataset){
	FILE *fp;
	char name_arq[100];
	struct timeval start, end;
	double time;
	int lim, i;

	printf("\nIndexing\n");

	gettimeofday(&start, NULL);

	lim = tam/1000000;
	if(tam%1000000!=0){
		lim = (tam/1000000) + 1;
	}

	//Cria a lista invertida correspondente ao trecho da base assinalado a esse processo
	#pragma omp parallel for num_threads(threads) schedule(dynamic)
		for(i=0; i<lim; i++){

			ivf_t *ivf;
			int aux;
			mat vbase;
			ivf = (ivf_t *)malloc(sizeof(ivf_t)*ivfpq.coarsek);

			vbase = pq_test_load_base(dataset, i, my_rank-last_assign);

			ivfpq_assign(ivfpq, vbase, ivf);

			free(vbase.mat);

			for(int j=0; j<ivfpq.coarsek; j++){
				for(int l=0; l<ivf[j].idstam; l++){
					ivf[j].ids[l]+=1000000*i+tam*(my_rank-last_assign-1);
				}

				aux = ivf[j].idstam;
				#pragma omp critical
				{	
					sprintf(name_arq, "/pylon5/ac3uump/freire/ivf/ivf_%d_%d_%d.bin", ivfpq.coarsek, tam, j);
					fp = fopen(name_arq,"ab");
					fwrite(&ivfpq.coarsek, sizeof(int), 1, fp);
					fwrite(&ivf[j].idstam, sizeof(int), 1, fp);
					fwrite(&ivf[j].ids[0], sizeof(int), ivf[j].idstam, fp);
					fwrite(&ivf[j].codes.n, sizeof(int), 1, fp);
					fwrite(&ivf[j].codes.d, sizeof(int), 1, fp);
					fwrite(&ivf[j].codes.mat[0], sizeof(int), ivf[j].codes.n*ivf[j].codes.d, fp);
					fclose(fp);
				}
				free(ivf[j].ids);
				free(ivf[j].codes.mat);
			}
			free(ivf);
		}

	gettimeofday(&end, NULL);
	time = ((end.tv_sec * 1000000 + end.tv_usec)-(start.tv_sec * 1000000 + start.tv_usec))/1000;

	printf ("\nTempo de criacao da lista invertida: %g\n",time);
}

ivf_t* read_ivf(ivfpq_t ivfpq, int tam, int my_rank){

	ivf_t* ivf;
	FILE *fp;
	char name_arq[100];
	int coarsek;

	ivf = (ivf_t*)malloc(sizeof(ivf_t)*ivfpq.coarsek);

	for(int i=0; i<ivfpq.coarsek; i++){
		int idstam, codesn, codesd;
		
		ivf[i].ids = (int*)malloc(sizeof(int));
		ivf[i].idstam = 0;
		ivf[i].codes.mat = (int*)malloc(sizeof(int));
		ivf[i].codes.n = 0;
		ivf[i].codes.d = ivfpq.pq.nsq;
		
		sprintf(name_arq, "/pylon5/ac3uump/freire/ivf/ivf_%d_%d_%d.bin", ivfpq.coarsek, tam, i);
		fp = fopen(name_arq,"rb");
		
		for(int j=0; j<tam/1000000; j++){
			fread(&coarsek, sizeof(int), 1, fp);
			fread(&idstam, sizeof(int), 1, fp);
			ivf[i].idstam += idstam;
			ivf[i].ids = (int*)realloc(ivf[i].ids,sizeof(int)*ivf[i].idstam);
			fread(&ivf[i].ids[ivf[i].idstam-idstam], sizeof(int), idstam, fp);
			fread(&codesn, sizeof(int), 1, fp);
			ivf[i].codes.n += codesn;
			fread(&codesd, sizeof(int), 1, fp);
			ivf[i].codes.d = codesd;
			ivf[i].codes.mat = (int*)realloc(ivf[i].codes.mat,sizeof(int)*ivf[i].codes.n*ivf[i].codes.d);
			fread(&ivf[i].codes.mat[((ivf[i].codes.n)*(ivf[i].codes.d))-codesn*codesd], sizeof(int), codesn*codesd, fp);
		}
		fclose(fp);
	}

	return ivf;
}

int min(int a, int b){
	if(a>b){
		return b;
	}
	else
		return a;
}



float * sumidxtab2(mat D, matI ids, int offset){	
	static int max = 0;
	
	//aloca o vetor a ser retornado

	float *dis = (float*)malloc(sizeof(float)*ids.n);
	int i, j;

	//soma as distancias para cada vetor

	for (i = 0; i < ids.n ; i++) {
		float dis_tmp = 0;
		for(j=0; j<D.n; j++){
			dis_tmp += D.mat[j*D.d + ids.mat[i*ids.d+j]+ offset];
			
			if (ids.mat[i*ids.d+j] > max) max = ids.mat[i*ids.d+j];
		}
		dis[i]=dis_tmp;
	}

	return dis;
}

int* imat_new_transp (const int *a, int ncol, int nrow){
	int i,j;
	int *vt=(int*)malloc(sizeof(int)*ncol*nrow);

	for(i=0;i<ncol;i++)
		for(j=0;j<nrow;j++)
			vt[i*nrow+j]=a[j*ncol+i];

	return vt;
}
