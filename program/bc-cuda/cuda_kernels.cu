#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/time.h>
#include <cuda.h>
//#include <mpi.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include "cudamacro.h"
#include "bc2d.h"

// BEST  DRAKE 128 1 2 
// BEST PizDaint 2 2  256
#define THREADS (128)
#define ROWXTH 2
#define ROWXTHD 1
__device__ __constant__ LOCINT dN;
__device__ __constant__ LOCINT drow_bl;
__device__ __constant__ LOCINT dcol_bl;
__device__ __constant__ LOCINT drow_pp;

__device__ __constant__ int dC;
__device__ __constant__ int dR;
__device__ __constant__ int dmyrow;
__device__ __constant__ int dmycol; 

__device__ LOCINT dnfrt;

__device__ LOCINT d_reach_v0; 



static LOCINT	*d_msk=NULL;
static int	*d_lvl=NULL;

static LOCINT	*d_col=NULL;
static LOCINT	*d_row=NULL;
static LOCINT	*d_deg=NULL;

static LOCINT	*d_rbuf=NULL;
static LOCINT	*d_cbuf=NULL;
static LOCINT	*d_cbuf_start=NULL;



static LOCINT	*d_sbuf=NULL;
static uint32_t	*d_snum=NULL;

static LOCINT   *d_frt=NULL;
static LOCINT   *d_frt_start=NULL;
static LOCINT   *d_frt_sig=NULL;


static LOCINT   *d_sig=NULL;

static LOCINT   *d_tmp_sig=NULL;
static LOCINT	*d_rbuf_sig=NULL;
static LOCINT	*d_sbuf_sig=NULL;

static float    *d_delta=NULL;

static float    *d_fsbuf=NULL;
static float    *d_frbuf=NULL;
static float 	*d_bc=NULL;
static LOCINT   *d_reach= NULL;
static LOCINT   *d_all = NULL;

cudaEvent_t     start, stop;
cudaStream_t    stream[2];






FILE *Fopen(const char *path, const char *mode) {

	FILE *fp = NULL;
	fp = fopen(path, mode);
	if (!fp) {
		fprintf(stderr, "Cannot open file %s...\n", path);
		exit(EXIT_FAILURE);
	}
	return fp;
}

void dump_device_array(const char *name, LOCINT *d_arr, int n) {

	FILE	 *fp=NULL;
	char	 fname[MAX_LINE];
	int i;
	LOCINT *in;

	snprintf(fname, MAX_LINE, "%s_%d", name, myid);
	fp = Fopen(fname, "a");

	in = (LOCINT *)Malloc(n*sizeof(*in));
	MY_CUDA_CHECK( cudaMemcpy(in, d_arr, n*sizeof(*in), cudaMemcpyDeviceToHost) );

	for (i = 0; i < n ; i++)
	   fprintf(fp, " %d,", in[i]);

	fprintf(fp, "\n");
	fclose(fp);
	free(in);
	return;

}

void dump_array2(int *arr, int n, const char *name) {

	if (outdebug==NULL) return;
	int i;
	fprintf(outdebug, "%s - %d\n",name, n);

	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %d,", arr[i]);

	fprintf(outdebug, "\n");
	return;

}

void dump_uarray2(LOCINT *arr, int n, const char *name) {

	if (outdebug==NULL) return;
	int i;
	fprintf(outdebug, "%s - %d\n",name, n);

	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %d,", arr[i]);

	fprintf(outdebug, "\n");
	return;
}

void dump_farray2(float *arr, int n, const char *name) {

	if (outdebug==NULL) return;
	int i;
	fprintf(outdebug, "%s - %d\n",name, n);
	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %f,", arr[i]);

	fprintf(outdebug, "\n");
	return;

}

void dump_device_array2(int *d_arr, int n, const char * name) {

	if (outdebug==NULL) return;
	int i;
	int *in;

	fprintf(outdebug, "%s - %d\n",name, n);

	in = (int *)Malloc(n*sizeof(*in));
	MY_CUDA_CHECK( cudaMemcpy(in, d_arr, n*sizeof(*in), cudaMemcpyDeviceToHost) );

	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %d,", in[i]);

	fprintf(outdebug, "\n");
	fflush(outdebug);
	free(in);
	return;

}

void dump_device_uarray2(LOCINT *d_arr, int n, const char * name) {

	if (outdebug==NULL) return;
	int i;
	LOCINT *in;

	fprintf(outdebug, "%s - %d\n",name, n);

	in = (LOCINT *)Malloc(n*sizeof(*in));
	MY_CUDA_CHECK( cudaMemcpy(in, d_arr, n*sizeof(*in), cudaMemcpyDeviceToHost) );

	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %d,", in[i]);

	fprintf(outdebug, "\n");
	fflush(outdebug);
	free(in);
	return;

}

void dump_device_farray2(float *d_arr, int n, const char * name) {

	if (outdebug==NULL) return;
	int i;
	float *in;

	fprintf(outdebug, "%s - %d\n",name, n);

	in = (float *)Malloc(n*sizeof(*in));
	MY_CUDA_CHECK( cudaMemcpy(in, d_arr, n*sizeof(*d_arr), cudaMemcpyDeviceToHost) );

	for (i = 0; i < n ; i++)
	   fprintf(outdebug, " %f,", in[i]);

	fprintf(outdebug, "\n");
	fflush(outdebug);
	free(in);
	return;

}


// returns the index of the maximum i | v[i] <= val
__device__ LOCINT bmaxlt(const LOCINT *__restrict__ v, LOCINT num, LOCINT val) {

	LOCINT	min = 0;
	LOCINT	max = num-1;
	LOCINT	mid = max >> 1;

	while(min <= max) {

		if (v[mid] == val)	return mid;
		if (v[mid]  < val)	min = mid+1;
		else			max = mid-1;
		mid = (max>>1)+(min>>1)+((min&max)&1); //(max + min) >> 1
	}
	return mid;
} 
	
__global__ void read_edge_count(const LOCINT *__restrict__ deg, const LOCINT *__restrict__ rbuf, LOCINT n, LOCINT *cbuf) {
	
	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;
	cbuf[tid] = deg[rbuf[tid]];
	return;
}

/*
 * write_sigma (d_sbuf+i*ld, d_sig, d_tmp_sig, snum[i], d_sbuf_sig+i*ld);
 */
__global__ void write_sigma(const LOCINT *__restrict__ sbuf, const LOCINT *__restrict__ sigma,
		                    LOCINT * tmp_sig, LOCINT n, LOCINT *sbuf_sig) {
	
	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;

	sbuf_sig[tid] = sigma[sbuf[tid]] + tmp_sig[sbuf[tid]]; // Calculate the total sigma and prepare for sending
	tmp_sig[sbuf[tid]] = 0; // So we already have the array zero for next round

	return;
}



__global__ void update_bc(const float *__restrict__ delta, int r0, LOCINT n,  float *bc, LOCINT *reach, const uint64_t nvisited) {

	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;
	
	if (r0 == tid)  {
		if (d_reach_v0 > 0) bc[tid] = (bc[tid])+ (d_reach_v0)*(nvisited-2);
		return;
	}

    	bc[tid] = (bc[tid]) +  delta[tid]*(d_reach_v0 + 1);

	return;
}


void update_bc_cuda(uint64_t v0, int ncol, const uint64_t __restrict__ nvisited) {
	int r0 = -1;
	if (GI2PI(v0) == myrow) {
		r0 = GI2LOCI(v0);
	}
	update_bc<<<(ncol+THREADS-1)/THREADS, THREADS>>>(d_delta, r0, ncol, d_bc, d_reach, nvisited);
}


void sort_by_degree(LOCINT *deg, LOCINT *bc_order){

	thrust::sort_by_key(deg, deg + N, bc_order);
}

__inline__ __device__	int warpReduceSum(int val) {
	for (int offset = warpSize/2; offset > 0; offset /= 2) 
    		val += __shfl_down(val, offset);
	return val;
}

__inline__ __device__ int blockReduceSum(int val) {

 	static __shared__ int shared[32]; 	
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;
  	val = warpReduceSum(val); 
  	if (lane==0) shared[wid]=val;	  __syncthreads();              
  	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  	if (wid==0) val = warpReduceSum(val); //Final reduce within first warp
  	return val;
}

__global__ void deviceReduceKernel(const LOCINT *__restrict__ in, LOCINT* out, int N, const int * __restrict__ cond) {
	LOCINT sum = 0;
	const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = tid; i < N; i += THREADS/4 * THREADS/4) {
		//if (cond[i] > 0) sum+= in[i];
    		sum += in[i]*(cond[i] > 0 );
//		p = in[i];
//		if ( cond[i] == -1 ) p=0;
//               sum += p;


	}
	sum = blockReduceSum(sum);
	if (threadIdx.x == 0)atomicAdd(out, sum);
}



void pre_update_bc_cuda(LOCINT *reach, uint64_t v0, LOCINT *all){


	cudaMemsetAsync(d_all,0,sizeof(LOCINT));
  	deviceReduceKernel<<<THREADS/4, THREADS/4>>>(d_reach, d_all, row_pp, d_lvl);
	cudaMemcpy(all,d_all,sizeof(int),cudaMemcpyDeviceToHost);
}

__global__ void write_delta(const LOCINT *__restrict__ frt, const LOCINT *__restrict__ sigma,
		                    const LOCINT *__restrict__ reach,
						   float  *rbuf, LOCINT n,  float *sbuf) {

	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	LOCINT i;
	if (tid >= n) return;

	if (CUDA_ISMYCOLL(frt[tid])) {
		// No race condition since a node appears only ones in the frontier
		// Calculate delta only for my own vertices
		// Here delta is updated using row index
		i = CUDA_MYLOCJ2LOCI(frt[tid]);
		//sbuf[i] = rbuf[tid] * sigma[i] + reach[i]; // add reach[i]
                sbuf[i] = rbuf[tid] * sigma[i];

	    // Copy back the value into the send-receive buffer
	    //srbuf[tid] = delta[i] ;
	}
	rbuf[tid] = 0;

}

LOCINT write_delta_cuda(LOCINT ncol, float *hRFbuf, float *hSFbuf) {

	float	et=0;
	TIMER_DEF(1);

	TIMER_START(1);

	// Reset send buffer

	MY_CUDA_CHECK( cudaMemset(d_fsbuf, 0, row_pp*sizeof(*d_fsbuf)) );
	if (!ncol) {
		TIMER_STOP(1);
		goto out;
	}
	// Copy receive buffer into device memory
	MY_CUDA_CHECK( cudaMemcpy(d_frbuf, hRFbuf , ncol*sizeof(*hRFbuf), cudaMemcpyHostToDevice ));

	TIMER_STOP(1);

	MY_CUDA_CHECK( cudaEventRecord(start, 0) );

	// READ_DFRT
	write_delta<<<(ncol+THREADS-1)/THREADS, THREADS>>>(d_rbuf, d_sig, d_reach, d_frbuf, ncol, d_fsbuf);

	// Here we have d_delta updated
	MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
	MY_CHECK_ERROR("write_delta");
	MY_CUDA_CHECK( cudaEventSynchronize(stop) );
	MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );
out:
	MY_CUDA_CHECK( cudaMemcpy(hSFbuf, d_fsbuf, MAX(row_pp,col_bl)*sizeof(*hSFbuf), cudaMemcpyDeviceToHost ));

   return ncol;
}



__global__ void scan_col_mono(const LOCINT *__restrict__ row,  const LOCINT *__restrict__ col, LOCINT nrow, 
                                const LOCINT *__restrict__ rbuf, 
                                const LOCINT *__restrict__ cbuf, LOCINT ncol, 
                                LOCINT *msk, int *lvl, LOCINT* sig, int level, 
                                LOCINT *sbuf, uint32_t *snum) {

	// This processes ROWXTH elements together
	LOCINT r[ROWXTH];
	LOCINT c[ROWXTH]; // Vertex in the current frontier
	LOCINT s[ROWXTH]; // Sigma of the vertex in the current frontier
	LOCINT m[ROWXTH], q[ROWXTH], i[ROWXTH];

	const uint32_t tid = (blockDim.x*blockIdx.x + threadIdx.x)*ROWXTH;

	if (tid >= nrow) return;

	// Use binary search to calculate predecessor position in the rbuf array
	i[0] = bmaxlt(cbuf, /*(tid<ncol)?tid+1:*/ncol, tid);

	for(; (i[0]+1 < ncol) && (tid+0) >= cbuf[i[0]+1]; i[0]++); // Here increment i[0]
	#pragma unroll
	for(int k = 1; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		for(i[k]=i[k-1]; (i[k]+1 < ncol) && (tid+k) >= cbuf[i[k]+1]; i[k]++); // Here increment i[k]
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		c[k] = (rbuf[i[k]]); 
		s[k] = (sig[c[k]]); 
	} //c[k] is the predecessor, s[k] is its sigma

	// Here r[k] corresponds to the row and from it I can determine the processor hproc
	// col[c[k]] offset in the CSC where neightbour of c[k] starts
	// row[col[c[k]] first neightbour
	// r[k] this is the visited vertex
	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		r[k] = row[col[c[k]]+(tid+k)-cbuf[i[k]]];  // new vertex
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		m[k] = ((LOCINT)1) << (r[k]%BITS(msk));  // its mask value
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		if ((msk[r[k]/BITS(msk)])&m[k]) //continue;
			q[k] = m[k]; // the if below will eval to false...
		else
			q[k] = atomicOr(msk+r[k]/BITS(msk), m[k]);

		if (!(m[k]&q[k])) {  // New vertex
			uint32_t off = atomicInc(snum, 0xFFFFFFFF); // Offset // Increment + 1
			// This is the GLOBAL VERTEX !!
			sbuf[off] = r[k]; // Copy the new discovered vertex into the sbuf for sending
  		    	lvl[r[k]] = level; // Update level
		}

		if ((lvl[r[k]]) == -1 || (lvl[r[k]]) == level) { // Order in the OR is important!			// Update sigma
            // Update sigma
			atomicAdd(sig + r[k], s[k]);
                }

	} // end for over k
	return;
}

__global__ void scan_col_mono2(const LOCINT *__restrict__ row,  const LOCINT *__restrict__ col, LOCINT nrow, 
                                const LOCINT *__restrict__ rbuf, 
                                const LOCINT *__restrict__ cbuf, LOCINT ncol, 
                                LOCINT *msk, int *lvl, LOCINT* sig, int level, 
                                LOCINT *sbuf, uint32_t *snum) {

	// This processes ROWXTH elements together
	LOCINT r[ROWXTH];
	LOCINT c[ROWXTH]; // Vertex in the current frontier
	LOCINT m[ROWXTH], q[ROWXTH], i[ROWXTH];

	const uint32_t tid = (blockDim.x*blockIdx.x + threadIdx.x)*ROWXTH;

	if (tid >= nrow) return;

	// Use binary search to calculate predecessor position in the rbuf array
	i[0] = bmaxlt(cbuf, /*(tid<ncol)?tid+1:*/ncol, tid);

	for(; (i[0]+1 < ncol) && (tid+0) >= cbuf[i[0]+1]; i[0]++); // Here increment i[0]
	#pragma unroll
	for(int k = 1; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		for(i[k]=i[k-1]; (i[k]+1 < ncol) && (tid+k) >= cbuf[i[k]+1]; i[k]++); // Here increment i[k]
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		c[k] = (rbuf[i[k]]); 
	} //c[k] is the predecessor, s[k] is its sigma

	// Here r[k] corresponds to the row and from it I can determine the processor hproc
	// col[c[k]] offset in the CSC where neightbour of c[k] starts
	// row[col[c[k]] first neightbour
	// r[k] this is the visited vertex
	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		r[k] = row[col[c[k]]+(tid+k)-cbuf[i[k]]];  // new vertex
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		m[k] = ((LOCINT)1) << (r[k]%BITS(msk));  // its mask value
	}

	#pragma unroll
	for(int k = 0; k < ROWXTH; k++) {
		if (tid+k >= nrow) break;
		if ((msk[r[k]/BITS(msk)])&m[k]) //continue;
			q[k] = m[k]; // the if below will eval to false...
		else
			q[k] = atomicOr(msk+r[k]/BITS(msk), m[k]);

		if (!(m[k]&q[k])) {  // New vertex
			uint32_t off = atomicInc(snum, 0xFFFFFFFF); // Offset // Increment + 1
			// This is the GLOBAL VERTEX !!
			sbuf[off] = r[k]; // Copy the new discovered vertex into the sbuf for sending
  		    	lvl[r[k]] = level; // Update level
		}

		const int l = (lvl[r[k]]);
		if (l == -1 || l == level) { // Order in the OR is important!			// Update sigma
            // Update sigma
			atomicAdd(sig + r[k], (sig[c[k]]));
                }

	} // end for over k
	return;
}




__global__ void scan_frt_mono(const LOCINT *__restrict__ row,   const LOCINT *__restrict__ col, LOCINT nrow,
			                  const LOCINT *__restrict__ rbuf,  const LOCINT *__restrict__ cbuf, LOCINT ncol,
			                  const LOCINT *__restrict__ sigma, float  *delta,
			                  const int *__restrict__ lvl, int depth) {

	// This processes ROWXTH elements together
	LOCINT r[ROWXTHD];
	LOCINT c[ROWXTHD]; // Vertex in the current frontier
	LOCINT i[ROWXTHD];
	float a;

	const uint32_t tid = (blockDim.x*blockIdx.x + threadIdx.x)*ROWXTHD;

	if (tid >= nrow) return;

	// Use binary search to calculate predecessor position in the rbuf array
	i[0] = bmaxlt(cbuf, ncol, tid);

	for(; (i[0]+1 < ncol) && (tid+0) >= cbuf[i[0]+1]; i[0]++); // Here increment i[0]
	#pragma unroll
	for(int k = 1; k < ROWXTHD; k++) {
		if (tid+k >= nrow) break;
		for(i[k]=i[k-1]; (i[k]+1 < ncol) && (tid+k) >= (cbuf[i[k]+1]); i[k]++); // Here increment i[k]
	}

	#pragma unroll
	for(int k = 0; k < ROWXTHD; k++) {
		if (tid+k >= nrow) break;
		c[k] = (rbuf[i[k]]);  } //c[k] is the vertex in the input buffer

	// Here r[k] corresponds to the row and from it I can determine the processor hproc
	// col[c[k]] offset in the CSC where neightbour of c[k] starts
	// row[col[c[k]] first neightbour
	// r[k] this is the visited vertex
	#pragma unroll
	for(int k = 0; k < ROWXTHD; k++) {
		if (tid+k >= nrow) break;
		r[k] = row[col[c[k]]+(tid+k)-cbuf[i[k]]];  // new vertex
	}

	#pragma unroll
	for (int k = 0; k < ROWXTHD; k++) {
		if (tid+k >= nrow) break;

		if (lvl[r[k]] == depth+1) { // this is a successor
			// sigma and delta are indexed by row
			a = ((delta[r[k]]) + 1)/sigma[r[k]]*sigma[c[k]];
			// IN SINGLE DEVICE we multiply a * sigma[c[k]]
		    	// Need to add into the SRbuffer using the same index used to access rbuf
			atomicAdd(delta+c[k], a);
		}
	} // end for over k
	return;
}

__global__ void append_row(const LOCINT *__restrict__ row,  const LOCINT *__restrict__ row_sig, LOCINT n,
			               const LOCINT *__restrict__ cbuf, LOCINT np,
			               LOCINT *msk,  const LOCINT * __restrict__ reach, int *lvl,
			               int level, LOCINT *frt, LOCINT *tmp_sig, LOCINT * frt_sig, uint32_t *all) {

	LOCINT	 r, m, q, s;
	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;

	r = row[tid];
	s = row_sig[tid];

	m = ((LOCINT)1) << (r%BITS(msk));

	if (!(msk[r/BITS(msk)]&m)) {  // Check if the vertex was already visited
		q = atomicOr(msk+r/BITS(msk), m);  // Mark visited
		if (!(m&q)) { // Check if the vertex was already visited
			uint32_t off = atomicInc(&dnfrt, 0xFFFFFFFF);
			frt[off] = r;  // Still Global
			frt_sig[off] = 0;
			lvl[r] = level;
		}
	}

	if (lvl[r] == level || lvl[r] == -1) {
		// Update sigma with the value provided
		atomicAdd(tmp_sig+r, s);
	}

	return;
}

// append_sigma<<<(nfrt+THREADS-1)/THREADS, THREADS>>>(d_frt, d_sig, d_frt_sig, d_tmp_sig, nfrt);
__global__ void append_sigma(LOCINT * sbuf, LOCINT * sigma, LOCINT *sbuf_sig, LOCINT * tmp_sig, LOCINT n) {

	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;

	sbuf_sig[tid] = sbuf_sig[tid] + tmp_sig[sbuf[tid]];  // this is d_sbuf_sig

	sigma[sbuf[tid]] = sbuf_sig[tid];

	tmp_sig[sbuf[tid]] = 0;
	sbuf[tid] = CUDA_MYLOCI2LOCJ(sbuf[tid]); // Row index to Column Index

	return;
}

static size_t tot_dev_mem = 0;

static void *CudaMallocSet(size_t size, int val) {

        void *ptr;

        MY_CUDA_CHECK( cudaMalloc(&ptr, size) );
        MY_CUDA_CHECK( cudaMemset(ptr, val, size) );
        tot_dev_mem += size;

        return ptr;
}

void *CudaMallocHostSet(size_t size, int val) {

        void *ptr;

        MY_CUDA_CHECK( cudaMallocHost(&ptr, size) );
        memset(ptr, val, size);
        return ptr;
}

void CudaFreeHost(void *ptr) {

        MY_CUDA_CHECK( cudaFreeHost(ptr) );
        return;
}

__global__ void set_degree(LOCINT *col, LOCINT *deg, LOCINT n) {
	
	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= n) return;
	deg[tid] = col[tid+1] - col[tid];
	return;

}

void set_mlp_cuda(LOCINT row, int level, int sigma) {

	LOCINT v;
	MY_CUDA_CHECK( cudaMemcpy(&v, d_msk+row/BITS(d_msk), sizeof(v), cudaMemcpyDeviceToHost) );

	v |= (1ULL<<(row%BITS(d_msk))); 
	MY_CUDA_CHECK( cudaMemcpy(d_msk+row/BITS(d_msk), &v, sizeof(*d_msk), cudaMemcpyHostToDevice) );

	MY_CUDA_CHECK( cudaMemcpy(d_lvl+row, &level, sizeof(level), cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpy(d_sig+row, &sigma, sizeof(sigma), cudaMemcpyHostToDevice) );

	MY_CUDA_CHECK( cudaMemcpy(d_frt, &row, sizeof(row), cudaMemcpyHostToDevice) );

	return;
}

__global__ void compact(LOCINT *col, LOCINT *row, LOCINT *deg, LOCINT *msk) {

	int n;
	LOCINT *v;
        int bid = threadIdx.x;
        int lid = threadIdx.x % warpSize;
        int wid = threadIdx.x / warpSize;

        int goff = 0;
        int bcount = 0;

        __shared__ uint32_t sh_v[32];

        v = row + col[blockIdx.x];
        n = deg[blockIdx.x];

        // sync()s in the loop don't cause stall because
        // every warp that cycle has either all threads
        // cycling or some cycling and some returned.
        for(; bid < n; bid += blockDim.x) {

                LOCINT	 s = v[bid];
                uint32_t m;
                int	 t;

		m = ((LOCINT)1) << (s%BITS(msk));
	        t = (msk[s/BITS(msk)]&m) == 0;

                m = __ballot(t);

                if (lid == wid) sh_v[wid] = __popc(m);
                bcount = __syncthreads_count(t);

                if (wid == 0) {
                        uint32_t k;
                        uint32_t l = sh_v[lid];
                        for (k=1; k <= 16; k <<= 1) {
                                // source th is always present so shfl
                                // never returns the calling th's var
                                int r = __shfl_up((int)l, k, 32);
                                if (lid >= k) l += r;
                        }
                        sh_v[lid] = l;
                }
                uint32_t i = __popc(m & ((1<<lid)-1));
                __syncthreads();

                uint32_t off = (!wid) ? 0 : sh_v[wid-1];
                if (t) v[goff + off + i] = s;

                goff += bcount;
        }
        if (threadIdx.x == 0) deg[blockIdx.x] = goff;
        return;
}



/*
 * scan_frt_csc_cuda(frt, ncol, depth, hRFbuf);
 */
LOCINT scan_frt_csc_cuda_mono(int offset, int ncol, int depth) {

	LOCINT	i;
	int blocks, nrow=0;
	float	et=0;
	TIMER_DEF(1);


	LOCINT *d_ncbuf;

	TIMER_START(1);
	if (!ncol) {
		TIMER_STOP(1);
		goto out;
	}

#ifdef ONEPREFIX
	nrow = tlvl[depth+1];
	d_ncbuf = d_cbuf_start+offset;
#else


	static	thrust::device_ptr<LOCINT> d_val(d_cbuf);
	// calculate degree for each vertex in frt
	read_edge_count<<<(ncol+THREADS-1)/THREADS, THREADS>>>(d_deg, d_frt_start+offset, ncol, d_cbuf);

	MY_CUDA_CHECK( cudaMemcpy(&i, d_cbuf+ncol-1, sizeof(*d_cbuf), cudaMemcpyDeviceToHost) );
	nrow = i;

	// Prefix sum to count how many threads to launch
	thrust::exclusive_scan(d_val, d_val+ncol, d_val);
	MY_CUDA_CHECK( cudaMemcpy(&i, d_cbuf+ncol-1, sizeof(*d_cbuf), cudaMemcpyDeviceToHost) );

	nrow += i;
	d_ncbuf = d_cbuf;
#endif

	if (!nrow) {
		TIMER_STOP(1);
		goto out;
	}
	TIMER_STOP(1);

	MY_CUDA_CHECK( cudaEventRecord(start, 0) );

	blocks = (((nrow+ROWXTHD-1)/ROWXTHD)+THREADS-1)/THREADS;
	//dump_device_farray2(d_delta, row_pp, "d_delta");

	// Store result directly into d_delta
	scan_frt_mono<<<blocks, THREADS>>>(d_row, d_col, nrow, d_frt_start+offset, d_ncbuf, ncol,
			                           d_sig, d_delta, d_lvl, depth);

	MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
	MY_CHECK_ERROR("scan_frt");
	MY_CUDA_CHECK( cudaEventSynchronize(stop) );
	MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );

out:

   return ncol;

}





/**
 */
LOCINT scan_col_csc_cuda_mono(int ncol, int level) {

	int blocks;
	LOCINT	i;
	float	et=0;
	LOCINT	nfrt=0, nrow=0;
#ifdef ONEPREFIX

#ifdef THRUST
	thrust::device_ptr<LOCINT> d_val(d_cbuf);
#endif
	int *d_out = NULL;

#else

#ifdef THRUST
    	static thrust::device_ptr<LOCINT> d_val(d_cbuf);
#endif

#endif

	TIMER_DEF(1);
	TIMER_DEF(2);

	TIMER_START(1);

	MY_CUDA_CHECK( cudaMemset(d_snum, 0, sizeof(*d_snum)) );

	read_edge_count<<<(ncol+THREADS-1)/THREADS, THREADS>>>(d_deg, d_frt, ncol, d_cbuf);
	MY_CHECK_ERROR("read_edge_count");
	MY_CUDA_CHECK( cudaDeviceSynchronize() );

	MY_CUDA_CHECK( cudaMemcpy(&i, d_cbuf+ncol-1, sizeof(*d_cbuf), cudaMemcpyDeviceToHost) );

	nrow = i;


	// Prefix sum to count how many threads to launch
	thrust::exclusive_scan(d_val, d_val+ncol, d_val);
	MY_CUDA_CHECK( cudaMemcpy(&i, d_cbuf+ncol-1, sizeof(*d_cbuf), cudaMemcpyDeviceToHost) );
	nrow += i;

#ifdef ONEPREFIX
	tlvl[level] = nrow;
#endif

	if (!nrow) {
		TIMER_STOP(1);
		goto out;
	}
	TIMER_STOP(1);

	MY_CUDA_CHECK( cudaEventRecord(start, 0) );

	blocks = (((nrow+ROWXTH-1)/ROWXTH)+THREADS-1)/THREADS;


	scan_col_mono2<<<blocks, THREADS>>>(d_row, d_col, nrow, d_frt, d_cbuf, ncol, d_msk, d_lvl, 
                                        d_sig, level, d_frt+ncol, d_snum);

	// Here we have d_sbuf updated with the new discovered vertices and d_tmp_sig with the local value of the accumulated sigma
	MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
	MY_CHECK_ERROR("scan_col");
	MY_CUDA_CHECK( cudaEventSynchronize(stop) );
	MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );

	//dump_device_uarray2(d_sig, row_pp, "scan_col d_sig 2");
out:
	TIMER_START(2);

	// Prepare sbuf to send vertices to other processors (We need to send Sigma as well
	// copy d_snum back into CPU
	MY_CUDA_CHECK( cudaMemcpy(&nfrt, d_snum, sizeof(nfrt), cudaMemcpyDeviceToHost) );
	//dump_device_uarray2(d_frt+ncol, nfrt, "scan_col d_frt 3");

	d_frt = d_frt + ncol;
#ifdef ONEPREFIX
	d_cbuf = d_cbuf + ncol;
#endif

	TIMER_STOP(2);

	return nfrt;
}


		
LOCINT append_rows_cuda(LOCINT *rbuf, LOCINT ld,   int *rnum, int np,
			            LOCINT *frt, LOCINT *frt_sigma, LOCINT nfrt, int level) {
	float	et=0;
	LOCINT	nrow=0;
	LOCINT	p, q;

	LOCINT ld2 = ld*2;

	TIMER_DEF(1);
	
	TIMER_START(1);

	nrow = 0;
	for(int i = 0; i < np; i++) {
		if (rnum[i]) {
			MY_CUDA_CHECK( cudaMemcpy(d_rbuf+nrow, rbuf+i*ld2, rnum[i]*sizeof(*rbuf), cudaMemcpyHostToDevice) );
			MY_CUDA_CHECK( cudaMemcpy(d_rbuf_sig+nrow, rbuf+i*ld2+rnum[i], rnum[i]*sizeof(*rbuf), cudaMemcpyHostToDevice) );
			nrow += rnum[i];
		}
	}

	if (nrow > 0) {
		//	MY_CUDA_CHECK( cudaMemcpy(d_rbuf, rbuf, nrow*sizeof(*rbuf), cudaMemcpyHostToDevice) );
		// in-place prefix-sum of rnum (too small to bother thrust)
		p = rnum[0]; rnum[0] = 0;
		for(int i = 1; i < np; i++) {
			q = rnum[i];
			rnum[i] = p + rnum[i-i];
			p = q;
		}
		MY_CUDA_CHECK( cudaMemcpy(d_cbuf, rnum, np*sizeof(*rnum), cudaMemcpyHostToDevice) );
		MY_CUDA_CHECK( cudaMemcpyToSymbol(dnfrt, &nfrt, sizeof(dnfrt), 0, cudaMemcpyHostToDevice) );
		TIMER_STOP(1);

		//dump_device_array2(d_rbuf_sig, nrow, "append d_rbuf_sig");
		//dump_device_array2(d_frt, nfrt, "append d_frt 1 ");
		//dump_device_array2(d_frt_sig, nfrt, "append d_frt_sig 1 ");
		//dump_device_array2(d_tmp_sig, row_pp, "append d_tmp_sig 1");

		MY_CUDA_CHECK( cudaEventRecord(start, 0) );
		// Here update d_sbuf and d_sig, after that we need to update d_sbuf_sig

		// UPDATE DFRT sostituito d_pred con di reach... 7 arg
		append_row<<<(nrow+THREADS-1)/THREADS, THREADS>>>(d_rbuf, d_rbuf_sig, nrow, d_cbuf, np,
								  d_msk, d_reach, d_lvl, level, d_frt, d_tmp_sig, d_frt_sig, d_all);
		MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
		MY_CHECK_ERROR("append_row");
		MY_CUDA_CHECK( cudaEventSynchronize(stop) );
		MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );
		//if (myid == 0) fprintf(stdout, "\tappend_row time = %f + %f\n", TIMER_ELAPSED(1)/1.0E+6, et/1.0E3);
	
		MY_CUDA_CHECK( cudaMemcpyFromSymbol(&nfrt, dnfrt, sizeof(nfrt), 0, cudaMemcpyDeviceToHost) );

	}

	if (nfrt > 0) {

		MY_CUDA_CHECK( cudaEventRecord(start, 0) );

		// READ DFRT
	    append_sigma<<<(nfrt+THREADS-1)/THREADS, THREADS>>>(d_frt, d_sig, d_frt_sig, d_tmp_sig, nfrt);

	    MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
	    MY_CHECK_ERROR("append_sigma");
	    MY_CUDA_CHECK( cudaEventSynchronize(stop) );
	    MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );

	   // Add new vertices to the frontier
	   MY_CUDA_CHECK( cudaMemcpy(frt, d_frt, nfrt*sizeof(*d_frt), cudaMemcpyDeviceToHost) );
	   MY_CUDA_CHECK( cudaMemcpy(frt_sigma, d_frt_sig, nfrt*sizeof(*d_frt_sig), cudaMemcpyDeviceToHost) );
	}
	//dump_device_array2(d_frt, nfrt, "append d_frt 3");
	//dump_device_array2(d_frt_sig, nfrt, "append d_frt_sig 3");
	//dump_device_array2(d_tmp_sig, row_pp, "append d_sbuf_sig 3");
	//dump_device_array2(d_sig, row_pp, "append d_sig 3");
	
	return nfrt;
}




void get_lvl(int *lvl) {
	MY_CUDA_CHECK( cudaMemcpy(lvl+(mycol*row_bl), d_lvl+(mycol*row_bl), row_bl*sizeof(*d_lvl), cudaMemcpyDeviceToHost) );
}

void set_lvl(int *lvl) {
	MY_CUDA_CHECK( cudaMemcpy(d_lvl, lvl, row_pp*sizeof(*lvl), cudaMemcpyHostToDevice) );
}

void get_all(LOCINT *all) {
	MY_CUDA_CHECK( cudaMemcpy(all, d_all, sizeof(*d_all), cudaMemcpyDeviceToHost) );
	// and set to zero for next bc
	MY_CUDA_CHECK(cudaMemset(d_all, 0,sizeof(*d_all)));

}
	
void get_frt(LOCINT *frt) {
	MY_CUDA_CHECK( cudaMemcpy(frt, d_frt_start, row_pp*sizeof(LOCINT), cudaMemcpyDeviceToHost) );
}

void get_cbuf(LOCINT *cbuf) {
	MY_CUDA_CHECK( cudaMemcpy(cbuf, d_cbuf_start, row_pp*sizeof(LOCINT), cudaMemcpyDeviceToHost) );
}

void get_msk(LOCINT *msk) {
	MY_CUDA_CHECK( cudaMemcpy(msk, d_msk, ((row_pp+BITS(d_msk)-1)/BITS(d_msk))*sizeof(*d_msk), cudaMemcpyDeviceToHost) );
}

void get_deg(LOCINT *deg) {
	MY_CUDA_CHECK( cudaMemcpy(deg, d_deg, col_bl*sizeof(*d_deg), cudaMemcpyDeviceToHost) );
}

void get_sigma(LOCINT *sigma) {
	MY_CUDA_CHECK( cudaMemcpy(sigma+(mycol*row_bl), d_sig+(mycol*row_bl), row_bl*sizeof(*d_sig), cudaMemcpyDeviceToHost) );
}

void get_bc(float *bc) {
	MY_CUDA_CHECK( cudaMemcpy(bc, d_bc, row_pp*sizeof(*d_bc), cudaMemcpyDeviceToHost) );
}

void set_sigma(LOCINT *sigma) {
	MY_CUDA_CHECK( cudaMemcpy(d_sig, sigma, row_pp*sizeof(*sigma), cudaMemcpyHostToDevice) );
}

__global__ void set_delta(float *srbuf, float * delta, int nrow) {

	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= nrow) return;

	delta[tid] += srbuf[tid];
	srbuf[tid] = 0;

	return;
}

int set_delta_cuda(float *hSRbuf, int nrow) {

	float et=0;
	TIMER_DEF(1);

	TIMER_START(1);

	if (!nrow) {
		TIMER_STOP(1);
		return nrow;
	}

	//MY_CUDA_CHECK( cudaMemset(d_frbuf, 0, row_pp*sizeof(*d_frbuf)) );
	MY_CUDA_CHECK( cudaMemcpy(d_frbuf, hSRbuf , nrow*sizeof(*hSRbuf), cudaMemcpyHostToDevice ));
	MY_CUDA_CHECK( cudaEventRecord(start, 0) );
	set_delta<<<(nrow+THREADS-1)/THREADS, THREADS>>>(d_frbuf, d_delta, nrow);
	MY_CUDA_CHECK( cudaEventRecord(stop, 0) );
	MY_CHECK_ERROR("set_delta");
	MY_CUDA_CHECK( cudaEventSynchronize(stop) );
	MY_CUDA_CHECK( cudaEventElapsedTime(&et, start, stop) );

	return nrow;
}

void init_bc_1degree_device(LOCINT *reach) {
      //MY_CUDA_CHECK( cudaMemcpy(d_bc, bc_val,  row_pp*sizeof(*bc_val), cudaMemcpyHostToDevice) );
      MY_CUDA_CHECK( cudaMemcpy(d_reach, reach, row_pp*sizeof(*reach), cudaMemcpyHostToDevice) );
      return;
}

__global__ void init_delta(LOCINT *reach, float * delta, int nrow) {

	const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;

	if (tid >= nrow) return;

	delta[tid] = (float)reach[tid];

	return;
}

void setcuda(uint64_t ned, LOCINT *col, LOCINT *row, LOCINT reach_v0) {

	MY_CUDA_CHECK( cudaMemset(d_lvl,-1, row_pp*sizeof(*d_lvl)) );
	MY_CUDA_CHECK( cudaMemset(d_sig, 0, row_pp*sizeof(*d_sig)) );

      	init_delta<<<(row_pp+THREADS-1)/THREADS, THREADS>>>(d_reach, d_delta, row_pp);

	MY_CUDA_CHECK( cudaMemset(d_msk, 0, ((row_pp+BITS(d_msk)-1)/BITS(d_msk))*sizeof(*d_msk)) );

	MY_CUDA_CHECK( cudaMemcpyToSymbol(d_reach_v0, &reach_v0, sizeof(d_reach_v0), 0, cudaMemcpyHostToDevice) );

#ifdef ONEPREFIX
	memset(tlvl, 0, sizeof(*tlvl)*MAX_LVL);
	d_cbuf = d_cbuf_start;
#endif

	d_frt = d_frt_start;

	return;
}



size_t initcuda(uint64_t ned, LOCINT *col, LOCINT *row) {

	int dev;

	dev = 0; //FORCED 
	MY_CUDA_CHECK( cudaSetDevice(dev) );

	d_col = (LOCINT *)CudaMallocSet((col_bl+1)*sizeof(*d_col), 0);
	d_row = (LOCINT *)CudaMallocSet(ned*sizeof(*d_row), 0);
	MY_CUDA_CHECK( cudaMemcpy(d_col, col, (col_bl+1)*sizeof(*col), cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpy(d_row, row, ned*sizeof(*row), cudaMemcpyHostToDevice) );

	d_deg = (LOCINT *)CudaMallocSet(col_bl*sizeof(*d_deg), 0); 
	set_degree<<<(col_bl+THREADS-1)/THREADS, THREADS>>>(d_col, d_deg, col_bl);

	if (!mono) { // Run Multi-GPU code
                printf("MPI REMOVED\n");
	} else {
#ifdef ONEPREFIX
		d_cbuf_start = (LOCINT *)CudaMallocSet(MAX(row_bl,row_pp)*sizeof(*d_cbuf_start), 0);
		
#else
		d_cbuf_start = (LOCINT *)CudaMallocSet(MAX(col_bl, C)*sizeof(*d_cbuf_start), 0);
#endif
	}
	
	d_sbuf = (LOCINT *)CudaMallocSet(MAX(row_bl,row_pp)*sizeof(*d_sbuf), 0);
	d_snum = (uint32_t *)CudaMallocSet((C+1)*sizeof(*d_snum), 0);

	d_msk = (LOCINT *)CudaMallocSet(((row_pp+BITS(d_msk)-1)/BITS(d_msk))*sizeof(*d_msk), 0);
	d_lvl = (int *)CudaMallocSet(row_pp*sizeof(*d_lvl), -1);
	d_all = (uint32_t *)CudaMallocSet(1*sizeof(*d_all), 0);

	d_frt = d_sbuf;
	d_frt_start = d_sbuf;
	d_cbuf = d_cbuf_start;

	d_sig =       (LOCINT *)CudaMallocSet(row_pp*sizeof(*d_sig), 0);
	
	d_delta =     (float *)CudaMallocSet(row_pp*sizeof(*d_delta), 0);
	d_bc    =     (float *)CudaMallocSet(row_pp*sizeof(*d_bc), 0);
        d_reach =     (LOCINT*)CudaMallocSet(row_pp*sizeof(*d_reach), 0);


        
	printf("ROWBL = %i - ROWPP = %i\n",row_bl,row_pp);

	MY_CUDA_CHECK( cudaEventCreate(&start) );
    	MY_CUDA_CHECK( cudaEventCreate(&stop) );

	MY_CUDA_CHECK( cudaStreamCreate(stream+0) );
    	MY_CUDA_CHECK( cudaStreamCreate(stream+1) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dN, &N, sizeof(dN),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dC, &C, sizeof(dC),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dR, &R, sizeof(dR),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dmyrow, &myrow, sizeof(dmyrow),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dmycol, &mycol, sizeof(dmycol),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(drow_bl, &row_bl, sizeof(drow_bl),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(dcol_bl, &col_bl, sizeof(dcol_bl),  0, cudaMemcpyHostToDevice) );
	MY_CUDA_CHECK( cudaMemcpyToSymbol(drow_pp, &row_pp, sizeof(drow_pp),  0, cudaMemcpyHostToDevice) );

        MY_CUDA_CHECK( cudaFuncSetCacheConfig(read_edge_count, cudaFuncCachePreferL1) );
        MY_CUDA_CHECK( cudaFuncSetCacheConfig(update_bc, cudaFuncCachePreferL1) );
        MY_CUDA_CHECK( cudaFuncSetCacheConfig(deviceReduceKernel, cudaFuncCachePreferL1) );

        if (!mono){
            printf("MPI REMOVED\n");
        }
        else{
        // set cache mono
            MY_CUDA_CHECK( cudaFuncSetCacheConfig(scan_frt_mono, cudaFuncCachePreferL1) );
            MY_CUDA_CHECK( cudaFuncSetCacheConfig(scan_col_mono, cudaFuncCachePreferL1) );

        }
	return tot_dev_mem;
}
		

void fincuda() {

	MY_CUDA_CHECK( cudaFree(d_col) );
	MY_CUDA_CHECK( cudaFree(d_deg) ); //////////////////////
	MY_CUDA_CHECK( cudaFree(d_row) );
	MY_CUDA_CHECK( cudaFree(d_rbuf) );
	MY_CUDA_CHECK( cudaFree(d_cbuf_start) );

	MY_CUDA_CHECK( cudaFree(d_sbuf) );
	MY_CUDA_CHECK( cudaFree(d_snum) );
	MY_CUDA_CHECK( cudaFree(d_msk) );
	MY_CUDA_CHECK( cudaFree(d_lvl) );

	MY_CUDA_CHECK( cudaFree(d_tmp_sig) );
	MY_CUDA_CHECK( cudaFree(d_sig) );
	MY_CUDA_CHECK( cudaFree(d_rbuf_sig) );
	MY_CUDA_CHECK( cudaFree(d_sbuf_sig) );

	MY_CUDA_CHECK( cudaFree(d_frbuf) );
	MY_CUDA_CHECK( cudaFree(d_fsbuf) );
	MY_CUDA_CHECK( cudaFree(d_delta) );
	MY_CUDA_CHECK( cudaFree(d_bc) );
	MY_CUDA_CHECK( cudaEventDestroy(start) );
        MY_CUDA_CHECK( cudaEventDestroy(stop) );

	MY_CUDA_CHECK( cudaStreamDestroy(stream[0]) );
	MY_CUDA_CHECK( cudaStreamDestroy(stream[1]) );
        

        
	return;
}
