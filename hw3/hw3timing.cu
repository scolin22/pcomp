#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#include "time_it.h"

// HANDLE_ERROR is from "CUDA by Exmaple" by Sanders and Kandrot
//   I found the actual source code at
//     http://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda
//   (The URL in the book for getting the source appears to be no longer
//    available.)

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// struct for passing arguments through time_it_run to our kernel functions
struct kernel_arg {
  uint n;
  float *v, *z;
  uint *a;
  int m, nblks, tpb, warpSize, whichKernel;
  curandState *dev_randState;
};

/*****************************************************
*  print_vec: print the first few elements of a vector
******************************************************/

void print_vec(float *x, uint n, const char *fmt, const char *who) {
  printf("%s = ", who);
  for(int i = 0; i < n; i++) {
    if(i > 0) printf(", ");
    printf(fmt, x[i]);
  }
  if(n > 10) printf(", ...");
  printf("\n");
}

void print_vec(uint *x, uint n, const char *fmt, const char *who) {
  printf("%s = ", who);
  for(int i = 0; i < n; i++) {
    if(i > 0) printf(", ");
    printf(fmt, x[i]);
  }
  if(n > 10) printf(", ...");
  printf("\n");
}

/************************
*  Change by Colin Stone, Rest taken from examples.cu by Mark
*  logistic map
************************/
#define ALPHA 1.0f

__global__ void logmap(float *x, int n, int m) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
    for(int j = 0; j < m; j++)
      x[i] = ALPHA * x[i] * (1.0f - x[i]);
}

void do_logmap(void *void_arg) {
  struct kernel_arg *argk = (struct kernel_arg *)(void_arg);
  // printf("RUNNING LOGMAP m=%d nblks=%d tpb=%d\n", argk->m, argk->nblks, argk->tpb);
  logmap<<<argk->nblks,argk->tpb>>>(argk->v, argk->n, argk->m);
  cudaDeviceSynchronize();
}

/************************
*  Change by Colin Stone, Rest taken from examples.cu by Mark
*  norm
************************/

__device__ void reduce_sum_dev(uint n, float *x) {
  uint myId = threadIdx.x;
  for(uint m = n >> 1; m > 0; m = n >> 1) {
    n -= m;
    __syncthreads();
    if(myId < m)
      x[myId] += x[myId+n];
  }
}

__global__ void norm(float *x, int n, float *z) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  uint blockBase = blockDim.x * blockIdx.x;
  uint m = min(blockDim.x, n - blockBase);

  if (i < n)
    x[i] = pow(x[i], 2);

  __syncthreads();

  reduce_sum_dev(m, &(x[blockBase]));

  if (i < n && threadIdx.x == 0)
    z[blockIdx.x] = sqrt(x[i]);
}

void do_norm(void *void_arg) {
  struct kernel_arg *argk = (struct kernel_arg *)(void_arg);
  printf("RUNNING NORM nblks=%d tpb=%d\n", argk->nblks, argk->tpb);
  norm<<<argk->nblks,argk->tpb>>>(argk->v, argk->n, argk->z);
  cudaDeviceSynchronize();
  if (argk->nblks > 1)
    norm<<<1,argk->nblks>>>(argk->z, argk->nblks, argk->z);
  cudaDeviceSynchronize();
}

/************************
*  Change by Colin Stone, Rest taken from examples.cu by Mark
*  random number generator
************************/

__global__ void setup_kernel(uint n, curandState *state) {
  uint myId = blockDim.x * blockIdx.x + threadIdx.x;
  if(myId < n)
    curand_init(1234, myId, 0, &state[myId]);
}

__global__ void rndm(uint *a, int m, curandState *state) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int j = 0; j < m; j++) {
    // printf("idx=%d seed=%d\n", i*m+j, i);
    a[i*m + j] = curand_uniform(&state[i])*1000;
  }
}

void do_rndm(void *void_arg) {
  struct kernel_arg *argk = (struct kernel_arg *)(void_arg);
  printf("RUNNING RNDM nblks=%d tpb=%d m=%d\n", argk->nblks, argk->tpb, argk->m);

  setup_kernel<<<argk->nblks,argk->n>>>(argk->n, argk->dev_randState);
  cudaDeviceSynchronize();

  rndm<<<argk->nblks,argk->tpb>>>(argk->a, argk->m, argk->dev_randState);
  cudaDeviceSynchronize();

  int n = argk->n*argk->m;
  int size = n*sizeof(uint);
  uint *a = (uint *)malloc(size);
  cudaMemcpy(a, argk->a, size, cudaMemcpyDeviceToHost);
  print_vec(a, min(10, n), "%d", "a");
}

/************************
*  Rest of Code
************************/

int main(int argc, char **argv) {
  int nblks = 24; // default number of blocks in the grid
  int m = 1000;   // default number of "rounds" in kernel
  int tpb = 256;  // default threads per block
  int whichKernel = 1; // default kernel to run
  float *v, *dev_v, *dev_z;
  uint *dev_a;
  curandState *dev_randState;
  cudaDeviceProp prop;
  struct kernel_arg argk;
  struct time_it_raw *tr = time_it_create(10);
  struct time_it_stats stats;

  int ndev;
  HANDLE_ERROR(cudaGetDeviceCount(&ndev));
  if(ndev < 1) {
    fprintf(stderr, "No CUDA device found\n");
    exit(-1);
  }
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
  int sharedMemPerBlock = prop.sharedMemPerBlock;
  int regsPerBlock = prop.regsPerBlock;
  printf("GPU is a %s supporing CUDA level %d.%d\n", prop.name, prop.major, prop.minor);
  printf("  It has %d SMs and a warp size of %d\n", prop.multiProcessorCount, prop.warpSize);
  printf("  sharedMemPerBlock = %d, regsPerBlock = %d\n", sharedMemPerBlock, regsPerBlock);
  printf("  clock rate = %d\n", prop.clockRate);
  printf("Max Threads per Block = %d\n", prop.maxThreadsPerBlock);

  for(int i = 1; i < argc; i++) {
    if(strncmp(argv[i], "nblks=", strlen("nblks=")) == 0) {
      nblks = atoi(argv[i]+strlen("nblks="));
      if(nblks <= 0) {
        fprintf(stderr, "bad option: %s, nblks must be positive\n", argv[i]);
        exit(-1);
      }
    } else if(strncmp(argv[i], "m=", strlen("m=")) == 0) {
      m = atoi(argv[i]+strlen("m="));
      if(m < 0) {
        fprintf(stderr, "bad option: %s, m must be non-negative\n", argv[i]);
        exit(-1);
      }
    } else if(strncmp(argv[i], "tpb=", strlen("tpb=")) == 0) {
      tpb = atoi(argv[i]+strlen("tpb="));
      if(tpb <= 0) {
        fprintf(stderr, "bad option: %s, tpb must be positive\n", argv[i]);
        exit(-1);
      }
    } else if(strncmp(argv[i], "kern=", strlen("kern=")) == 0) {
      whichKernel = atoi(argv[i]+strlen("kern="));
      if((whichKernel < 1) || (2 < whichKernel)) {
        fprintf(stderr, "bad option: %s, kern must be 1 or 2\n", argv[i]);
        exit(-1);
      }
    } else {
      fprintf(stderr, "unknown command-line argument: %s\n", argv[i]);
      exit(-1);
    }
  }

  // allocate and initialize v
  int nv = nblks*tpb;
  int szv = nv*sizeof(float);
  v = (float *)malloc(szv);

  v[0] = 0.123;
  for(int i = 1; i < nv; i++)
    v[i] = 3.8*v[i-1]*(1.0 - v[i-1]);

  // allocate and initialize dev_v
  HANDLE_ERROR(cudaMalloc((void **)(&dev_v), szv));
  HANDLE_ERROR(cudaMemcpy(dev_v, v, szv, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMalloc((void **)(&dev_z), szv));
  HANDLE_ERROR(cudaMalloc((void **)(&dev_a), szv*m));
  HANDLE_ERROR(cudaMalloc((void **)(&dev_randState), nv*sizeof(curandState))); //sizeof(curandState) = 48

  // initialize argk
  argk.n = nv;
  argk.v = dev_v;
  argk.z = dev_z;
  argk.a = dev_a;
  argk.m = m;
  argk.nblks = nblks;
  argk.tpb = tpb;
  argk.warpSize = prop.warpSize;
  argk.whichKernel = whichKernel;
  argk.dev_randState = dev_randState;

  // run the kernel and report timing info
  time_it_run(tr, do_rndm, (void *)(&argk));
  time_it_get_stats(tr, &stats);
  HANDLE_ERROR(cudaMemcpy(v, dev_v, szv, cudaMemcpyDeviceToHost));
  printf("mean(T) = %10.3e, std(T) = %10.3e\n", stats.mean, stats.std);

  //clean up
  cudaFree(dev_v);
  free(v);
  time_it_free(tr);
  exit(0);
}
