#include <stdio.h>
#include <math.h>
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
  int m, nblks, tpb, warpSize, whichKernel;
};

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
*  Rest of Code
************************/

int main(int argc, char **argv) {
  int nblks = 24; // default number of blocks in the grid
  int m = 1000;   // default number of "rounds" in kernel
  int tpb = 256;  // default threads per block
  int whichKernel = 1; // default kernel to run
  float *v, *dev_v, *dev_z;
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
  HANDLE_ERROR(cudaMemcpy(dev_z, v, szv, cudaMemcpyHostToDevice));

  // initialize argk
  argk.n = nv;
  argk.v = dev_v;
  argk.z = dev_z;
  argk.m = m;
  argk.nblks = nblks;
  argk.tpb = tpb;
  argk.warpSize = prop.warpSize;
  argk.whichKernel = whichKernel;

  // run the kernel and report timing info
  time_it_run(tr, do_logmap, (void *)(&argk));
  time_it_get_stats(tr, &stats);
  HANDLE_ERROR(cudaMemcpy(v, dev_v, szv, cudaMemcpyDeviceToHost));
  printf("mean(T) = %10.3e, std(T) = %10.3e\n", stats.mean, stats.std);

  //clean up
  cudaFree(dev_v);
  free(v);
  time_it_free(tr);
  exit(0);
}
