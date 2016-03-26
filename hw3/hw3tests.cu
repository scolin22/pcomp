#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>

#define LOGMAP 1
#define NORM 2
#define PERC 3
#define DEFAULT_TEST LOGMAP

#define ALPHA 1.0f

// CITATION: Code snippets taken from examples.cu by Mark
/**************************************************************
 *  reduce_sum: compute the sum of the elements of an array
 *    Simple version: we only handle one block of threads
 ***************************************************************/

__device__ void reduce_sum_dev(uint n, float *x) {
  uint myId = threadIdx.x;
  for(uint m = n >> 1; m > 0; m = n >> 1) {
    n -= m;
    __syncthreads();
    if(myId < m)
      x[myId] += x[myId+n];
  }
}

__global__ void reduce_sum(uint n, float *x) {
  reduce_sum_dev(n, x);
}

/************************
*  logistic map
************************/

__global__ void logmap(float *x, int n, int m) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
    for(int j = 0; j < m; j++)
      x[i] = ALPHA * x[i] * (1.0f - x[i]);
}

void logmap_ref(float *x, int n, int m, float *z) {
  memcpy(z, x, n*sizeof(float));

  for(int j = 0; j < m; j++)
    for(int i = 0; i < n; i++)
      z[i] = ALPHA * z[i] * (1.0f - z[i]);
}

/************************
*  norm calculation
************************/

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

float norm_ref(float *x, int n) {
  float sum = 0.0;

  for(int i = 0; i < n; i++)
    sum += pow(x[i],2);

  return sqrt(sum);
}

/************************
*  perc
************************/

__global__ void setup_kernel(uint n, curandState *state) {
  uint myId = blockDim.x * blockIdx.x + threadIdx.x;
  if(myId < n)
    curand_init(1234, myId, 0, &state[myId]);
}

__global__ void rndm(uint *a, int m, curandState *state) {
  uint i = blockDim.x * blockIdx.x + threadIdx.x;
  for(int j = 0; j < m; j++)
    a[i*m + j] = curand_uniform(&state[i])*1000;
}

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

/*****************************************************
*  near(x, y): true if x and y are "nearly" equal
******************************************************/
int near(uint n, float x, float y) {
  return(abs(x-y) < max(10.0, sqrt((float)n))*1.0e-7*max(1.0, max(abs(x), abs(y))));
}

int main(int argc, char **argv) {
  uint n = (argc >= 2) ? atoi(argv[1]) : 1000000;
  uint nn = n;
  uint what = (argc >= 3) ? atoi(argv[2]) : DEFAULT_TEST;
  // m for LOGMAP
  uint m = (argc >= 4 && what == LOGMAP) ? atoi(argv[3]) : 0;
  float *x, *y, *z, *z_ref;
  float *dev_x, *dev_y, *dev_z;
  uint *a, *dev_a;
  curandState *dev_randState;
  cudaDeviceProp prop;

  int ndev;
  cudaGetDeviceCount(&ndev);
  if(ndev < 1) {
    fprintf(stderr, "No CUDA device found\n");
    exit(-1);
  }
  cudaGetDeviceProperties(&prop, 0);

  int size = n*sizeof(float);
  x  = (float *)malloc(size);
  y  = (float *)malloc(size);
  z  = (float *)malloc(size);
  a  = (uint *)malloc(size);
  z_ref = (float *)malloc(size);

  // Use a logistic map to make some pseudo-random numbers
  // It's fast, but the distribution isn't very uniform, and
  //   the other statistical properties are lousy.  But it's
  //   fast, and that's all we need for some simple tests.
  x[0] = 0.123;
  y[0] = sqrt(0.3);
  for(int i = 1; i < n; i++) {
    x[i] = 3.8*x[i-1]*(1.0 - x[i-1]);
    y[i] = 3.9*y[i-1]*(1.0 - y[i-1]);
  }

  printf("The GPU is a %s\n", prop.name);
  printf("Cuda capability %d.%d.\n", prop.major, prop.minor);
  print_vec(x, min(10, n), "%5.3f", "x");
  print_vec(y, min(10, n), "%5.3f", "y");

  cudaMalloc((void**)(&dev_x), size);
  cudaMalloc((void**)(&dev_y), size);
  cudaMalloc((void**)(&dev_z), size);
  cudaMalloc((void**)(&dev_a), size);
  cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, size, cudaMemcpyHostToDevice);
  cudaMalloc((void **)(&dev_randState), n*sizeof(curandState));

  switch(what) {
    case LOGMAP:
      logmap<<<ceil(n/256.0),256>>>(dev_x, n, m);
      printf("a: size = %d, z=%016llx dev_x=%016llx\n", size, z, dev_x);
      cudaMemcpy(z, dev_x, size, cudaMemcpyDeviceToHost);
      printf("b\n");
      logmap_ref(x, n, m, z_ref);
      break;
    case NORM:
      norm<<<ceil(n/1024.0),1024>>>(dev_x, n, dev_z);
      if (ceil(n/1024.0) > 1)
        norm<<<1,ceil(n/1024.0)>>>(dev_z, ceil(n/1024.0), dev_z);
      cudaMemcpy(z, dev_z, size, cudaMemcpyDeviceToHost);
      z_ref[0] = norm_ref(x, n);
      nn = 1;
      break;
    case PERC:
      setup_kernel<<<ceil(n/1024.0),1024>>>(n, dev_randState);
      rndm<<<ceil(n/1024.0),1024>>>(dev_a, 1, dev_randState);
      cudaMemcpy(a, dev_a, size, cudaMemcpyDeviceToHost);
      print_vec(a, min(10, nn), "%d", "a");
      break;
    default:
      fprintf(stderr, "ERROR: unknown test case -- %d\n", what);
      exit(-1);
  }

  for(int i = 0; i < nn; i++) { // check the result
    if(!near(n, z[i], z_ref[i])) {
      fprintf(stderr, "ERROR: i=%d: z[i] = %15.10f, z_ref[i] = %15.10f\n", i, z[i], z_ref[i]);
      exit(-1);
    }
  }
  print_vec(z, min(10, nn), "%5.3f", "z");
  printf("The results match!\n");

  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);
  free(x);
  free(y);
  free(z);
  free(z_ref);
  exit(0);
}
