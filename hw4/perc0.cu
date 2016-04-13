/* perc0.cu: percolation simulation using GPUs
 *   Original version.
 *   Usage: perc0 [w=WIDTH] [h=HEIGHT] [p=PROBABILITY]
 *     Where
 *       w is the width of the graph.
 *       h is the height of the graph -- number of rounds to simulate.
 *       p is the probability of marking a vertex if one
 *         of its predecessors is marked.  If neither is marked, then this
 *         vertex won't get marked either.
 *     Results:
 *       Print the mean and standard deviation for the elapsed time for
 *         the simulation.  Computed based on 10 trials.
 */
#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "time_it.h"

// the typedef below makes it 'easy' to switch to using a differeng random number generator
typedef curandState randstate_t;

// HANDLE_ERROR is from "CUDA by Example" by Sanders and Kandrot
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

// Initialize the prng states.  From
//   http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
__global__ void setup_kernel(uint n, randstate_t *state)
{
  uint myId = blockDim.x * blockIdx.x + threadIdx.x;
  /* Each thread gets same seed, a different sequence number, no offset */
  if(myId < n)
    curand_init(1234, myId, 0, &state[myId]);
}

// perc: a kernel that computes m rounds of percolation
__global__ void perc(uint n, uint m, float p, curandState *randState, uint *v) {
  uint myId = threadIdx.x; // my index -- assume there's only one block
  uint id1 = ((myId == 0) ? n : myId) - 1; // index of my neighbour to the left
  curandState *myRandState = &(randState[myId]); // my random number generator

  v[myId] = 1; // all initial vertices are marked (top row)
  for(int j = 1; j <= m; j++) { // for each round of percolation
    __syncthreads(); // make sure it's safe to read from v
    // should we be marked in this round?
    uint next = (v[myId] || v[id1]) && (curand_uniform(myRandState) <= p);
    __syncthreads(); // make sure it's safe to write to v.
    v[myId] = next; // update the global state so our neigbour can see it.
  }
}

// arguments to the do_perc function (below)
struct perc_arg {
  uint w,  // width of the graph
       h;  // height of the graph
  float p; // probabitity that a vertex is marked if one of its predecessors is marked
  curandState *dev_randState; // an array of n random number generators
  uint *dev_v; // write the final state here.
};

// do_perc: launch the perc kernel
//   wrapped up as a CPU function so timing_run can call us.
void do_perc(void *void_arg) {
  perc_arg *arg = (perc_arg *)(void_arg);
  perc<<<1,arg->w>>>(arg->w, arg->h, arg->p, arg->dev_randState, arg->dev_v);
  cudaDeviceSynchronize();
}

main(int argc, char **argv) {
  int ndev;
  cudaDeviceProp prop;
  uint *v;
  struct time_it_raw *tr = time_it_create(10);
  struct time_it_stats stats;
  perc_arg parg; // parameters for do_perc

  // make sure we have a GPU
  HANDLE_ERROR(cudaGetDeviceCount(&ndev));
  if(ndev < 1) {
    fprintf(stderr, "No CUDA device found\n");
    exit(-1);
  }
  HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));

  parg.w = 256;
  parg.h = 10000;
  parg.p = 0.7;
  for(int i = 1; i < argc; i++) {
    if(strncmp(argv[i], "h=", strlen("h=")) == 0) {
      parg.h = strtoul(argv[i]+strlen("h="), NULL, 10);
      if(parg.h == 0) {
        fprintf(stderr, "bad option: %s, m must be non-zero\n", argv[i]);
        exit(-1);
      }
    } else if(strncmp(argv[i], "p=", strlen("p=")) == 0) {
      parg.p = strtod(argv[i]+strlen("p="), NULL);
      if((parg.p < 0.0) || (1.0 < parg.p)) {
        fprintf(stderr, "bad option: %s, p must be between 0.0 and 1.0\n", argv[i]);
        exit(-1);
      }
    } else if(strncmp(argv[i], "w=", strlen("w=")) == 0) {
      parg.w = strtoul(argv[i]+strlen("w="), NULL, 10);
      if(parg.w == 0) {
        fprintf(stderr, "bad option: %s, n must be non-zero\n", argv[i]);
        exit(-1);
      }
    } else {
      fprintf(stderr, "unknown command-line argument: %s\n", argv[i]);
      exit(-1);
    }
  }

  // Use one thread for each column in the graph.
  //   All the threads are in one block so we can use __syncthreads().
  //   If the graph is too wide for this approach, report an error and exit.
  if(parg.w > prop.maxThreadsPerBlock) {
    fprintf(stderr, "perc: array size too big, max = %d\n",
      prop.maxThreadsPerBlock);
    exit(-1);
  }

  // allocate an array for the result on the CPU
  int vsz = parg.w*sizeof(uint);
  v = (uint *)malloc(vsz);

  // allocate the result array and pseudo-random number generator states on the GPU
  HANDLE_ERROR(cudaMalloc((void **)(&parg.dev_v), vsz));
  HANDLE_ERROR(cudaMalloc((void **)(&parg.dev_randState), parg.w*sizeof(curandState)));

  // initialize the random number generators. 
  setup_kernel<<<1,parg.w>>>(parg.w, parg.dev_randState);

  // make the timing measurements.
  time_it_run(tr, do_perc, (void *)(&parg));
  // fetch the final state from the GPU
  cudaMemcpy(v, parg.dev_v, vsz, cudaMemcpyDeviceToHost);

  // count the number of marked vertices in the final generation.
  //   If the number is greater than zero, than the last row can be reached
  //   from the first row with a path where every vertex is marked.
  int sum = 0;
  for(uint i = 0; i < parg.w; i++) 
    sum += v[i];
  printf("%d vertices reachable after %d generations\n", sum, parg.h);
  time_it_get_stats(tr, &stats);

  // print the mean and standard deviation of the elapsed times for
  //   running the simulations.
  printf("perc(%u, %u, %5.3f): mean(T) = %10.3le, stddev(T) = %10.3le\n",
            parg.w, parg.h, parg.p, stats.mean, stats.std);

  // clean up
  HANDLE_ERROR(cudaFree(parg.dev_randState));
  HANDLE_ERROR(cudaFree(parg.dev_v));
  time_it_free(tr);
  free(v);
  exit(0);
}
