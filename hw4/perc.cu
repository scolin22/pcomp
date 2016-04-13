/* perc.cu:  a "faster" percolation simulation.
 *  Run a percolation simulation:
 *    ./perc
 *  Find out all the command-line options:
 *    ./perc help
 */

/* This file is most of my solution -- I just deleted the "interesting"
 *    CUDA parts.  I am expecting you to use this.  Here's a quick list
 *    of the functions and one struct declared in this file.
 *
 *      struct kernel_arg -- arguments for the kernels.  I pass a pointer
 *        to a struct kernel_arg value as my argument to time_it_run.
 *
 *      typedef curandState randstate_t -- I added this typedef to make
 *        it easier to switch to using other pseudo-random number generators
 *        from the curand package.
 *
 *      __global__ void setup(uint n, randstate_t *state): a CUDA kernel
 *        It initializes the pseudo-random number generator states.
 *
 *      __global__ void rndb(uint n, uint m, uint q, randstate_t *randState, uint *rbits):
 *        a CUDA kernel.  It generates an array (in global memory) of random bits.
 *
 *      __global__ void perc(uint m, uint *randBits, uint *v): a CUDA kernel
 *        It simulates a percolation network.
 *
 *      uint count_survivors(kernel_arg *argk): host function.
 *        Count the number of marked vertices in the final generation of a
 *        percolation network simulation.
 *
 *      void do_perc(void *void_arg): host function
 *        This is the function that gets called by time_it_run.  It launches
 *        the kernel(s) to be timed, and does a cudaDeviceSynchronize.
 *
 *      void usage(FILE *f): host function
 *        Prints a "usage" message (i.e. how to run this program from the
 *        command line.
 *
 *      void give_up(const char *why, const char *arg): host function
 *        Prints an error message to describe a bad command line option.
 *        Then the program exits.
 *
 *      void do_args(int argc, char **argv, kernel_arg *argk): host function
 *        Processes command line arguments.
 *
 *      main(int argc, char **argv): host function
 *        The top-level function.  Processes command line arguments.  Allocates
 *        and initializes data structures.  Performs the timing measurements.
 *        Prints results.  Cleans-up. Exits.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "time_it.h"

#define BITS_PER_UINT 32
#define WARPSIZE 32

// I used a typedef so I can switch to using a different random number generator
//   from curand fairly easily.
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

// kernel_arg -- arguments for the kernels
//   Notes:
//     dev_randBits is an array of rand_nblks*rand_tpb*rand_m uint's.
//     For now, the percolation thread just uses one warp of threads.
//       *dev_v is an array of warpSize uints.
struct kernel_arg {
  // which kernels to time
  bool time_setup,  // include initialization of the random-number generator
       time_rndb,   // include generation of random bits
       time_perc;   // include the percolation simulation

  // stuff for the random-bit generator
  uint rand_nblks,  // how many blocks to use
       rand_tpb,    // how many threads per block
       rand_m,      // how many uints of random bits per thread
       rand_n;      // how many uints of random bits per thread
  randstate_t *dev_randState; // prng state
  uint *dev_randBits; // put the uints of random bits here

  // stuff for percolation
  uint perc_m; // number of percolation generations to simulate
  double p;     // vertex probability
  uint *dev_v; // the initial state is placed here, and the final
               // state will be written here.
};


// Initialize the prng states.  From
//   http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
__global__ void setup_kernel(uint n, randstate_t *state)
{
  uint myId = blockDim.x * blockIdx.x + threadIdx.x;
  /* Each thread gets same seed, a different sequence number, no offset */
  if(myId < n)
    curand_init(1234, myId, 0, &state[myId]);
}

// calculate m uints of random bits.
//   each bit is 1 with probability p, where p = q/UINT_MAX
__global__ void rndb(uint n, uint m, uint q, randstate_t *randState, uint *rbits) {
  uint myId = blockDim.x * blockIdx.x + threadIdx.x;
  uint n_threadsTotal = gridDim.x * blockDim.x;
  randstate_t *myRandState = &(randState[myId]);

  if(myId < n) {
    for(int j = 0; j < m; j++) {  // each word generated by this thread
      uint w = 0;
      for(int b = 0; b < BITS_PER_UINT; b++) // make a word of random bits
        // TODO: write this
        w |= (curand_uniform(myRandState) <= q/UINT_MAX) << b;
      rbits[n_threadsTotal*j + myId] = w;
    }
  }
}

/* perc: a faster simulation of percolation.
 *  use bit-wise logicals to update 32 vertices at a time.
 *  assume graph width is at most 1024 -- this means we run in a single
 *    warp and don't need to call __syncthreads()
 *  random-bits are pre-computed using rndb above -- this lets us use
 *    more threads and blocks for the random bit calculation.
 *  I assume that m is a multiple of BITS_PER_UINT -- otherwise, we'll
 *    do slightly less work than requested.
 */
__global__ void perc(uint m, uint *randBits, uint *v) {
  // TODO: declare a array to share data between threads
  __shared__ uint state[blockDim.x];
  uint myId = threadIdx.x;
  uint left = (myId == 0) ? (WARPSIZE-1) : (myId-1);
  uint *rb = &(randBits[0]);

  uint64_t x = UINT64_MAX;
  for(int j = 0; j < m/BITS_PER_UINT; j++) {
     // TODO: write my state to shared memory
    state[myId] = v[myId];
     // TODO: read my left neighbour's state
    uint left_state = v[left];
     // TODO: make a 64 bit word from my state and my neighbour's state
    x = left_state << BITS_PER_UINT | state[myId];
    for(int k = 0; k < BITS_PER_UINT; k++) {
      // TODO: get the random bits for me and my neighbour
      uint r = rb[left] << BITS_PER_UINT | rb[myId];
      // TODO: update my state, and my representation of my neighbour's state.
      //   Note: This "loses" on state of my left-neighbour each cycle because
      //     we don't know its left neighbour.  That's why we can do this for
      //     BITS_PER_UINT steps, and then have to coordinate through shared
      //     memory.  Note that we need to read the global memory to get
      //     random bits with each iteration of this inner loop.  Also,
      //     we need to update our copy of the left-neighbour using the
      //     same random bits that our left neighbour uses.
      x = (x | (x >> 1)) & r;
    }
  }
  v[myId] = x & (UINT_MAX);
}

// count_survivors: read the final row of vertices from the perolation
//   simulation and count how many vertices are marked.  This is only
//   called once by main to print out the final tally.  I wrapped this
//   up as a separate function because I found it useful when debugging.
uint count_survivors(kernel_arg *argk) {
  uint v[WARPSIZE];

  HANDLE_ERROR(cudaMemcpy(v, argk->dev_v, sizeof(v), cudaMemcpyDeviceToHost));
  uint sum = 0;
  for(uint i = 0; i < WARPSIZE; i++) {
    uint vv = v[i];
    for(uint b = 0; b < BITS_PER_UINT; b++)
      sum += (vv >> b) & 1;
  }
  return(sum);
}


// do_perc: this is the function that time_it_run will call
//   We check the time_setup, time_rndb, and time_perc flags from argk
//   and launch the requested kernels.  We do a cudaDeviceSynchronize
//   at the end.  This approach works because:
//     1.  The percolation kernel sets an initial state of all vertices
//           reachable.
//     2.  The percolation doesn't modify the random number generator
//           states or the array of random bits from the rndb kernel.
//     3.  It's OK if we run time_rndb again -- we'll just run the
//           percolation kernel with different random bits.
//   Note that you can add copies of
//     HANDLE_ERROR(cudaDeviceSynchronize());
//   after each kernel launch when debugging -- I found that makes
//   it easier to track down bugs.
void do_perc(void *void_arg) {
  kernel_arg *argk = (kernel_arg *)(void_arg);
  uint q = UINT_MAX*(argk->p); // scale p so rndb can use curand(...)
                               // instead of curand_uniform(...)
  if(argk->time_setup) {
    setup_kernel<<<argk->rand_nblks, argk->rand_tpb>>>
                (argk->rand_n, argk->dev_randState);
  }
  if(argk->time_rndb) {
    rndb<<<argk->rand_nblks, argk->rand_tpb>>>
        (argk->rand_n, argk->rand_m, q, argk->dev_randState, argk->dev_randBits);
  }
  if(argk->time_perc) {
    perc<<<1,WARPSIZE>>>
        (argk->perc_m, argk->dev_randBits, argk->dev_v);
  }
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void usage(FILE *f) {
  const char *text[] = {
    "   usage ./perc [options]",
    "   options:",
    "     help           Print this message, and then exit.",
    "     --help         Same as \"help\".",
    "     n=INT          The number of blocks to use for random-bit generation.",
    "     tpb=INT        The number of threads-per-block for random-bit",
    "                       generation.",
    "     m=INT          The number of random-uints per thread, for random-bit",
    "                       generation.",
    "     p=DOUBLE       The probability that a random bit is 1.  p should be",
    "                       between 0.0 and 1.0.",
    "     time_perc=INT  If non-zero, then the time to execute the percolation",
    "                       kernel will be included in the timing measurements.",
    "                       Default: 1.",
    "     time_rndb=INT  If non-zero, then the time to execute the random-bit",
    "                       generation kernel will be included in the timing",
    "                       measurements.  Default: 1.",
    "     time_setup=INT If non-zero, then the time to execute the random-bit",
    "                       generation kernel will be included in the timing",
    "                       measurements.  Default: 1.",
    NULL
  };
  for(int i = 0; text[i] != NULL; i++)
    fprintf(f, "%s\n", text[i]);
}

// give_up: report a bad command line option and exit.
void give_up(const char *why, const char *arg) {
  fprintf(stderr, "bad option: %s, %s\n", arg, why);
  fprintf(stderr, "for help, run\n");
  fprintf(stderr, "  ./perc help\n");
  exit(-1);
}

// do_args: process command line options and initialize argk.
//   The options set various fields of argk.
//   See the usage() to get a description of all of the options.
void do_args(int argc, char **argv, kernel_arg *argk) {
  // set default values for fields of argk
  argk->time_setup = false;
  argk->time_rndb = true;
  argk->time_perc = true;
  argk->rand_nblks = 48;
  argk->rand_tpb = 256;
  argk->rand_m = 256;
  argk->p = 0.75;

  // process command line options
  for(int i = 1; i < argc; i++) {
    if((strcmp(argv[i], "help") == 0) || (strcmp(argv[i], "--help") == 0)) {
      usage(stdout);
      exit(0);
    } else if(strncmp(argv[i], "m=", strlen("m=")) == 0) {
      argk->rand_m = strtoul(argv[i]+strlen("m="), NULL, 10);
      if(argk->rand_m == 0)
        give_up("m must be non-zero", argv[i]);
    } else if(strncmp(argv[i], "nblks=", strlen("nblks=")) == 0) {
      argk->rand_nblks = strtoul(argv[i]+strlen("nblks="), NULL, 10);
      if(argk->rand_nblks == 0)
        give_up("nblks must be non-zero", argv[i]);
    } else if(strncmp(argv[i], "p=", strlen("p=")) == 0) {
      argk->p = strtod(argv[i]+strlen("p="), NULL);
      if((argk->p < 0.0) || (1.0 < argk->p))
        give_up("p must be between 0.0 and 1.0.", argv[i]);
    } else if(strncmp(argv[i], "time_perc=", strlen("time_perc=")) == 0) {
      argk->time_perc = strtol(argv[i]+strlen("time_perc="), NULL, 10) != 0;
    } else if(strncmp(argv[i], "time_rndb=", strlen("time_rndb=")) == 0) {
      argk->time_rndb = strtol(argv[i]+strlen("time_rndb="), NULL, 10) != 0;
    } else if(strncmp(argv[i], "time_setup=", strlen("time_setup=")) == 0) {
      argk->time_setup = strtol(argv[i]+strlen("time_setup="), NULL, 10) != 0;
    } else if(strncmp(argv[i], "tpb=", strlen("tpb=")) == 0) {
      argk->rand_tpb = strtoul(argv[i]+strlen("tpb="), NULL, 10);
      if(argk->rand_tpb == 0)
        give_up("tpb must be non-zero", argv[i]);
      else if((argk->rand_tpb % WARPSIZE) != 0) {
        char buf[1000];
        sprintf(buf, "tpb must be a multple of %d (the warpsize)", WARPSIZE);
        give_up(buf, argv[i]);
      }
    } else {
      give_up("unknown option", argv[i]);
    }
  }
}

main(int argc, char **argv) {
  int ndev;
  struct time_it_raw *tr = time_it_create(10);
  struct time_it_stats stats;
  kernel_arg argk;

  // make sure we have a GPU
  HANDLE_ERROR(cudaGetDeviceCount(&ndev));
  if(ndev < 1) {
    fprintf(stderr, "No CUDA device found\n");
    exit(-1);
  }

  // process command line arguments
  do_args(argc, argv, &argk);

  // The following code allocates and initializes the arrays we use.
  // I check that the size calculations don't overflow.  First, I
  // calculate the number of threads that will be used for random number
  // generation, as a double.  This guarantees that the calculation won't
  // overflow.
  double rand_n_dbl = ((double)argk.rand_tpb)*((double)argk.rand_nblks);
  argk.rand_n = rand_n_dbl;

  // allocate and initialize the random number generator states
  if(rand_n_dbl*sizeof(randstate_t) > UINT_MAX) {
    fprintf(stderr,
      "prng state array is too big to allocate: %lf entries (must be at most %u)\n",
      rand_n_dbl, UINT_MAX/sizeof(randstate_t));
    exit(-1);
  }
  uint sz_rnd_st = argk.rand_n*sizeof(randstate_t);
  HANDLE_ERROR(cudaMalloc((void **)(&argk.dev_randState), sz_rnd_st));
  if(!(argk.time_setup && argk.time_rndb)) {
    // If we don't initialize the random number generator states with each
    // timing measurment (the default), then we'll do it here.
    // A special case: the command line could ask to time the setup kernel
    // but not the actual random number generation.  In that case, we'll do
    // the random number generation below, so we initialize the random
    // number generator states here.
    setup_kernel<<<argk.rand_nblks, argk.rand_tpb>>>
                (argk.rand_n, argk.dev_randState);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // allocate the array of random bits
  if(rand_n_dbl*argk.rand_m*sizeof(uint) > UINT_MAX) {
    fprintf(stderr,
      "random bit array is too big to allocate: %lf entries (must be at most %u)\n",
      rand_n_dbl, UINT_MAX/sizeof(uint));
    exit(-1);
  }
  uint sz_rndb = argk.rand_n*argk.rand_m*sizeof(uint);
  HANDLE_ERROR(cudaMalloc((void **)(&argk.dev_randBits), sz_rndb));
  if(!argk.time_rndb) {
    uint q = UINT_MAX*(argk.p);
    rndb<<<argk.rand_nblks, argk.rand_tpb>>>
        (argk.rand_n, argk.rand_m, q, argk.dev_randState, argk.dev_randBits);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // allocate the array of percolation bits
  HANDLE_ERROR(cudaMalloc((void **)(&argk.dev_v), WARPSIZE*sizeof(uint)));
  // the calculation for perc_m can't overflow because of the check performed
  // above for sz_rndb.
  argk.perc_m = argk.rand_n*argk.rand_m/WARPSIZE;

  // now we can measure the execution times -- yay!!!
  time_it_run(tr, do_perc, (void *)(&argk));

  if(!argk.time_perc) { // run the percolation kernel once to get the results
    perc<<<1,WARPSIZE>>>(argk.perc_m, argk.dev_randBits, argk.dev_v);
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

  // report the results
  printf("%d vertices reachable after %d generations\n",
    count_survivors(&argk), argk.perc_m);
  time_it_get_stats(tr, &stats);
  printf("perc(%lf): mean(T) = %10.3le, stddev(T) = %10.3le\n",
         (argc < 4) ? "perc" : "rndm",
         argk.p, stats.mean, stats.std);
  printf("throughput = %10.4e vertices/second\n",
    ((double)(argk.perc_m))*((double)(WARPSIZE*BITS_PER_UINT))/(stats.mean));

  // clean up, and exit
  HANDLE_ERROR(cudaFree(argk.dev_randState));
  HANDLE_ERROR(cudaFree(argk.dev_randBits));
  HANDLE_ERROR(cudaFree(argk.dev_v));
  exit(0);
}
