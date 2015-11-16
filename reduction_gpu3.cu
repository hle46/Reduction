#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define BLOCK_SIZE 128
#define CHECK(call)                                                            \
  {                                                                            \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                   \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  }

using namespace std;

__global__ void getMin(float *input, int len, float *output_val,
                       int *output_idx) {
  __shared__ float smem_val[BLOCK_SIZE];
  __shared__ int smem_idx[BLOCK_SIZE];

  int tx = threadIdx.x;

  int i = tx + blockIdx.x * BLOCK_SIZE * 8;

  float min_val = INFINITY;
  int min_idx = i;

  if (i < len) {
    float a1, a2, a3, a4, a5, a6, a7, a8;
    a1 = input[i];
    a2 = (i + BLOCK_SIZE) < len ? input[i + BLOCK_SIZE] : INFINITY;
    a3 = (i + 2 * BLOCK_SIZE) < len ? input[i + 2 * BLOCK_SIZE] : INFINITY;
    a4 = (i + 3 * BLOCK_SIZE) < len ? input[i + 3 * BLOCK_SIZE] : INFINITY;
    a5 = (i + 4 * BLOCK_SIZE) < len ? input[i + 4 * BLOCK_SIZE] : INFINITY;
    a6 = (i + 5 * BLOCK_SIZE) < len ? input[i + 5 * BLOCK_SIZE] : INFINITY;
    a7 = (i + 6 * BLOCK_SIZE) < len ? input[i + 6 * BLOCK_SIZE] : INFINITY;
    a8 = (i + 7 * BLOCK_SIZE) < len ? input[i + 7 * BLOCK_SIZE] : INFINITY;
    min_val = a1;
    min_idx = i;
    if (a2 < min_val) {
      min_val = a2;
      min_idx = i + BLOCK_SIZE;
    }
    if (a3 < min_val) {
      min_val = a3;
      min_idx = i + 2 * BLOCK_SIZE;
    }
    if (a4 < min_val) {
      min_val = a4;
      min_idx = i + 3 * BLOCK_SIZE;
    }
    if (a5 < min_val) {
      min_val = a5;
      min_idx = i + 4 * BLOCK_SIZE;
    }
    if (a6 < min_val) {
      min_val = a6;
      min_idx = i + 5 * BLOCK_SIZE;
    }
    if (a7 < min_val) {
      min_val = a7;
      min_idx = i + 6 * BLOCK_SIZE;
    }
    if (a8 < min_val) {
      min_val = a8;
      min_idx = i + 7 * BLOCK_SIZE;
    }
  }

  smem_val[tx] = min_val;
  smem_idx[tx] = min_idx;
  __syncthreads();

  // in-place reduction in shared memory
  if (blockDim.x >= 1024 && tx < 512 && smem_val[tx + 512] < smem_val[tx]) {
    smem_val[tx] = min_val = smem_val[tx + 512];
    smem_idx[tx] = min_idx = smem_idx[tx + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tx < 256 && smem_val[tx + 256] < smem_val[tx]) {
    smem_val[tx] = min_val = smem_val[tx + 256];
    smem_idx[tx] = min_idx = smem_idx[tx + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tx < 128 && smem_val[tx + 128] < smem_val[tx]) {
    smem_val[tx] = min_val = smem_val[tx + 128];
    smem_idx[tx] = min_idx = smem_idx[tx + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tx < 64 && smem_val[tx + 64] < smem_val[tx]) {
    smem_val[tx] = min_val = smem_val[tx + 64];
    smem_idx[tx] = min_idx = smem_idx[tx + 64];
  }
  __syncthreads();

  // unrolling warp
  if (tx < 32) {
    volatile float *vsmem_val = smem_val;
    volatile int *vsmem_idx = smem_idx;
    if (vsmem_val[tx + 32] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 32];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 32];
    }
    if (vsmem_val[tx + 16] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 16];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 16];
    }
    if (vsmem_val[tx + 8] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 8];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 8];
    }
    if (vsmem_val[tx + 4] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 4];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 4];
    }
    if (vsmem_val[tx + 2] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 2];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 2];
    }
    if (vsmem_val[tx + 1] < vsmem_val[tx]) {
      vsmem_val[tx] = min_val = vsmem_val[tx + 1];
      vsmem_idx[tx] = min_idx = vsmem_idx[tx + 1];
    }
  }

  if (tx == 0) {
    output_val[blockIdx.x] = min_val;
    output_idx[blockIdx.x] = min_idx;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 2) {
    cout << "Usage: " << argv[0] << " size\n";
    return -1;
  }
  const int len = 1L << (atoi(argv[1]));
  float *h_a = (float *)malloc(len * sizeof(float));

  if (h_a == nullptr) {
    cout << "Cannot allocate memory\n";
    exit(-1);
  }
  srand(0);
  clock_t begin = clock();
  for (int i = 0; i < len; ++i) {
    h_a[i] = rand() / (float)RAND_MAX;
  }
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";

  begin = clock();
  int len_out = ceil((float)len / (BLOCK_SIZE * 8));
  float *h_val = (float *)malloc(sizeof(float) * len_out);
  int *h_idx = (int *)malloc(sizeof(int) * len_out);

  float *d_a;
  float *d_val;
  int *d_idx;


  CHECK(cudaMalloc((void **)&d_a, sizeof(float) * len));
  CHECK(cudaMalloc((void **)&d_val, sizeof(float) * len_out));
  CHECK(cudaMalloc((void **)&d_idx, sizeof(int) * len_out));

  CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * len, cudaMemcpyHostToDevice));

  getMin<<<len_out, BLOCK_SIZE>>>(d_a, len, d_val, d_idx);

  CHECK(cudaMemcpy(h_val, d_val, sizeof(float) * len_out,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_idx, d_idx, sizeof(int) * len_out,
                   cudaMemcpyDeviceToHost));

  CHECK(cudaDeviceSynchronize());

  float val = h_val[0];
  int idx = h_idx[0];
  for (int i = 0; i < len_out; ++i) {
    if (h_val[i] < val) {
      val = h_val[i];
      idx = h_idx[i];
    }
  }
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";

  cout << "Number of elements: " << len << ", min val: " << val
       << ", min idx: " << idx << "\n";

  // Free device
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_val));
  CHECK(cudaFree(d_idx));

  // Free host
  free(h_a);
  free(h_val);
  free(h_idx);
  return 0;
}
