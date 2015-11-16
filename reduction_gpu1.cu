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

__global__ void getMin(float *input, size_t len, float *output_val,
                       size_t *output_idx) {
  __shared__ float smem_val[BLOCK_SIZE];
  __shared__ size_t smem_idx[BLOCK_SIZE];

  int tx = threadIdx.x;

  size_t i = tx + blockIdx.x * BLOCK_SIZE * 4;

  float min_val = INFINITY;
  size_t min_idx = i;

  if (i < len) {
    float a1, a2, a3, a4;
    a1 = input[i];
    min_val = a1;
    min_idx = i;
    if ((i + BLOCK_SIZE) < len) {
      a2 = input[i + BLOCK_SIZE];
      if (a2 < min_val) {
        min_val = a2;
        min_idx = i + BLOCK_SIZE;
      }
    }
    if ((i + 2 * BLOCK_SIZE) < len) {
      a3 = input[i + 2 * BLOCK_SIZE];
      if (a3 < min_val) {
        min_val = a3;
        min_idx = i + 2 * BLOCK_SIZE;
      }
    }
    if (i + 3 * BLOCK_SIZE < len) {
      a4 = input[i + 3 * BLOCK_SIZE];
      if (a4 < min_val) {
        min_val = a4;
        min_idx = i + 3 * BLOCK_SIZE;
      }
    }
  }

  smem_val[tx] = min_val;
  smem_idx[tx] = min_idx;
  __syncthreads();

  // in-place reduction in shared memory
  if (blockDim.x >= 1024 && tx < 512) {
    if (smem_val[tx + 512] < smem_val[tx]) {
      smem_val[tx] = smem_val[tx + 512];
      smem_idx[tx] = smem_idx[tx + 512];
    }
  }
  __syncthreads();

  if (blockDim.x >= 512 && tx < 256) {
    if (smem_val[tx + 256] < smem_val[tx]) {
      smem_val[tx] = smem_val[tx + 256];
      smem_idx[tx] = smem_idx[tx + 256];
    }
  }
  __syncthreads();

  if (blockDim.x >= 256 && tx < 128) {
    if (smem_val[tx + 128] < smem_val[tx]) {
      smem_val[tx] = smem_val[tx + 128];
      smem_idx[tx] = smem_idx[tx + 128];
    }
  }
  __syncthreads();

  if (blockDim.x >= 128 && tx < 64) {
    if (smem_val[tx + 64] < smem_val[tx]) {
      smem_val[tx] = smem_val[tx + 64];
      smem_idx[tx] = smem_idx[tx + 64];
    }
  }
  __syncthreads();

  // unrolling warp
  if (tx < 32) {
    volatile float *vsmem_val = smem_val;
    volatile size_t *vsmem_idx = smem_idx;
    if (vsmem_val[tx + 32] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 32];
      vsmem_idx[tx] = vsmem_idx[tx + 32];
    }
    if (vsmem_val[tx + 16] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 16];
      vsmem_idx[tx] = vsmem_idx[tx + 16];
    }
    if (vsmem_val[tx + 8] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 8];
      vsmem_idx[tx] = vsmem_idx[tx + 8];
    }
    if (vsmem_val[tx + 4] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 4];
      vsmem_idx[tx] = vsmem_idx[tx + 4];
    }
    if (vsmem_val[tx + 2] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 2];
      vsmem_idx[tx] = vsmem_idx[tx + 2];
    }
    if (vsmem_val[tx + 1] < vsmem_val[tx]) {
      vsmem_val[tx] = vsmem_val[tx + 1];
      vsmem_idx[tx] = vsmem_idx[tx + 1];
    }
  }

  if (tx == 0) {
    output_val[blockIdx.x] = smem_val[0];
    output_idx[blockIdx.x] = smem_idx[0];
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 2) {
    cout << "Usage: " << argv[0] << " size\n";
    return -1;
  }
  const size_t len = 1L << (atoi(argv[1]));
  float *h_a = (float *)malloc(len * sizeof(float));

  if (h_a == nullptr) {
    cout << "Cannot allocate memory\n";
    exit(-1);
  }
  srand(0);
  clock_t begin = clock();
  for (size_t i = 0; i < len; ++i) {
    h_a[i] = rand() / (float)RAND_MAX;
  }
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";

  begin = clock();
  size_t len_out = ceil((float)len / (BLOCK_SIZE * 4));
  float *h_val = (float *)malloc(sizeof(float) * len_out);
  size_t *h_idx = (size_t *)malloc(sizeof(size_t) * len_out);

  float *d_a;
  float *d_val;
  size_t *d_idx;

  CHECK(cudaMalloc((void **)&d_a, sizeof(float) * len));
  CHECK(cudaMalloc((void **)&d_val, sizeof(float) * len_out));
  CHECK(cudaMalloc((void **)&d_idx, sizeof(size_t) * len_out));

  CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * len, cudaMemcpyHostToDevice));

  getMin<<<len_out, BLOCK_SIZE>>>(d_a, len, d_val, d_idx);

  CHECK(cudaMemcpy(h_val, d_val, sizeof(float) * len_out,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_idx, d_idx, sizeof(size_t) * len_out,
                   cudaMemcpyDeviceToHost));

  CHECK(cudaDeviceSynchronize());

  float val = h_val[0];
  size_t idx = h_idx[0];
  for (size_t i = 0; i < len_out; ++i) {
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