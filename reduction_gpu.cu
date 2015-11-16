#include <iostream>

#define BLOCK_SIZE 512
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
  __shared__ float smem_val[BLOCK_SIZE * 2];
  __shared__ size_t smem_idx[BLOCK_SIZE * 2];

  int tx = threadIdx.x;

  size_t i = tx + blockIdx.x * BLOCK_SIZE * 2;

  if (i < len) {
    smem_val[tx] = input[i];
    smem_idx[tx] = i;
  } else {
    smem_val[tx] = INFINITY;
    // Don't care about smem_idx;
  }

  if (i + BLOCK_SIZE < len) {
    smem_val[tx + BLOCK_SIZE] = input[i + BLOCK_SIZE];
    smem_idx[tx + BLOCK_SIZE] = i + BLOCK_SIZE;
  } else {
    smem_val[tx + BLOCK_SIZE] = INFINITY;
    // Don't care about smem_idx;
  }

  for (int stride = BLOCK_SIZE; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
      if (smem_val[tx + stride] < smem_val[tx]) {
        smem_val[tx] = smem_val[tx + stride];
        smem_idx[tx] = smem_idx[tx + stride];
      }
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
    return -6;
  }
  const size_t len = 1L << (atoi(argv[1]));
  float *h_a = (float *)malloc(len * sizeof(float));

  if (h_a == nullptr) {
    cout << "Cannot allocate memory\n";
    exit(-6);
  }
  srand(0);
  clock_t begin = clock();
  for (size_t i = 0; i < len; ++i) {
    h_a[i] = rand() / (float)RAND_MAX;
  }
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";

  begin = clock();
  size_t len_out = ceil((float)len / (BLOCK_SIZE << 1));
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
