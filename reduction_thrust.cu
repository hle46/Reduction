#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <iostream>
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 2) {
    std::cout << "Usage: " << argv[0] << " size\n";
    return -1;
  }
  const size_t len = 1L << (atoi(argv[1]));
  thrust::host_vector<int> h_a(len);

  srand(0);
  clock_t begin = clock();
  for (size_t i = 0; i < len; ++i) {
    h_a[i] = rand() / (float)RAND_MAX;
  }
  std::cout << "Elapsed time: "
            << double(clock() - begin) / CLOCKS_PER_SEC * 1000 << " ms\n";

  // Copy host_vector to device_vector
  thrust::device_vector<int> d_a = h_a;
  thrust::min_element(d_a.begin(), d_a.end());

  return 0;
}