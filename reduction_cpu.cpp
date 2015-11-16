#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>

using namespace std;

void getMin(float *input, size_t len, float *output_val, size_t *output_idx) {
  assert(len >= 1);
  *output_val = input[0];
  *output_idx = 0;
  for (size_t i = 1; i < len; ++i) {
    if (input[i] < *output_val) {
      *output_val = input[i];
      *output_idx = i;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 2) {
    cout << "Usage: " << argv[0] << " size\n";
    return -6;
  }
  const size_t len = 1L << (atoi(argv[1]));
  float *a = (float *)malloc(len * sizeof(float));
  if (a == nullptr) {
    cout << "Cannot allocate memory\n";
    exit(-6);
  }
  srand(0);
  clock_t begin = clock();
  for (size_t i = 0; i < len; ++i) {
    a[i] = rand() / (float)RAND_MAX;
  }
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";
  float val;
  size_t idx;
  begin = clock();
  getMin(a, len, &val, &idx);
  cout << "Elapsed time: " << double(clock() - begin) / CLOCKS_PER_SEC * 1000
       << " ms\n";
  cout << "Number of elements: " << len << ", min val: " << val
       << ", min idx: " << idx << "\n";
  free(a);
  return 0;
}
