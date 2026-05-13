#include "tensor.h"

#include <stdio.h>

int main(void) {
  int shape[] = {3, 3};

  Tensor* a = tensor_create_zeros(shape, 2);
  Tensor* b = tensor_create_zeros(shape, 2);
  Tensor* sum = tensor_create_zeros(shape, 2);

  for (size_t i = 0; i < a->total_elements; i++) {
    a->values[i] = (float)i + 1.0f;
    b->values[i] = 10.0f * ((float)i + 1.0f);
  }
  tensor_matmul_2d(sum, a, b);

  printf("a: ");
  tensor_print(a);
  printf("\n");

  printf("b: ");
  tensor_print(b);
  printf("\n");

  printf("a * b: ");
  tensor_print(sum);
  printf("\n");

  tensor_free(a);
  tensor_free(b);
  tensor_free(sum);

  return 0;
}
