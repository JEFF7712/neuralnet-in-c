#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

Tensor* tensor_create(int* shape, int rank) {
  Tensor* t = (Tensor*)malloc(sizeof(Tensor));
  
  t->rank = rank;
  t->shape = (int*)malloc(rank * sizeof(int));
  t->strides = (int*)malloc(rank * sizeof(int));
  
  memcpy(t->shape, shape, sizeof(int) * rank);
  
  int total_elements = 1;
  int current_stride = 1;
  
  for (int i = rank - 1; i >= 0; i--) {
    t->strides[i] = current_stride;
    current_stride *= shape[i];
    total_elements *= shape[i];
  }
  
  t->total_elements = total_elements;
  
  t->values = (float*)malloc(total_elements * sizeof(*t->values));
  
  return t;
}

Tensor* tensor_create_zeros(int* shape, int rank) {
  Tensor* t = (Tensor*)malloc(sizeof(Tensor));
  
  t->rank = rank;
  t->shape = (int*)malloc(rank * sizeof(int));
  t->strides = (int*)malloc(rank * sizeof(int));
  
  memcpy(t->shape, shape, sizeof(int) * rank);
  
  int total_elements = 1;
  int current_stride = 1;
  
  for (int i = rank - 1; i >= 0; i--) {
    t->strides[i] = current_stride;
    current_stride *= shape[i];
    total_elements *= shape[i];
  }
  
  t->total_elements = total_elements;
  
  t->values = (float*)calloc(total_elements, sizeof(*t->values));
  
  return t;
}

Tensor* tensor_clone(const Tensor* t) {
  if (!t) return NULL;
  
  Tensor* clone = (Tensor*)malloc(sizeof(Tensor));
  
  clone->rank = t->rank;
  clone->total_elements = t->total_elements;
  
  clone->values = (float*)malloc(t->total_elements * sizeof(*t->values));
  clone->shape = (int*)malloc(t->rank * sizeof(int));
  clone->strides = (int*)malloc(t->rank * sizeof(int));
  
  memcpy(clone->values, t->values, t->total_elements * sizeof(*t->values));
  memcpy(clone->shape, t->shape, t->rank * sizeof(int));
  memcpy(clone->strides, t->strides, t->rank * sizeof(int));
  
  return clone;
}

void tensor_zeros(Tensor* t) {  
  memset(t->values, 0, sizeof(float) * t->total_elements);
}

void tensor_ones(Tensor* t) {  
  for (size_t i = 0; i < t->total_elements; i++) {
    t->values[i] = 1;
  }
}

void tensor_rand(Tensor* t, float low, float high) {  
  for (size_t i = 0; i < t->total_elements; i++) {
    t->values[i] = ((float)rand() / (float)RAND_MAX) * (high - low) + low;
  }
}

void tensor_free(Tensor* t) {
  if (!t) return;
  free(t->values);
  free(t->shape);
  free(t->strides);
  free(t);
}

bool tensor_shape_equal(const Tensor* a, const Tensor* b) {
  if (a->rank != b->rank) return false;
  return memcmp(a->shape, b->shape, a->rank * sizeof(int)) == 0;
}

void tensor_add(Tensor* dest, const Tensor* a, const Tensor* b) {
  assert(tensor_shape_equal(a, b) && "Operand shapes must match");
  assert(tensor_shape_equal(a, dest) && "Destination shape must match operands");
  
  for (size_t i = 0; i < dest->total_elements; i++) {
    dest->values[i] = a->values[i] + b->values[i];
  }
}

void tensor_add_scalar(Tensor* dest, const Tensor* a, float scalar) {
  assert(tensor_shape_equal(a, dest) && "Destination shape must match operands");
  
  for (size_t i = 0; i < dest->total_elements; i++) {
    dest->values[i] = a->values[i] + scalar;
  }
}

void tensor_mul_elementwise(Tensor* dest, const Tensor* a, const Tensor* b) {
  assert(tensor_shape_equal(a, b) && "Operand shapes must match");
  assert(tensor_shape_equal(a, dest) && "Destination shape must match operands");
  
  for (size_t i = 0; i < dest->total_elements; i++) {
    dest->values[i] = a->values[i] * b->values[i];
  }
  
}

void tensor_matmul_2d(Tensor* dest, const Tensor* a, const Tensor* b) {
  assert(a->rank == 2 && b->rank == 2 && dest->rank == 2 && "Tensors must be 2D");
  assert(a->shape[1] == b->shape[0] && "Inner dimensions must match (A cols == B rows)");
  assert(dest->shape[0] == a->shape[0] && dest->shape[1] == b->shape[1] && "Invalid dest shape");

  int M = a->shape[0]; // Rows of A
  int K = a->shape[1]; // Cols of A / Rows of B
  int N = b->shape[1]; // Cols of B
  
  tensor_zeros(dest);
  
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        size_t idx_a = (size_t)i * a->strides[0] + (size_t)k * a->strides[1];
        size_t idx_b = (size_t)k * b->strides[0] + (size_t)j * b->strides[1];
        
        sum += a->values[idx_a] * b->values[idx_b];
      }
      size_t idx_dest = (size_t)i * dest->strides[0] + (size_t)j * dest->strides[1];
      dest->values[idx_dest] = sum;
    }
  }
}

void tensor_transpose(Tensor* t, int dim0, int dim1) {
  assert(t != NULL && "Tensor cannot be NULL");
  assert(dim0 >= 0 && dim0 < t->rank && "dim0 is out of bounds");
  assert(dim1 >= 0 && dim1 < t->rank && "dim1 is out of bounds");

  if (dim0 == dim1) return;

  int temp_shape = t->shape[dim0];
  t->shape[dim0] = t->shape[dim1];
  t->shape[dim1] = temp_shape;
  
  int temp_stride = t->strides[dim0];
  t->strides[dim0] = t->strides[dim1];
  t->strides[dim1] = temp_stride;
}

static void print_recursive(const Tensor* t, int dim, size_t offset) {
  if (t->rank == 0) {
    printf("%f", t->values[0]);
    return;
  }
  
  printf("[");
  
  for (int i = 0; i < t->shape[dim]; i++) {
    size_t current_offset = offset + (i * t->strides[dim]);

    if (dim == t->rank - 1) {
      printf("%.4f", t->values[current_offset]);
    } else {
      print_recursive(t, dim + 1, current_offset);
    }

    if (i < t->shape[dim] - 1) {
      if (dim == t->rank - 1) {
        printf(", ");
      } else {
        printf(",\n");
        for (int spaces = 0; spaces <= dim; spaces++) {
          printf(" ");
        }
      }
    }
  }
  printf("]");
}

int tensor_argmax(const Tensor* t) {
  int max_idx = 0;
  float max = t->values[0];
  for (size_t i = 0; i < t->total_elements; i++) {
    if (t->values[i] > max) {
      max = t->values[i];
      max_idx = i;
    }
  }
  
  return max_idx;
}

void tensor_print(const Tensor* t) {
  if (!t) return;
    print_recursive(t, 0, 0);
    printf("\n");
  }
