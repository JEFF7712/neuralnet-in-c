#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdbool.h>

typedef struct {
  float* values;
  int* shape;
  int* strides;
  int rank;
  size_t total_elements;
}Tensor;

/*
 * values = flat array of the data: {1, 2, 3, 4, 5, 6, 7, 8, 9}
 * shape = array of the dimensions: {rows, cols}
 * strides = step size for each dimension: for [[1, 2, 3], values is [1, 2, 3, 4, 5, 6, 7, 8, 9].
                                                [4, 5, 6], To move from 1 to 2, you move 1 step. So, strides[1] = 1.
                                                [7, 8, 9]] To move from 1 to 4 (the next row), that is 3 steps. So, strides[0] = 3
 * rank = the number of dimensions: 1 = vector, 2 = matrix, 3 = cube
 * total_elements = the total number of elements in the tensor: # of elements in values array
 */

Tensor* tensor_create(int* shape, int rank);
Tensor* tensor_create_zeros(int* shape, int rank);
Tensor* tensor_clone(const Tensor* t);
void tensor_zeros(Tensor* t);
void tensor_ones(Tensor* t);
void tensor_rand(Tensor* t, float low, float high);
void tensor_free(Tensor* t);
bool tensor_shape_equal(const Tensor* a, const Tensor* b);
void tensor_add(Tensor* dest, const Tensor* a, const Tensor* b);
void tensor_add_scalar(Tensor* dest, const Tensor* a, float scalar);
void tensor_mul_elementwise(Tensor* dest, const Tensor* a, const Tensor* b);
void tensor_matmul_2d(Tensor* dest, const Tensor* a, const Tensor* b);
void tensor_transpose(Tensor* t, int dim0, int dim1);
int tensor_argmax(const Tensor* t);
void tensor_print(const Tensor* t);

#endif