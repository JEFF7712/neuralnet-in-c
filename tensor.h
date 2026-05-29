#ifndef TENSOR_H
#define TENSOR_H

#include "arena.h"

#include <stdlib.h>
#include <stdbool.h>

typedef struct Tensor{
  float* values;
  float* grad;
  int* shape;
  int* strides;
  int rank;
  size_t total_elements;
  
  bool requires_autograd;
  void (*backward_fn)(struct Tensor*);
  struct Tensor** parents;
  int num_parents;
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

typedef struct {
  Arena* arena;
  
  Tensor** gradient_tape;
  int tape_count;
  int tape_capacity;
} GraphContext;

GraphContext* ctx_create(size_t initial_memory);
void ctx_reset(GraphContext* ctx);
void ctx_destroy(GraphContext* ctx);

Tensor* tensor_create(GraphContext* ctx, int* shape, int rank, bool requires_grad);
Tensor* tensor_create_zeros(GraphContext* ctx, int* shape, int rank, bool requires_grad);
Tensor* tensor_clone(GraphContext* ctx, Tensor* t);
void tensor_zeros(Tensor* t);
void tensor_ones(Tensor* t);
void tensor_rand(Tensor* t, float low, float high);
bool tensor_shape_equal(const Tensor* a, const Tensor* b);
Tensor* tensor_add(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* tensor_add_scalar(GraphContext* ctx, Tensor* a, float scalar);
Tensor* tensor_mul_elementwise(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* tensor_matmul_2d(GraphContext* ctx, Tensor* a, Tensor* b);
Tensor* tensor_transpose(GraphContext* ctx, Tensor* t, int dim0, int dim1);
int tensor_argmax(const Tensor* t);
void tensor_print(const Tensor* t);

void tensor_backward(GraphContext* ctx, Tensor* loss);
void tensor_zero_grad(Tensor* t);

#endif