#include "tensor.h"
#include "arena.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

GraphContext* ctx_create(size_t initial_memory) {
  GraphContext* ctx = (GraphContext*)malloc(sizeof(GraphContext));
  ctx->arena = (Arena*)malloc(sizeof(Arena));
  arena_init(ctx->arena, initial_memory);
  
  ctx->tape_count = 0;
  ctx->tape_capacity = 1024;
  ctx->gradient_tape = (Tensor**)malloc(ctx->tape_capacity * sizeof(Tensor*)); // list of tensors
  
  return ctx;
}

void ctx_reset(GraphContext* ctx) {
  arena_reset(ctx->arena);
  ctx->tape_count = 0;
}
void ctx_destroy(GraphContext* ctx) {
  arena_destroy(ctx->arena);
  free(ctx->arena);
  free(ctx->gradient_tape);
  free(ctx);
  
}

Tensor* tensor_create(GraphContext* ctx, int* shape, int rank, bool requires_grad) {
  Tensor* t = (Tensor*)arena_alloc(ctx->arena, sizeof(Tensor), 16);
  
  t->rank = rank;
  t->shape = (int*)arena_alloc(ctx->arena, rank * sizeof(int), 4);
  t->strides = (int*)arena_alloc(ctx->arena, rank * sizeof(int), 4);
  
  memcpy(t->shape, shape, sizeof(int) * rank);
  
  int total_elements = 1;
  int current_stride = 1;
  
  for (int i = rank - 1; i >= 0; i--) {
    t->strides[i] = current_stride;
    current_stride *= shape[i];
    total_elements *= shape[i];
  }
  
  t->total_elements = total_elements;
  
  t->values = (float*)arena_alloc(ctx->arena, total_elements * sizeof(*t->values), 32);
  
  t->requires_autograd = requires_grad;
  t->backward_fn = NULL;
  t->parents = NULL;
  t->num_parents = 0;
  
  if (requires_grad) {
    t->grad = (float*)arena_calloc(ctx->arena, total_elements * sizeof(float), 32);
  } else {
    t->grad = NULL;
  }
  
  return t;
}

Tensor* tensor_create_zeros(GraphContext* ctx, int* shape, int rank, bool requires_grad) {
  Tensor* t = (Tensor*)arena_alloc(ctx->arena, sizeof(Tensor), 16);
  
  t->rank = rank;
  t->shape = (int*)arena_alloc(ctx->arena, rank * sizeof(int), 4);
  t->strides = (int*)arena_alloc(ctx->arena, rank * sizeof(int), 4);
  
  memcpy(t->shape, shape, sizeof(int) * rank);
  
  int total_elements = 1;
  int current_stride = 1;
  
  for (int i = rank - 1; i >= 0; i--) {
    t->strides[i] = current_stride;
    current_stride *= shape[i];
    total_elements *= shape[i];
  }
  
  t->total_elements = total_elements;
  
  t->values = (float*)arena_calloc(ctx->arena, total_elements * sizeof(*t->values), 32);
  
  t->requires_autograd = requires_grad;
  t->backward_fn = NULL;
  t->parents = NULL;
  t->num_parents = 0;
  
  if (requires_grad) {
    t->grad = (float*)arena_calloc(ctx->arena, total_elements * sizeof(float), 32);
  } else {
    t->grad = NULL;
  }
  
  return t;
}

void backward_clone(Tensor* out) {
  Tensor* parent = out->parents[0];
  if (!parent->requires_autograd) return;
  
  for (size_t i = 0; i < out->total_elements; i++) {
    parent->grad[i] += out->grad[i];
  }
}

Tensor* tensor_clone(GraphContext* ctx, Tensor* t) {
  if (!t) return NULL;
  
  Tensor* clone = tensor_create(ctx, t->shape, t->rank, t->requires_autograd);
  
  memcpy(clone->values, t->values, t->total_elements * sizeof(float));
  
  if (t->requires_autograd) {
    clone->num_parents = 1;
    clone->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    clone->parents[0] = t;
    clone->backward_fn = backward_clone;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = clone;
  }
  
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

bool tensor_shape_equal(const Tensor* a, const Tensor* b) {
  if (a->rank != b->rank) return false;
  return memcmp(a->shape, b->shape, a->rank * sizeof(int)) == 0;
}

void backward_add(Tensor* out) {
  Tensor* a = out->parents[0];
  Tensor* b = out->parents[1];
  
  for (size_t i = 0; i < out->total_elements; i++) {
    if (a->requires_autograd) a->grad[i] += out->grad[i];
    if (b->requires_autograd) b->grad[i] += out->grad[i];
  }
}

Tensor* tensor_add(GraphContext* ctx, Tensor* a, Tensor* b) {
  assert(tensor_shape_equal(a, b) && "Operand shapes must match");
  
  bool req_grad = a->requires_autograd || b->requires_autograd;
  Tensor* out = tensor_create(ctx, a->shape, a->rank, req_grad);
  
  for (size_t i = 0; i < out->total_elements; i++) {
    out->values[i] = a->values[i] + b->values[i];
  }
  
  if (req_grad) {
    out->num_parents = 2;
    out->parents = (Tensor**)arena_alloc(ctx->arena, 2 * sizeof(Tensor*), 8);
    out->parents[0] = a;
    out->parents[1] = b;
    out->backward_fn = backward_add;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
    
  }
  
  return out;
}

void backward_scalar_add(Tensor* out) {
  Tensor* a = out->parents[0];
  if (!a->requires_autograd) return;
  for (size_t i = 0; i < out->total_elements; i++) {
    a->grad[i] += out->grad[i];
  }
}

Tensor* tensor_add_scalar(GraphContext* ctx, Tensor* a, float scalar) { 
  bool req_grad = a->requires_autograd;
  Tensor* out = tensor_create(ctx, a->shape, a->rank, req_grad);
  
  for (size_t i = 0; i < out->total_elements; i++) {
    out->values[i] = a->values[i] + scalar;
  }
  
  if (req_grad) {
    out->num_parents = 1;
    out->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    out->parents[0] = a;
    out->backward_fn = backward_scalar_add;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
  }
  
  return out;
}

void backward_mul_elementwise(Tensor* out) {
  Tensor* a = out->parents[0];
  Tensor* b = out->parents[1];
  
  for (size_t i = 0; i < out->total_elements; i++) {
    if (a->requires_autograd) a->grad[i] += b->values[i] * out->grad[i];
    if (b->requires_autograd) b->grad[i] += a->values[i] * out->grad[i];
  }
}

Tensor* tensor_mul_elementwise(GraphContext* ctx, Tensor* a, Tensor* b) {
  assert(tensor_shape_equal(a, b) && "Operand shapes must match");
  
  bool req_grad = a->requires_autograd || b->requires_autograd;
  Tensor* out = tensor_create(ctx, a->shape, a->rank, req_grad);
  
  for (size_t i = 0; i < out->total_elements; i++) {
    out->values[i] = a->values[i] * b->values[i];
  }
  
  if (req_grad) {
    out->num_parents = 2;
    out->parents = (Tensor**)arena_alloc(ctx->arena, 2 * sizeof(Tensor*), 8);
    out->parents[0] = a;
    out->parents[1] = b;
    out->backward_fn = backward_mul_elementwise;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
    
  }
  
  return out;
  
}

void backward_matmul_2d(Tensor* out) {
  Tensor* a = out->parents[0];
  Tensor* b = out->parents[1];

  int M = a->shape[0];
  int K = a->shape[1];
  int N = b->shape[1];
  
  if (a->requires_autograd) {
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
          size_t idx_out = (size_t)i * out->strides[0] + (size_t)j * out->strides[1];
          size_t idx_b = (size_t)k * b->strides[0] + (size_t)j * b->strides[1]; 
          
          sum += out->grad[idx_out] * b->values[idx_b];
        }
        size_t idx_a = (size_t)i * a->strides[0] + (size_t)k * a->strides[1];
        a->grad[idx_a] += sum;
      }
    }
  }
  
  if (b->requires_autograd) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
          size_t idx_a = (size_t)i * a->strides[0] + (size_t)k * a->strides[1];
          size_t idx_out = (size_t)i * out->strides[0] + (size_t)j * out->strides[1];
          
          sum += a->values[idx_a] * out->grad[idx_out];
        }
        size_t idx_b = (size_t)k * b->strides[0] + (size_t)j * b->strides[1];
        b->grad[idx_b] += sum;
      }
    }
  }
}

Tensor* tensor_matmul_2d(GraphContext* ctx, Tensor* a, Tensor* b) {
  assert(a->rank == 2 && b->rank == 2 && "Tensors must be 2D");
  assert(a->shape[1] == b->shape[0] && "Inner dimensions must match (A cols == B rows)");
  
  bool req_grad = a->requires_autograd || b->requires_autograd;
  int shape[] = {a->shape[0], b->shape[1]};
  Tensor* out = tensor_create_zeros(ctx, shape, a->rank, req_grad);

  int M = a->shape[0]; // Rows of A
  int K = a->shape[1]; // Cols of A / Rows of B
  int N = b->shape[1]; // Cols of B
  
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        size_t idx_a = (size_t)i * a->strides[0] + (size_t)k * a->strides[1];
        size_t idx_b = (size_t)k * b->strides[0] + (size_t)j * b->strides[1];
        
        sum += a->values[idx_a] * b->values[idx_b];
      }
      size_t idx_dest = (size_t)i * out->strides[0] + (size_t)j * out->strides[1];
      out->values[idx_dest] = sum;
    }
  }
  
  if (req_grad) {
    out->num_parents = 2;
    out->parents = (Tensor**)arena_alloc(ctx->arena, 2 * sizeof(Tensor*), 8);
    out->parents[0] = a;
    out->parents[1] = b;
    out->backward_fn = backward_matmul_2d;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
    
  }
  
  return out;
}

Tensor* tensor_transpose(GraphContext* ctx, Tensor* t, int dim0, int dim1) {
  assert(t != NULL && "Tensor cannot be NULL");
  assert(dim0 >= 0 && dim0 < t->rank && "dim0 is out of bounds");
  assert(dim1 >= 0 && dim1 < t->rank && "dim1 is out of bounds");

  if (dim0 == dim1) return t;
  
  Tensor* out = (Tensor*)arena_alloc(ctx->arena, sizeof(Tensor), 16);
  
  out->rank = t->rank;
  out->total_elements = t->total_elements;
  out->requires_autograd = t->requires_autograd;
  
  out->values = t->values;
  out->grad = t->grad;
  
  out->shape = (int*)arena_alloc(ctx->arena, t->rank * sizeof(int), 4);
  out->strides = (int*)arena_alloc(ctx->arena, t->rank * sizeof(int), 4);
  memcpy(out->shape, t->shape, t->rank * sizeof(int));
  memcpy(out->strides, t->strides, t->rank * sizeof(int));

  out->shape[dim0] = t->shape[dim1];
  out->shape[dim1] = t->shape[dim0];
  out->strides[dim0] = t->strides[dim1];
  out->strides[dim1] = t->strides[dim0];
  
  if (t->requires_autograd) {
    out->num_parents = 1;
    out->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    out->parents[0] = t;
    out->backward_fn = NULL;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
  }
  
  return out;
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

void tensor_print(const Tensor* t) {
  if (!t) return;
    print_recursive(t, 0, 0);
    printf("\n");
  }

void tensor_backward(GraphContext* ctx, Tensor* loss) {
  assert(loss->requires_autograd && "Loss tensor must require gradients");
  
  for (size_t i = 0; i < loss->total_elements; i++) {
    loss->grad[i] = 1.0f;
  }
  
  for (int i = ctx->tape_count - 1; i >= 0; i--) {
    Tensor* t = ctx->gradient_tape[i];
    
    if (t->backward_fn != NULL) {
      t->backward_fn(t);
    }
  }
}