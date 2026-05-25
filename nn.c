#include "nn.h"
#include "arena.h"
#include "tensor.h"
#include <math.h>


LinearLayer* linear_layer_create(GraphContext* ctx, int input_dim, int output_dim) {
  LinearLayer* layer= (LinearLayer*)arena_alloc(ctx->arena, sizeof(LinearLayer), 16);
  
  int weight_shape[] = {output_dim, input_dim};
  int bias_shape[] = {output_dim, 1};
  
  bool req_grad = true;
  
  layer->weights = tensor_create(ctx, weight_shape, 2, req_grad);
  layer->bias = tensor_create_zeros(ctx, bias_shape, 2, req_grad);
  
  layer->m_weight = tensor_create_zeros(ctx, weight_shape, 2, req_grad);
  layer->v_weight = tensor_create_zeros(ctx, weight_shape, 2, req_grad);
  
  layer->m_bias = tensor_create_zeros(ctx, bias_shape, 2, req_grad);
  layer->v_bias = tensor_create_zeros(ctx, bias_shape, 2, req_grad);
  
  float scale = sqrtf(2.0f / (float)input_dim);
  tensor_rand(layer->weights, -scale, scale);
  
  return layer;
}

Tensor* nn_linear_forward(GraphContext* ctx, LinearLayer* layer, Tensor* input) {
  Tensor* matmul_out = tensor_matmul_2d(ctx, layer->weights, input); 
  Tensor* final_out = tensor_add(ctx, matmul_out, layer->bias);
  
  return final_out;
}

void backward_relu(Tensor* out) {
  Tensor* t = out->parents[0];
  if (!t->requires_autograd) return;
  
  for (size_t i = 0; i < out->total_elements; i++) {
    if (t->values[i] <= 0) {
      t->grad[i] += 0;
    } else {
      t->grad[i] += out->grad[i];
    }
  }
}

Tensor* nn_relu_forward(GraphContext* ctx, Tensor* t) {
  bool req_grad = t->requires_autograd;
  Tensor* out = tensor_create(ctx, t->shape, t->rank, req_grad);
  
  for(size_t i = 0; i < t->total_elements; i++) {
    if (t->values[i] <= 0) {
      out->values[i] = 0;
    } else {
      out->values[i] = t->values[i];
    }
  }
  
  if (req_grad) {
    out->num_parents = 1;
    out->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    out->parents[0] = t;
    out->backward_fn = backward_relu;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
  }
  
  return out;
}

void backward_gelu(Tensor* out) {
  Tensor* t = out->parents[0];
  if (!t->requires_autograd) return;
  
  const float SQRT_2_OVER_PI = 0.7978845608f;
  const float COEF = 0.044715f;
  const float COEF_DERIV = 0.134145f;
  
  for(size_t i = 0; i < out->total_elements; i++) {
    float x = t->values[i];
    float x2 = x * x;
    float x3 = x2 * x;
    float u = x + COEF * x3;
    float tanh_u = tanhf(SQRT_2_OVER_PI * u);
    float sech2 = 1.0f - (tanh_u * tanh_u);
    float du = 1.0f + COEF_DERIV * x2;

    float local_grad = 0.5f * (1.0f + tanh_u) + (0.5f * x * sech2 * SQRT_2_OVER_PI * du);

    t->grad[i] += local_grad * out->grad[i];
  }
}

Tensor* nn_gelu_forward(GraphContext* ctx, Tensor* t) {
  bool req_grad = t->requires_autograd;
  Tensor* out = tensor_create(ctx, t->shape, t->rank, req_grad);
  
  const float SQRT_2_OVER_PI = 0.7978845608f;
  
  for(size_t i = 0; i < t->total_elements; i++) {
    float x = t->values[i];
    out->values[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI*(x + 0.044715f * x * x * x)));
  }
  
  if (req_grad) {
    out->num_parents = 1;
    out->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    out->parents[0] = t;
    out->backward_fn = backward_gelu;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    
    ctx->gradient_tape[ctx->tape_count++] = out;
  }
  
  return out;
}

void backward_softmax(Tensor* out) {
  Tensor* input = out->parents[0];
  if (!input->requires_autograd) return;

  float sum_ds_s = 0.0f;
  for (size_t i = 0; i < out->total_elements; i++) {
    sum_ds_s += out->grad[i] * out->values[i];
  }

  for (size_t i = 0; i < out->total_elements; i++) {
    input->grad[i] += out->values[i] * (out->grad[i] - sum_ds_s);
  }
}

Tensor* nn_softmax_forward(GraphContext* ctx, Tensor* t) {
  bool req_grad = t->requires_autograd;
  Tensor* out = tensor_create(ctx, t->shape, t->rank, req_grad);

  float max = t->values[0];
  for (size_t i = 1; i < t->total_elements; i++) {
    if (t->values[i] > max) {
      max = t->values[i];
    }
  }

  float sum = 0.0f;
  for (size_t i = 0; i < t->total_elements; i++) {
    out->values[i] = expf(t->values[i] - max);
    sum += out->values[i];
  }

  for (size_t i = 0; i < t->total_elements; i++) {
    out->values[i] /= sum;
  }

  if (req_grad) {
    out->num_parents = 1;
    out->parents = (Tensor**)arena_alloc(ctx->arena, sizeof(Tensor*), 8);
    out->parents[0] = t;
    out->backward_fn = backward_softmax;

    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    ctx->gradient_tape[ctx->tape_count++] = out;
  }

  return out;
}

void backward_layer_norm(Tensor* out) {
  Tensor* input = out->parents[0];
  Tensor* gamma = out->parents[1];
  Tensor* beta = out->parents[2];
  
  size_t N = input->total_elements;
  float inv_N = 1.0f / (float)N;

  float sum = 0.0f;
  for (size_t i = 0; i < N; i++) {
    sum += input->values[i];
  }
  float mean = sum * inv_N;

  float var = 0.0f;
  for (size_t i = 0; i < N; i++) {
    float diff = input->values[i] - mean;
    var += diff * diff;
  }
  var *= inv_N;
  float inv_stddev = 1.0f / sqrtf(var + 1e-5f); 

  float sum_dx_hat = 0.0f;
  float sum_dx_hat_x_hat = 0.0f;

  for (size_t i = 0; i < N; i++) {
    float dy = out->grad[i];
    float x_hat = (input->values[i] - mean) * inv_stddev;

    if (gamma->requires_autograd) gamma->grad[i] += dy * x_hat;
    if (beta->requires_autograd) beta->grad[i] += dy;

    float dx_hat = dy * gamma->values[i];
    sum_dx_hat += dx_hat;
    sum_dx_hat_x_hat += dx_hat * x_hat;
  }

  if (input->requires_autograd) {
    for (size_t i = 0; i < N; i++) {
      float dx_hat = out->grad[i] * gamma->values[i];
      float x_hat = (input->values[i] - mean) * inv_stddev;

      input->grad[i] += inv_stddev * (dx_hat - (sum_dx_hat * inv_N) - (x_hat * sum_dx_hat_x_hat * inv_N));
    }
  }
}

Tensor* nn_layer_norm_forward(GraphContext* ctx, LayerNormLayer* layer, Tensor* input) {
  bool req_grad = input->requires_autograd || layer->gamma->requires_autograd || layer->beta->requires_autograd;
  Tensor* out = tensor_create(ctx, input->shape, input->rank, req_grad);

  size_t N = input->total_elements;
  float sum = 0.0f;
  for (size_t i = 0; i < N; i++) {
    sum += input->values[i];
  }
  float mean = sum / (float)N;
  
  float var = 0.0f;
  for (size_t i = 0; i < N; i++) {
    float diff = (input->values[i] - mean);
    var += diff * diff;
  }
  var /= (float)N;
  float inv_stddev = 1.0f / sqrtf(var + layer->epsilon);
  
  for (size_t i = 0; i < N; i++) {
    float norm_val = (input->values[i] - mean) * inv_stddev;
    out->values[i] = (norm_val * layer->gamma->values[i]) + layer->beta->values[i];
  }

  if (req_grad) {
    out->num_parents = 3;
    out->parents = (Tensor**)arena_alloc(ctx->arena, 3 * sizeof(Tensor*), 8);
    out->parents[0] = input;
    out->parents[1] = layer->gamma;
    out->parents[2] = layer->beta;
    out->backward_fn = backward_layer_norm;
    
    if (ctx->tape_count >= ctx->tape_capacity) {
      ctx->tape_capacity *= 2;
      ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
    }
    ctx->gradient_tape[ctx->tape_count++] = out;
  }
  
  return out;
}

void backward_mse(Tensor* out) {
  Tensor* preds = out->parents[0];
  Tensor* targets = out->parents[1];
  size_t n = preds->total_elements;
  
  if (!preds->requires_autograd) return;

  for(size_t i = 0; i < n; i++) {
    preds->grad[i] += out->grad[0] * (2.0f / (float)n) * (preds->values[i] - targets->values[i]);
  }
}

Tensor* nn_mse_loss(GraphContext* ctx, Tensor* predictions, Tensor* targets) { 
  Tensor* loss = tensor_create(ctx, NULL, 0, true);
  loss->values[0] = 0.0f;
  
  for(size_t i = 0; i < predictions->total_elements; i++) {
    float diff = (targets->values[i] - predictions->values[i]);
    loss->values[0] += diff * diff;
  }
  loss->values[0] /= (float)predictions->total_elements;
  
  loss->num_parents = 2;
  loss->parents = (Tensor**)arena_alloc(ctx->arena, 2 * sizeof(Tensor*), 8);
  loss->parents[0] = predictions;
  loss->parents[1] = targets;
  loss->backward_fn = backward_mse;
  
  if (ctx->tape_count >= ctx->tape_capacity) {
    ctx->tape_capacity *= 2;
    ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
  }
  ctx->gradient_tape[ctx->tape_count++] = loss;
  
  return loss;
}

void backward_cross_entropy(Tensor* out) {
  Tensor* preds = out->parents[0];
  Tensor* targets = out->parents[1];
  float epsilon = 1e-7f;
  
  if (!preds->requires_autograd) return;
  
  for (size_t i = 0; i < targets->total_elements; i++) {
    preds->grad[i] += -out->grad[0] * (targets->values[i] / (preds->values[i] + epsilon));
  }
}

Tensor* nn_cross_entropy_loss(GraphContext* ctx, Tensor* predictions, Tensor* targets) {
  Tensor* loss = tensor_create(ctx, NULL, 0, true);
  loss->values[0] = 0.0f;
  float epsilon = 1e-7f;
  
  for(size_t i = 0; i < targets->total_elements; i++) {
    if (targets->values[i] > 0.0f) {
      loss->values[0] -= targets->values[i] * logf(predictions->values[i] + epsilon);
    }
  }
  
  loss->num_parents = 2;
  loss->parents = (Tensor**)arena_alloc(ctx->arena, 2 * sizeof(Tensor*), 8);
  loss->parents[0] = predictions;
  loss->parents[1] = targets;
  loss->backward_fn = backward_cross_entropy;
  
  if (ctx->tape_count >= ctx->tape_capacity) {
    ctx->tape_capacity *= 2;
    ctx->gradient_tape = (Tensor**)realloc(ctx->gradient_tape, ctx->tape_capacity * sizeof(Tensor*));
  }
  ctx->gradient_tape[ctx->tape_count++] = loss;
  
  return loss;
}

void nn_sgd_update(LinearLayer* layer, float learning_rate) {
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    layer->weights->values[i] -= learning_rate * layer->weights->grad[i];
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    layer->bias->values[i] -= learning_rate * layer->bias->grad[i];
  }
  
  tensor_zero_grad(layer->weights);
  tensor_zero_grad(layer->bias);
}

void nn_rmsprop_update(LinearLayer* layer, float lr, float rho, float epsilon, float weight_decay) {
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    float g = layer->weights->grad[i];
    float w = layer->weights->values[i];
    
    w = w - (lr * weight_decay * w);
    
    layer->v_weight->values[i] = rho * layer->v_weight->values[i] + (1.0f - rho) * (g * g);
    
    layer->weights->values[i] = w - lr / (sqrtf(layer->v_weight->values[i]) + epsilon) * g;
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    float g = layer->bias->grad[i];
    
    layer->v_bias->values[i] = rho * layer->v_bias->values[i] + (1.0f - rho) * (g * g);
    
    layer->bias->values[i] = layer->bias->values[i] - lr / (sqrtf(layer->v_bias->values[i]) + epsilon) * g;
  }
  
  tensor_zero_grad(layer->weights);
  tensor_zero_grad(layer->bias);
}

void nn_adamw_update(LinearLayer* layer, float lr, int t, float beta1, float beta2, float epsilon, float weight_decay) {
  float b1_t = 1.0f - powf(beta1, t);
  float b2_t = 1.0f - powf(beta2, t);
  
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    float g = layer->weights->grad[i];
    float w = layer->weights->values[i];
    
    w = w - (lr * weight_decay * w);
    
    layer->m_weight->values[i] = beta1 * layer->m_weight->values[i] + (1.0f - beta1) * g;
    layer->v_weight->values[i] = beta2 * layer->v_weight->values[i] + (1.0f - beta2) * (g * g);
    
    float m_hat = layer->m_weight->values[i] / b1_t;
    float v_hat = layer->v_weight->values[i] / b2_t;
    
    layer->weights->values[i] = w - lr * (m_hat / (sqrtf(v_hat) + epsilon));
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    float g = layer->bias->grad[i];
    
    layer->m_bias->values[i] = beta1 * layer->m_bias->values[i] + (1.0f - beta1) * g;
    layer->v_bias->values[i] = beta2 * layer->v_bias->values[i] + (1.0f - beta2) * (g * g);
    
    float m_hat = layer->m_bias->values[i] / b1_t;
    float v_hat = layer->v_bias->values[i] / b2_t;
    
    layer->bias->values[i] -= lr * (m_hat / (sqrtf(v_hat) + epsilon));
  }
  
  tensor_zero_grad(layer->weights);
  tensor_zero_grad(layer->bias);
}
