#include "nn.h"
#include "tensor.h"
#include <math.h>


LinearLayer* linear_layer_create(int input_dim, int output_dim) {
  LinearLayer* layer= (LinearLayer*)malloc(sizeof(LinearLayer));
  
  int weight_shape[] = {output_dim, input_dim};
  int bias_shape[] = {output_dim, 1};
  
  layer->weights = tensor_create(weight_shape, 2);
  layer->bias = tensor_create_zeros(bias_shape, 2);
  
  layer->grad_weight = tensor_create_zeros(weight_shape, 2);
  layer->grad_bias = tensor_create_zeros(bias_shape, 2);
  
  layer->m_weight = tensor_create_zeros(weight_shape, 2);
  layer->v_weight = tensor_create_zeros(weight_shape, 2);
  
  layer->m_bias = tensor_create_zeros(bias_shape, 2);
  layer->v_bias = tensor_create_zeros(bias_shape, 2);
  
  float scale = sqrtf(2.0f / (float)input_dim);
  tensor_rand(layer->weights, -scale, scale);
  
  layer->input_cache = NULL;
  
  return layer;
}

void nn_linear_free(LinearLayer* layer) {
  if (!layer) return;
  tensor_free(layer->weights);
  tensor_free(layer->bias);
  tensor_free(layer->grad_weight);
  tensor_free(layer->grad_bias);
  tensor_free(layer->m_weight);
  tensor_free(layer->v_weight);
  tensor_free(layer->m_bias);
  tensor_free(layer->v_bias);
  tensor_free(layer->input_cache);
  free(layer);
}

void nn_linear_forward(LinearLayer* layer, const Tensor* input, Tensor* output) {
  if (layer->input_cache) {
    tensor_free(layer->input_cache);
  }
  layer->input_cache = tensor_clone(input);
  
  tensor_matmul_2d(output, layer->weights, input);
  tensor_add(output, output, layer->bias);
}

void nn_linear_backward(LinearLayer* layer, const Tensor* grad_output, Tensor* grad_input) {
  for (size_t i = 0; i < layer->grad_bias->total_elements; i++) {
    layer->grad_bias->values[i] += grad_output->values[i];
  }
  
  Tensor* input_T = tensor_clone(layer->input_cache);
  tensor_transpose(input_T, 0, 1);
  tensor_matmul_2d(layer->grad_weight, grad_output, input_T);
  
  Tensor* weight_T = tensor_clone(layer->weights);
  tensor_transpose(weight_T, 0, 1);
  tensor_zeros(grad_input);
  tensor_matmul_2d(grad_input, weight_T, grad_output);
  
  tensor_free(input_T);
  tensor_free(weight_T);
}

void nn_relu_forward(Tensor* t) {
  for(size_t i = 0; i < t->total_elements; i++) {
    if (t->values[i] <= 0) {
      t->values[i] = 0;
    }
  }
}

void nn_relu_backward(Tensor* grad_output, const Tensor* input_cache) {
  for (size_t i = 0; i < grad_output->total_elements; i++) {
    if (input_cache->values[i] <= 0) {
      grad_output->values[i] = 0;
    }
  }
}

void nn_gelu_forward(Tensor* t) {
  const float SQRT_2_OVER_PI = 0.7978845608f;
  
  for(size_t i = 0; i < t->total_elements; i++) {
    float x = t->values[i];
    t->values[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI*(x + 0.044715f * x * x * x)));
  }
}

void nn_gelu_backward(Tensor* grad_output, const Tensor* input_cache) {
  const float SQRT_2_OVER_PI = 0.7978845608f;
  const float COEF = 0.044715f;
  const float COEF_DERIV = 0.134145f;
  
  for(size_t i = 0; i < grad_output->total_elements; i++) {
    float x = input_cache->values[i];
    float x2 = x * x;
    float x3 = x2 * x;
    float u = x + COEF * x3;
    float tanh_u = tanhf(SQRT_2_OVER_PI * u);
    float sech2 = 1.0f - (tanh_u * tanh_u);
    float du = 1.0f + COEF_DERIV * x2;
    
    float local_grad = 0.5f * (1.0f + tanh_u) + (0.5f * x * sech2 * SQRT_2_OVER_PI * du);
        
    grad_output->values[i] *= local_grad;
  }
}

void nn_softmax_forward(Tensor* t) {
  float max = t->values[0];
  
  for (size_t i = 0; i < t->total_elements; i++) {
    if (t->values[i] > max) {
      max = t->values[i];
    }
  }
  
  float sum = 0.0f;
  
  for (size_t i = 0; i < t->total_elements; i++) {
    t->values[i] = expf(t->values[i] - max);
    sum += t->values[i];
  }
  
  for (size_t i = 0; i < t->total_elements; i++) {
    t->values[i] /= sum;
  }
  
}

float nn_mse_loss(const Tensor* predictions, const Tensor* targets) { 
  float sum = 0.0f;
  for(size_t i = 0; i < predictions->total_elements; i++) {
    float diff = (targets->values[i] - predictions->values[i]);
    sum += diff * diff;
  }
  return sum / (float)predictions->total_elements;
}

void nn_mse_gradient(const Tensor* predictions, const Tensor* targets, Tensor* grad_output) {
  size_t n = predictions->total_elements;
  for(size_t i = 0; i < n; i++) {
     grad_output->values[i] = (-2.0f / (float)n) * (targets->values[i] - predictions->values[i]);
  }
}

float nn_cross_entropy_loss(const Tensor* predictions, const Tensor* targets) {
  float loss = 0.0f;
  float epsilon = 1e-7f;
  
  for(size_t i = 0; i < targets->total_elements; i++) {
    if (targets->values[i] > 0.0f) {
      loss -= targets->values[i] * logf(predictions->values[i] + epsilon);
    }
  }
  
  return loss;
}

void nn_cross_entropy_gradient(const Tensor* predictions, const Tensor* targets, Tensor* grad_output) {
  for (size_t i = 0; i < targets->total_elements; i++) {
    grad_output->values[i] = predictions->values[i] - targets->values[i];
  }
}

void nn_sgd_update(LinearLayer* layer, float learning_rate) {
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    layer->weights->values[i] -= learning_rate * layer->grad_weight->values[i];
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    layer->bias->values[i] -= learning_rate * layer->grad_bias->values[i];
  }
  
  tensor_zeros(layer->grad_weight);
  tensor_zeros(layer->grad_bias);
}

void nn_rmsprop_update(LinearLayer* layer, float lr, float rho, float epsilon, float weight_decay) {
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    float g = layer->grad_weight->values[i];
    float w = layer->weights->values[i];
    
    w = w - (lr * weight_decay * w);
    
    layer->v_weight->values[i] = rho * layer->v_weight->values[i] + (1.0f - rho) * (g * g);
    
    layer->weights->values[i] = w - lr / (sqrtf(layer->v_weight->values[i]) + epsilon) * g;
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    float g = layer->grad_bias->values[i];
    
    layer->v_bias->values[i] = rho * layer->v_bias->values[i] + (1.0f - rho) * (g * g);
    
    layer->bias->values[i] = layer->bias->values[i] - lr / (sqrtf(layer->v_bias->values[i]) + epsilon) * g;
  }
  
  tensor_zeros(layer->grad_weight);
  tensor_zeros(layer->grad_bias);
}

void nn_adamw_update(LinearLayer* layer, float lr, int t, float beta1, float beta2, float epsilon, float weight_decay) {
  float b1_t = 1.0f - powf(beta1, t);
  float b2_t = 1.0f - powf(beta2, t);
  
  for (size_t i = 0; i < layer->weights->total_elements; i++) {
    float g = layer->grad_weight->values[i];
    float w = layer->weights->values[i];
    
    w = w - (lr * weight_decay * w);
    
    layer->m_weight->values[i] = beta1 * layer->m_weight->values[i] + (1.0f - beta1) * g;
    layer->v_weight->values[i] = beta2 * layer->v_weight->values[i] + (1.0f - beta2) * (g * g);
    
    float m_hat = layer->m_weight->values[i] / b1_t;
    float v_hat = layer->v_weight->values[i] / b2_t;
    
    layer->weights->values[i] = w - lr * (m_hat / (sqrtf(v_hat) + epsilon));
  }
  
  for (size_t i = 0; i < layer->bias->total_elements; i++) {
    float g = layer->grad_bias->values[i];
    
    layer->m_bias->values[i] = beta1 * layer->m_bias->values[i] + (1.0f - beta1) * g;
    layer->v_bias->values[i] = beta2 * layer->v_bias->values[i] + (1.0f - beta2) * (g * g);
    
    float m_hat = layer->m_bias->values[i] / b1_t;
    float v_hat = layer->v_bias->values[i] / b2_t;
    
    layer->bias->values[i] -= lr * (m_hat / (sqrtf(v_hat) + epsilon));
  }
  
  tensor_zeros(layer->grad_weight);
  tensor_zeros(layer->grad_bias);
}
