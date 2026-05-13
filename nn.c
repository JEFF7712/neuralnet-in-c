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
  tensor_relu(t);
}

void nn_relu_backward(Tensor* grad_output, const Tensor* input_cache) {
  for (size_t i = 0; i < grad_output->total_elements; i++) {
    if (input_cache->values[i] <= 0) {
      grad_output->values[i] = 0;
    }
  }
}

float nn_mse_loss(const Tensor* predictions, const Tensor* targets){ 
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