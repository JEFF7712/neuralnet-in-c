#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct {
  Tensor* weights;
  Tensor* bias;
  
  Tensor* input_cache;
    
  Tensor* grad_weight; 
  Tensor* grad_bias;
  
  //adamw
  Tensor* m_weight;
  Tensor* v_weight;
  
  Tensor* m_bias;
  Tensor* v_bias;
  
} LinearLayer;

LinearLayer* linear_layer_create(int input_dim, int output_dim);
void nn_linear_free(LinearLayer* layer);
void nn_linear_forward(LinearLayer* layer, const Tensor* input, Tensor* output);
void nn_linear_backward(LinearLayer* layer, const Tensor* grad_output, Tensor* grad_input);

void nn_relu_forward(Tensor* t);
void nn_relu_backward(Tensor* grad_output, const Tensor* input_cache);
void nn_gelu_forward(Tensor* t);
void nn_gelu_backward(Tensor* grad_output, const Tensor* input_cache);

void nn_softmax_forward(Tensor* t);
void nn_softmax_backward(Tensor* grad_output, Tensor* input_cache);

float nn_mse_loss(const Tensor* predictions, const Tensor* targets);
void nn_mse_gradient(const Tensor* predictions, const Tensor* targets, Tensor* grad_output);
float nn_cross_entropy_loss(const Tensor* predictions, const Tensor* targets);
void nn_cross_entropy_gradient(const Tensor* predictions, const Tensor* targets, Tensor* grad_output);

void nn_sgd_update(LinearLayer* layer, float learning_rate);
void nn_rmsprop_update(LinearLayer* layer, float lr, float rho, float epsilon, float weight_decay);
void nn_adamw_update(LinearLayer* layer, float lr, int t, float beta1, float beta2, float epsilon, float weight_decay);

#endif