#ifndef NN_H
#define NN_H

#include "tensor.h"

typedef struct {
  Tensor* weights;
  Tensor* bias;
  
  //AdamW
  Tensor* m_weight;
  Tensor* v_weight;
  Tensor* m_bias;
  Tensor* v_bias;
  
} LinearLayer;

typedef struct {
  Tensor* gamma;
  Tensor* beta;
  
  // Optimizer states
  Tensor* m_gamma;
  Tensor* v_gamma;
  Tensor* m_beta;
  Tensor* v_beta;
  
  float epsilon;
} LayerNormLayer;

LinearLayer* linear_layer_create(GraphContext* ctx, int input_dim, int output_dim);
void nn_linear_free(LinearLayer* layer);
Tensor* nn_linear_forward(GraphContext* ctx, LinearLayer* layer, Tensor* input);

Tensor* nn_relu_forward(GraphContext* ctx, Tensor* t);
Tensor* nn_gelu_forward(GraphContext* ctx, Tensor* t);
Tensor* nn_softmax_forward(GraphContext* ctx, Tensor* t);
void nn_softmax_backward(Tensor* grad_output, Tensor* input_cache);
Tensor* nn_layer_norm_forward(GraphContext* ctx, LayerNormLayer* layer, Tensor* input);
Tensor* nn_mse_loss(GraphContext* ctx, Tensor* predictions, Tensor* targets);
Tensor* nn_cross_entropy_loss(GraphContext* ctx, Tensor* predictions, Tensor* targets);

void nn_sgd_update(LinearLayer* layer, float learning_rate);
void nn_rmsprop_update(LinearLayer* layer, float lr, float rho, float epsilon, float weight_decay);
void nn_adamw_update(LinearLayer* layer, float lr, int t, float beta1, float beta2, float epsilon, float weight_decay);

#endif