#include "tensor.h"
#include "nn.h"
#include <stdio.h>
#include <time.h>

int main(void) {
  srand(time(NULL));
  
  float x_data[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
  float y_data[4][1] = {{0}, {1}, {1}, {0}};
      
  LinearLayer* l1 = linear_layer_create(2, 4);
  LinearLayer* l2 = linear_layer_create(4, 1);
  
  int h_shape[] = {4, 1};
  int o_shape[] = {1, 1};
  
  Tensor* hidden = tensor_create(h_shape, 2);
  Tensor* output = tensor_create(o_shape, 2);
  Tensor* target = tensor_create(o_shape, 2);
  
  Tensor* grad_output = tensor_create(o_shape, 2);
  Tensor* grad_hidden = tensor_create(h_shape, 2);
  
  float lr = 0.01f;
  int epochs = 1000;
  
  printf("Training XOR...\n");
  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0;
    
    for (int i = 0; i < 4; i++) {
      int in_shape[] = {2, 1};
      Tensor* input = tensor_create(in_shape, 2);
      input->values[0] = x_data[i][0];
      input->values[1] = x_data[i][1];
      target->values[0] = y_data[i][0];
      
      nn_linear_forward(l1, input, hidden);
      nn_relu_forward(hidden);
      nn_linear_forward(l2, hidden, output);
      
      epoch_loss += nn_mse_loss(output, target);
      
      nn_mse_gradient(output, target, grad_output);
      
      nn_linear_backward(l2, grad_output, grad_hidden);

      nn_relu_backward(grad_hidden, hidden);
      Tensor* grad_input_dummy = tensor_create(in_shape, 2);
      nn_linear_backward(l1, grad_hidden, grad_input_dummy);

      nn_sgd_update(l1, lr);
      nn_sgd_update(l2, lr);

      tensor_free(input);
      tensor_free(grad_input_dummy);
    }
    
    if (epoch % 100 == 0) {
      printf("Epoch %d - Loss: %f\n", epoch, epoch_loss / 4.0f);
    }
  }
  
  printf("\nTesting Results:\n");
  for (int i = 0; i < 4; i++) {
    int in_shape[] = {2, 1};
    Tensor* input = tensor_create(in_shape, 2);
    input->values[0] = x_data[i][0];
    input->values[1] = x_data[i][1];

    nn_linear_forward(l1, input, hidden);
    nn_relu_forward(hidden);
    nn_linear_forward(l2, hidden, output);

    printf("In: [%.0f, %.0f] Expected: %.0f Pred: %.4f\n", x_data[i][0], x_data[i][1], y_data[i][0], output->values[0]);
    
    tensor_free(input);
  }
  
  nn_linear_free(l1);
  nn_linear_free(l2);
  tensor_free(hidden);
  tensor_free(output);
  tensor_free(target);
  tensor_free(grad_output);
  tensor_free(grad_hidden);
      
  return 0;
}
