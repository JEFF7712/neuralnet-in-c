# neuralnet

A small neural network library in C17 with reverse-mode autograd, arena allocation, and no external dependencies (just `libm`).

## Features

- Tensor engine with strides, broadcasting, and a gradient tape (`tensor.[ch]`)
- Layers: linear, layer norm; activations: ReLU, GELU, softmax (`nn.[ch]`)
- Losses: MSE, cross-entropy
- Optimizers: SGD, RMSprop, AdamW
- Bump/arena allocator for graph memory (`arena.[ch]`)

## MNIST results

Trained 3 epochs on 60k images with a linear + softmax model:

```
Epoch 1/3 - loss: 0.2227 - accuracy: 93.36%
Epoch 2/3 - loss: 0.0980 - accuracy: 97.10%
Epoch 3/3 - loss: 0.0682 - accuracy: 97.99%
Test accuracy on 10000 examples: 96.82%
```
