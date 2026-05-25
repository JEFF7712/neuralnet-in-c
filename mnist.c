#include "nn.h"
#include "tensor.h"
#include "arena.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define MNIST_IMAGE_SIZE 784
#define MNIST_CLASSES 10

typedef struct {
  Tensor** images;
  Tensor** labels;
  uint8_t* digits;
  int count;
  int rows;
  int cols;
} MnistDataset;

static uint32_t read_be_u32(FILE* f) {
  uint8_t bytes[4];
  if (fread(bytes, 1, sizeof(bytes), f) != sizeof(bytes)) {
    return 0;
  }
  return ((uint32_t)bytes[0] << 24) |
         ((uint32_t)bytes[1] << 16) |
         ((uint32_t)bytes[2] << 8) |
         (uint32_t)bytes[3];
}

static int mnist_load_images(GraphContext* ctx, const char* path, MnistDataset* dataset) {
  FILE* f = fopen(path, "rb");
  if (!f) return 0;

  uint32_t magic = read_be_u32(f);
  uint32_t count = read_be_u32(f);
  uint32_t rows = read_be_u32(f);
  uint32_t cols = read_be_u32(f);

  if (magic != 2051 || rows != 28 || cols != 28) {
    fclose(f);
    return 0;
  }

  // Allocate pointer array directly in the persistent arena
  dataset->images = (Tensor**)arena_calloc(ctx->arena, count * sizeof(Tensor*), 8);
  dataset->count = (int)count;
  dataset->rows = (int)rows;
  dataset->cols = (int)cols;

  int image_shape[] = {MNIST_IMAGE_SIZE, 1};
  uint8_t pixels[MNIST_IMAGE_SIZE];

  for (int i = 0; i < dataset->count; i++) {
    if (fread(pixels, 1, sizeof(pixels), f) != sizeof(pixels)) return 0;
    
    // Dataset images do not require gradients
    dataset->images[i] = tensor_create(ctx, image_shape, 2, false);
    for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
      dataset->images[i]->values[j] = (float)pixels[j] / 255.0f;
    }
  }

  fclose(f);
  return 1;
}

static int mnist_load_labels(GraphContext* ctx, const char* path, MnistDataset* dataset) {
  FILE* f = fopen(path, "rb");
  if (!f) return 0;

  uint32_t magic = read_be_u32(f);
  uint32_t count = read_be_u32(f);
  
  if (magic != 2049 || count != (uint32_t)dataset->count) {
    fclose(f);
    return 0;
  }

  dataset->labels = (Tensor**)arena_calloc(ctx->arena, count * sizeof(Tensor*), 8);
  dataset->digits = (uint8_t*)arena_calloc(ctx->arena, count * sizeof(uint8_t), 1);
  
  int label_shape[] = {MNIST_CLASSES, 1};

  for (int i = 0; i < dataset->count; i++) {
    uint8_t digit;
    if (fread(&digit, 1, sizeof(digit), f) != sizeof(digit)) return 0;

    dataset->digits[i] = digit;
    dataset->labels[i] = tensor_create_zeros(ctx, label_shape, 2, false);
    dataset->labels[i]->values[digit] = 1.0f;
  }

  fclose(f);
  return 1;
}

static int arg_or_default(char** argv, int argc, int index, int fallback) {
  if (argc <= index) return fallback;
  int value = atoi(argv[index]);
  return value > 0 ? value : fallback;
}

int main(int argc, char** argv) {
  srand((unsigned int)time(NULL));

  // 1. Initialize persistent memory context (256 MB)
  GraphContext* model_ctx = ctx_create(256 * 1024 * 1024);
  
  // 2. Initialize transient compute context (64 MB)
  GraphContext* compute_ctx = ctx_create(64 * 1024 * 1024);

  const char* train_images_path = "data/mnist/train-images-idx3-ubyte";
  const char* train_labels_path = "data/mnist/train-labels-idx1-ubyte";
  const char* test_images_path = "data/mnist/t10k-images-idx3-ubyte";
  const char* test_labels_path = "data/mnist/t10k-labels-idx1-ubyte";

  MnistDataset train;
  MnistDataset test;

  if (!mnist_load_images(model_ctx, train_images_path, &train) ||
      !mnist_load_labels(model_ctx, train_labels_path, &train) ||
      !mnist_load_images(model_ctx, test_images_path, &test) ||
      !mnist_load_labels(model_ctx, test_labels_path, &test)) {
    printf("Failed to load dataset.\n");
    return 1;
  }

  printf("Loaded MNIST: %d train images, %d test images (%dx%d)\n",
         train.count, test.count, train.rows, train.cols);

  int train_limit = arg_or_default(argv, argc, 1, 10000);
  int epochs = arg_or_default(argv, argc, 2, 1);
  int test_limit = arg_or_default(argv, argc, 3, 1000);

  if (train_limit > train.count) train_limit = train.count;
  if (test_limit > test.count) test_limit = test.count;

  LinearLayer* l1 = linear_layer_create(model_ctx, MNIST_IMAGE_SIZE, 128);
  LinearLayer* l2 = linear_layer_create(model_ctx, 128, MNIST_CLASSES);

  const float learning_rate = 0.001f;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < train_limit; i++) {
      // Forward Pass
      Tensor* h1 = nn_linear_forward(compute_ctx, l1, train.images[i]);
      Tensor* h2 = nn_gelu_forward(compute_ctx, h1);
      Tensor* logits = nn_linear_forward(compute_ctx, l2, h2);
      Tensor* probs = nn_softmax_forward(compute_ctx, logits);
      
      Tensor* loss = nn_cross_entropy_loss(compute_ctx, probs, train.labels[i]);

      // Metrics
      epoch_loss += loss->values[0];
      if (tensor_argmax(probs) == train.digits[i]) {
        correct++;
      }

      // Backward Pass 
      tensor_backward(compute_ctx, loss);

      // Optimizer Step
      nn_adamw_update(l1, learning_rate, epoch * train_limit + i + 1, 0.9f, 0.999f, 1e-8f, 1e-4f);
      nn_adamw_update(l2, learning_rate, epoch * train_limit + i + 1, 0.9f, 0.999f, 1e-8f, 1e-4f);
      
      // nn_sgd_update(l1, learning_rate);
      // nn_sgd_update(l2, learning_rate);

      // Instantly drop the computational graph memory
      ctx_reset(compute_ctx);
    }

    printf("Epoch %d/%d - loss: %.4f - accuracy: %.2f%%\n",
           epoch + 1, epochs,
           epoch_loss / (float)train_limit,
           100.0f * (float)correct / (float)train_limit);
  }

  // Testing Loop
  int correct = 0;
  for (int i = 0; i < test_limit; i++) {
    Tensor* h1 = nn_linear_forward(compute_ctx, l1, test.images[i]);
    Tensor* h2 = nn_gelu_forward(compute_ctx, h1);
    Tensor* logits = nn_linear_forward(compute_ctx, l2, h2);
    Tensor* probs = nn_softmax_forward(compute_ctx, logits);

    int prediction = tensor_argmax(probs);
    if (prediction == test.digits[i]) {
      correct++;
    }
    
    ctx_reset(compute_ctx);
  }

  printf("Test accuracy on %d examples: %.2f%%\n", test_limit, 100.0f * (float)correct / (float)test_limit);

  ctx_destroy(compute_ctx);
  ctx_destroy(model_ctx);

  return 0;
}