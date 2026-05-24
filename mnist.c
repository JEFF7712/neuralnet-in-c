#include "nn.h"
#include "tensor.h"

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

static void mnist_dataset_init(MnistDataset* dataset) {
  dataset->images = NULL;
  dataset->labels = NULL;
  dataset->digits = NULL;
  dataset->count = 0;
  dataset->rows = 0;
  dataset->cols = 0;
}

static void mnist_free(MnistDataset* dataset) {
  if (!dataset) return;

  for (int i = 0; i < dataset->count; i++) {
    if (dataset->images) tensor_free(dataset->images[i]);
    if (dataset->labels) tensor_free(dataset->labels[i]);
  }

  free(dataset->images);
  free(dataset->labels);
  free(dataset->digits);
  mnist_dataset_init(dataset);
}

static int mnist_load_images(const char* path, MnistDataset* dataset) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 0;
  }

  uint32_t magic = read_be_u32(f);
  uint32_t count = read_be_u32(f);
  uint32_t rows = read_be_u32(f);
  uint32_t cols = read_be_u32(f);

  if (magic != 2051 || rows != 28 || cols != 28) {
    fprintf(stderr, "Invalid MNIST image file: %s\n", path);
    fclose(f);
    return 0;
  }

  dataset->images = (Tensor**)calloc(count, sizeof(*dataset->images));
  if (!dataset->images) {
    fclose(f);
    return 0;
  }

  dataset->count = (int)count;
  dataset->rows = (int)rows;
  dataset->cols = (int)cols;

  int image_shape[] = {MNIST_IMAGE_SIZE, 1};
  uint8_t pixels[MNIST_IMAGE_SIZE];

  for (int i = 0; i < dataset->count; i++) {
    if (fread(pixels, 1, sizeof(pixels), f) != sizeof(pixels)) {
      fprintf(stderr, "Unexpected end of image file: %s\n", path);
      fclose(f);
      return 0;
    }

    dataset->images[i] = tensor_create(image_shape, 2);
    if (!dataset->images[i]) {
      fclose(f);
      return 0;
    }

    for (int j = 0; j < MNIST_IMAGE_SIZE; j++) {
      dataset->images[i]->values[j] = (float)pixels[j] / 255.0f;
    }
  }

  fclose(f);
  return 1;
}

static int mnist_load_labels(const char* path, MnistDataset* dataset) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 0;
  }

  uint32_t magic = read_be_u32(f);
  uint32_t count = read_be_u32(f);

  if (magic != 2049 || (int)count != dataset->count) {
    fprintf(stderr, "Invalid MNIST label file: %s\n", path);
    fclose(f);
    return 0;
  }

  dataset->labels = (Tensor**)calloc(count, sizeof(*dataset->labels));
  dataset->digits = (uint8_t*)calloc(count, sizeof(*dataset->digits));
  if (!dataset->labels || !dataset->digits) {
    fclose(f);
    return 0;
  }

  int label_shape[] = {MNIST_CLASSES, 1};

  for (int i = 0; i < dataset->count; i++) {
    uint8_t digit;
    if (fread(&digit, 1, sizeof(digit), f) != sizeof(digit)) {
      fprintf(stderr, "Unexpected end of label file: %s\n", path);
      fclose(f);
      return 0;
    }

    if (digit >= MNIST_CLASSES) {
      fprintf(stderr, "Invalid MNIST digit %u in %s\n", digit, path);
      fclose(f);
      return 0;
    }

    dataset->digits[i] = digit;
    dataset->labels[i] = tensor_create_zeros(label_shape, 2);
    if (!dataset->labels[i]) {
      fclose(f);
      return 0;
    }
    dataset->labels[i]->values[digit] = 1.0f;
  }

  fclose(f);
  return 1;
}

static int mnist_load(const char* image_path,
                      const char* label_path,
                      MnistDataset* dataset) {
  mnist_dataset_init(dataset);

  if (!mnist_load_images(image_path, dataset) ||
      !mnist_load_labels(label_path, dataset)) {
    mnist_free(dataset);
    return 0;
  }

  return 1;
}

static int arg_or_default(char** argv, int argc, int index, int fallback) {
  if (argc <= index) return fallback;

  int value = atoi(argv[index]);
  return value > 0 ? value : fallback;
}

static float cross_entropy_for_one_hot(const Tensor* probabilities,
                                       const Tensor* target) {
  const float epsilon = 1e-7f;
  float loss = 0.0f;

  for (int i = 0; i < MNIST_CLASSES; i++) {
    if (target->values[i] > 0.0f) {
      loss -= logf(probabilities->values[i] + epsilon);
      break;
    }
  }

  return loss;
}

int main(int argc, char** argv) {
  srand((unsigned int)time(NULL));

  const char* train_images_path = "data/mnist/train-images-idx3-ubyte";
  const char* train_labels_path = "data/mnist/train-labels-idx1-ubyte";
  const char* test_images_path = "data/mnist/t10k-images-idx3-ubyte";
  const char* test_labels_path = "data/mnist/t10k-labels-idx1-ubyte";

  MnistDataset train;
  MnistDataset test;

  if (!mnist_load(train_images_path, train_labels_path, &train) ||
      !mnist_load(test_images_path, test_labels_path, &test)) {
    return 1;
  }

  printf("Loaded MNIST: %d train images, %d test images (%dx%d)\n",
         train.count, test.count, train.rows, train.cols);

  int train_limit = arg_or_default(argv, argc, 1, 10000);
  int epochs = arg_or_default(argv, argc, 2, 1);
  int test_limit = arg_or_default(argv, argc, 3, 1000);

  if (train_limit > train.count) train_limit = train.count;
  if (test_limit > test.count) test_limit = test.count;

  LinearLayer* l1 = linear_layer_create(MNIST_IMAGE_SIZE, 128);
  LinearLayer* l2 = linear_layer_create(128, MNIST_CLASSES);

  int hidden_shape[] = {128, 1};
  int output_shape[] = {MNIST_CLASSES, 1};
  int input_grad_shape[] = {MNIST_IMAGE_SIZE, 1};

  Tensor* hidden = tensor_create(hidden_shape, 2);
  Tensor* hidden_before_relu = tensor_create(hidden_shape, 2);
  Tensor* output = tensor_create(output_shape, 2);
  Tensor* grad_output = tensor_create(output_shape, 2);
  Tensor* grad_hidden = tensor_create(hidden_shape, 2);
  Tensor* grad_input = tensor_create(input_grad_shape, 2);

  const float learning_rate = 0.001f;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float epoch_loss = 0.0f;
    int correct = 0;

    for (int i = 0; i < train_limit; i++) {
      nn_linear_forward(l1, train.images[i], hidden);

      for (int j = 0; j < (int)hidden->total_elements; j++) {
        hidden_before_relu->values[j] = hidden->values[j];
      }

      nn_gelu_forward(hidden);
      nn_linear_forward(l2, hidden, output);
      nn_softmax_forward(output);

      epoch_loss += cross_entropy_for_one_hot(output, train.labels[i]);
      if (tensor_argmax(output) == train.digits[i]) {
        correct++;
      }

      nn_cross_entropy_gradient(output, train.labels[i], grad_output);
      nn_linear_backward(l2, grad_output, grad_hidden);
      nn_gelu_backward(grad_hidden, hidden_before_relu);
      nn_linear_backward(l1, grad_hidden, grad_input);
      
      nn_adamw_update(l1, learning_rate, epoch * train_limit + i + 1, 0.9f, 0.999f, 1e-8f, 1e-4f);
      nn_adamw_update(l2, learning_rate, epoch * train_limit + i + 1, 0.9f, 0.999f, 1e-8f, 1e-4f);
      
      //nn_sgd_update(l1, 0.01f);
      //nn_sgd_update(l2, 0.01f);
      
    }

    printf("Epoch %d/%d - loss: %.4f - accuracy: %.2f%%\n",
           epoch + 1,
           epochs,
           epoch_loss / (float)train_limit,
           100.0f * (float)correct / (float)train_limit);
  }

  int correct = 0;
  for (int i = 0; i < test_limit; i++) {
    nn_linear_forward(l1, test.images[i], hidden);
    nn_gelu_forward(hidden);
    nn_linear_forward(l2, hidden, output);
    nn_softmax_forward(output);

    int prediction = tensor_argmax(output);
    if (prediction == test.digits[i]) {
      correct++;
    }

    if (i < 10) {
      printf("test[%d] expected=%u predicted=%d confidence=%.2f%%\n",
             i,
             test.digits[i],
             prediction,
             100.0f * output->values[prediction]);
    }
  }

  printf("Test accuracy on %d examples: %.2f%%\n",
         test_limit,
         100.0f * (float)correct / (float)test_limit);

  tensor_free(hidden);
  tensor_free(hidden_before_relu);
  tensor_free(output);
  tensor_free(grad_output);
  tensor_free(grad_hidden);
  tensor_free(grad_input);
  nn_linear_free(l1);
  nn_linear_free(l2);
  mnist_free(&train);
  mnist_free(&test);

  return 0;
}
