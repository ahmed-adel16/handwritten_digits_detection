# Load necessary libraries
library(keras)
library(ggplot2)
library(grid)
library(tensorflow)
library(keras)
# Read train and test data
mnist_train <- read.csv("data/mnist_train.csv", stringsAsFactors = FALSE)
mnist_test <- read.csv("data/mnist_test.csv", stringsAsFactors = FALSE)

# Split labels and features
x_train <- as.matrix(sapply(mnist_train[, -1], as.numeric)) / 255
y_train <- as.numeric(mnist_train[, 1])

x_test <- as.matrix(sapply(mnist_test[, -1], as.numeric)) / 255
y_test <- as.numeric(mnist_test[, 1])

# Build model
model <- keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = 784) %>%
  layer_dense(units = 10, activation = "softmax")

