# Load necessary libraries
library(keras)
library(ggplot2)
library(grid)

# Read train and test data
mnist_train <- read.csv("mnist_train.csv", header = FALSE)
mnist_test <- read.csv("mnist_test.csv", header = FALSE)

# Split the features from output
x_train <- as.matrix(mnist_train[, -1]) / 255
y_train <- mnist_train[, 1]

x_test <- as.matrix(mnist_test[, -1]) / 255
y_test <- mnist_test[, 1]

# Build the model
model <- keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = 784) %>%
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Train the model
model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1
)

# Evaluate the models
score <- model %>% evaluate(x_test, y_test)
cat(sprintf("Test Accuracy: %.2f%%\n", score$accuracy * 100))

# Predict a random image from za dataset
index <- sample(1:nrow(x_test), 1)
prediction <- model %>% predict_classes(x_test[index, , drop = FALSE])

# Convert pixels to 28x28 image to flat vector 
image <- matrix(x_test[index, ], nrow = 28, byrow = TRUE)

# Ploting the image
grid.raster(image, interpolate = FALSE)
title <- paste("Predicted:", prediction, "True:", y_test[index])
title(main = title)
