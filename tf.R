library(reticulate)
library(keras)
library(tensorflow)

num_samples <- 1000
height <- 224
width <- 224
num_classes <- 1000

#Use all available GPUs
strategy <- tf$distribute$MirroredStrategy()

#You also can selectively choose GPUs 
#strategy <- tf$distribute$MirroredStrategy(devices=c("GPU:0","GPU:2"))

#Print available replicas / GPUs
strategy$num_replicas_in_sync

#Define model and compile it
with (strategy$scope(), {
  parallel_model <- application_xception(
    weights = NULL,
    input_shape = c(height, width, 3),
    classes = num_classes
  )
  parallel_model %>% compile(
    loss = "categorical_crossentropy",
    optimizer = "rmsprop"
  )
})

# Generate dummy data.
x <- array(runif(num_samples * height * width*3),
           dim = c(num_samples, height, width, 3))
y <- array(runif(num_samples * num_classes),
           dim = c(num_samples, num_classes))

# This `fit` call will be distributed on the selected GPUs.
# Since the batch size is 256, each GPU will process 256/#GPU samples.
parallel_model %>% fit(x, y, epochs = 20, batch_size = 256)

# Save model via the template model (which shares the same weights):
model %>% save_model_hdf5("my_model.h5")

