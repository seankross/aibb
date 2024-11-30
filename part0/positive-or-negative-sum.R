library(torch)

perceptron <- nn_module(
  initialize = function() {
    self$layer1 <- nn_linear(2, 1)
  },
  
  forward = function(x) {
    torch_sigmoid(self$layer1(x))
  }
)

set.seed(2024-05-14)
n <- 100
inputs <- matrix(runif(2 * n, -10, 10), ncol = 2)
labels <- matrix(ifelse(inputs[,1] + inputs[,2] > 0, 1, 0), ncol = 1)

inputs <- torch_tensor(inputs, dtype = torch_float())
labels <- torch_tensor(labels, dtype = torch_float())

model <- perceptron()

loss_fn <- nnf_mse_loss
optimizer <- optim_sgd(model$parameters, lr = 0.01)

num_epochs <- 10000
for (epoch in 1:num_epochs) {
  #model$zero_grad()
  outputs <- model$forward(inputs)
  loss <- loss_fn(outputs, labels)
  loss$backward()
  optimizer$step()
  
  if (epoch %% 5 == 0) {
    cat(sprintf("Epoch: %d, Loss: %f\n", epoch, loss$item()))
  }
}

test_inputs <- torch_tensor(matrix(c(4, -3, 1, 2, 3, -7), ncol = 2), dtype = torch_float())
model$forward(test_inputs)

model$forward(matrix(c(2, 3), ncol = 2) |> torch_tensor(dtype = torch_float()))

model$layer1$weight
model$layer1$bias

x <- (matrix(c(2, 3), ncol = 2) %*% matrix(c(-0.5743, 0.0678), nrow = 2) + -0.3523)

1 / (1 + exp(-x))

###

new_weights <- model$parameters$layer1.weight - 0.01 * model$parameters$layer1.weight$grad




### multiplication!! wide learning!!

library(torch)

perceptron <- nn_module(
  initialize = function() {
    self$layer1 <- nn_linear(2, 4)
    self$layer2 <- nn_linear(4, 1)
  },
  
  forward = function(x) {
    self$layer1(x) |> 
      torch_relu() |> 
      self$layer2() |> 
      torch_sigmoid()
  }
)

set.seed(2024-05-14)
n <- 100
inputs <- matrix(runif(2 * n, -10, 10), ncol = 2)
labels <- matrix(ifelse(inputs[,1] * inputs[,2] > 0, 1, 0), ncol = 1)

inputs <- torch_tensor(inputs, dtype = torch_float())
labels <- torch_tensor(labels, dtype = torch_float())

model <- perceptron()

loss_fn <- nnf_mse_loss
optimizer <- optim_sgd(model$parameters, lr = 0.01)

num_epochs <- 10000
for (epoch in 1:num_epochs) {
  #model$zero_grad()
  outputs <- model$forward(inputs)
  loss <- loss_fn(outputs, labels)
  loss$backward()
  optimizer$step()
  
  if (epoch %% 5 == 0) {
    cat(sprintf("Epoch: %d, Loss: %f\n", epoch, loss$item()))
  }
}

test_inputs <- torch_tensor(matrix(c(4, -3, 1, 2, 3, -7), ncol = 2), dtype = torch_float())
model$forward(test_inputs)
