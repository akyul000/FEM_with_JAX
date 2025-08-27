import jax.numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import numpy as onp
import matplotlib.pyplot as plt
m = 4
b = 2



### Generate data ###

def create_data(n, std):
    # n = number of points
    # std = standard deviation

    key = jax.random.key(0)

    def eval_f(x):
        # y = np.sin(2 * np.pi * x)
        y = x * m + b
        return y

    x = onp.linspace(0, 1, n)
    y = jax.vmap(eval_f)(x)
    y += std * jax.random.normal(key, shape=y.shape)

    return x, y

n, std = 100, .2

x, y = create_data(n, std)

# fig, ax = plt.subplots()
# ax.plot(x, y, "ro")
# plt.show()



def initialize_parameters(n_x, n_y):
    onp.random.seed(3) # Fixate the results for testing

    W = np.array(onp.random.rand(n_y, n_x))
    b = np.array(onp.random.rand(n_y,1))
    
    parameters = {'W':W,
                  'b':b}
    
    return parameters

parameters = initialize_parameters(1, 1)
print('Initialization of parameters(W and b respectively)', parameters)
def Forward_Propagation(parameters, X):
    W = parameters['W'][0][0]
    b = parameters['b'][0][0]
    
    z = W*X + b
    # There are no activation function as sigmoid
    y_hat = z
    
    return y_hat

Y_hat = Forward_Propagation(parameters, x)
# Compute cost with cost or loss function
def Compute_Cost(params, X, Y):
    m = max(X.shape)
    y_hat = Forward_Propagation(params, X)
    Z = (y_hat - Y)
    L = 1 / (2*m) * np.sum(Z**2) 
    return L

cost = Compute_Cost(parameters, x, y)
print('Initial cost is ', cost)


@jax.jit
def Backwards_Propagation(parameters, X, Y):
    grad_fnctn = jax.grad(Compute_Cost)  # directly refer to the cost function here
    grads = grad_fnctn(parameters, X, Y)
    return grads


grads = Backwards_Propagation( parameters=parameters, X=x, Y=y)
print('Initial gradients', grads)

def update_parameters(params, grads, learning_rate=1.2):
    params = {
        "W": params["W"] - learning_rate * grads["W"],
        "b": params["b"] - learning_rate * grads["b"]
    }
    return params


parameters = update_parameters(parameters, grads)
print('Updated parameters', parameters)

learning_rate = 1.2
num_epochs = 1000

cost_history = []

# layerSize = layerSize(C)


num_epochs = 1000
def NN_model_JAX(X, Y, num_epochs = 1000, learning_rate=1.2):

    parameters = initialize_parameters(1,1)

    for i in range(num_epochs):
        # cost = Compute_Cost(parameters, X, Y)
        grads = Backwards_Propagation(parameters=parameters, X=X, Y=Y)
        parameters = update_parameters(parameters, grads)
        if i % 100 == 0:
            cost = Compute_Cost(parameters, X, Y)
            cost_history.append(cost)
            print(f"Epoch {i}: Cost = {cost:.4f}")

    return parameters

parameters_final = NN_model_JAX(x, y,num_epochs,learning_rate=0.2 )

def predict(X,parameters):
    prediction = Forward_Propagation(parameters, X)
    return prediction


Y_hat_predicted = predict(x, parameters_final)
print(f"final parameters {parameters_final}")
# Plot the final results
plt.figure()
plt.scatter(x, y, c='k',label='Data')
plt.scatter(x, Y_hat_predicted, c='r',label='Linear Model')
plt.xlabel('Normalized BMI')
plt.ylabel('Diabetes Risk')
plt.legend()
plt.show()

    


plt.figure()
plt.plot(cost_history)
plt.show()


# ############### 
# Y = y.reshape(-1,1)
# # create vander
# X = np.vander(x, N=2)

# theta = np.linalg.inv(X.T @ X) @ X.T @ Y
# print(theta)

# import jax
# import jax.numpy as np
# import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)

# # ----------- Generate Nonlinear Data -----------
# key = jax.random.key(0)
# n = 200
# x = np.linspace(-2, 2, n).reshape(-1, 1)
# true_y = np.sin(2 * np.pi * x) + 0.1 * x
# y = true_y + 0.1 * jax.random.normal(key, shape=true_y.shape)  # noisy data

# # Normalize input
# x = (x - np.mean(x)) / np.std(x)

# # ----------- Initialize Parameters -----------
# def init_params(key, hidden_size=50):
#     k1, k2, k3, k4 = jax.random.split(key, 4)
#     params = {
#         "W1": jax.random.normal(k1, (hidden_size, 1)) * np.sqrt(2/1),  # He init
#         "b1": np.zeros((hidden_size, 1)),
#         "W2": jax.random.normal(k2, (1, hidden_size)) * np.sqrt(2/hidden_size),
#         "b2": np.zeros((1, 1)),
#     }
#     return params

# # ----------- Forward Pass -----------
# def forward(params, X):
#     z1 = np.dot(params["W1"], X.T) + params["b1"]
#     a1 = np.tanh(z1)   # hidden layer
#     z2 = np.dot(params["W2"], a1) + params["b2"]
#     return z2.T  # shape (n, 1)

# # ----------- Loss Function -----------
# def loss_fn(params, X, Y):
#     Y_hat = forward(params, X)
#     return np.mean((Y_hat - Y)**2)

# # ----------- Training Loop -----------
# @jax.jit
# def update(params, X, Y, lr=0.01):
#     grads = jax.grad(loss_fn)(params, X, Y)
#     params = {k: params[k] - lr * grads[k] for k in params}
#     return params

# params = init_params(key)

# loss_history = []
# for epoch in range(5000):
#     params = update(params, x, y, lr=0.01)
#     if epoch % 500 == 0:
#         l = loss_fn(params, x, y)
#         loss_history.append(l)
#         print(f"Epoch {epoch}, Loss = {l:.4f}")

# # ----------- Predictions -----------
# y_pred = forward(params, x)

# # ----------- Plot Results -----------
# plt.figure(figsize=(8,5))
# plt.scatter(x, y, color="k", alpha=0.5, label="Noisy Data")
# plt.plot(x, true_y, "g--", label="True Function")
# plt.plot(x, y_pred, "r", label="NN Prediction")
# plt.legend()
# plt.show()

# # Plot training loss
# plt.figure()
# plt.plot(np.arange(0, 5000, 500), loss_history, marker="o")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.show()


# import jax
# import jax.numpy as np
# import numpy as onp
# import matplotlib.pyplot as plt

# # Define layer sizes
# def layer_sizes(X, Y, hidden_units=10):
#     n_x = X.shape[0]   # input size
#     n_h = hidden_units # hidden layer size
#     n_y = Y.shape[0]   # output size
#     return (n_x, n_h, n_y)

# # Initialize parameters for multiple layers
# def initialize_parameters(n_x, n_h, n_y):
#     onp.random.seed(3)
#     parameters = {
#         "W1": np.array(onp.random.randn(n_h, n_x) * 0.01),
#         "b1": np.zeros((n_h, 1)),
#         "W2": np.array(onp.random.randn(n_y, n_h) * 0.01),
#         "b2": np.zeros((n_y, 1))
#     }
#     return parameters

# # Forward propagation
# def forward_propagation(parameters, X):
#     W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    
#     Z1 = np.dot(W1, X) + b1
#     A1 = np.tanh(Z1)                # hidden activation
#     Z2 = np.dot(W2, A1) + b2
#     A2 = Z2                         # linear output (regression)
    
#     cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
#     return A2, cache

# # Cost function (MSE)
# def compute_cost(parameters, X, Y):
#     m = X.shape[1]
#     Y_hat, _ = forward_propagation(parameters, X)
#     cost = (1/(2*m)) * np.sum((Y_hat - Y) ** 2)
#     return cost

# # Backpropagation with JAX
# grad_fn = jax.grad(compute_cost)

# # Update step
# def update_parameters(parameters, grads, learning_rate=0.01):
#     return {k: v - learning_rate * grads[k] for k, v in parameters.items()}

# # Training loop
# def nn_model(X, Y, hidden_units=10, num_epochs=1000, learning_rate=0.01):
#     n_x, n_h, n_y = layer_sizes(X, Y, hidden_units)
#     parameters = initialize_parameters(n_x, n_h, n_y)
#     cost_history = []

#     for i in range(num_epochs):
#         grads = grad_fn(parameters, X, Y)
#         parameters = update_parameters(parameters, grads, learning_rate)

#         if i % 100 == 0:
#             cost = compute_cost(parameters, X, Y)
#             cost_history.append(cost)
#             print(f"Epoch {i}: Cost = {cost:.4f}")

#     return parameters, cost_history

# # Predict
# def predict(X, parameters):
#     Y_hat, _ = forward_propagation(parameters, X)
#     return Y_hat

# X = x.reshape(1, -1)
# Y = y.reshape(1, -1)

# params = nn_model(X, Y, hidden_units=10, num_epochs=1000, learning_rate=0.01)[0]