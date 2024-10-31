import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(200)

# Helper Functions:

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU Activation Function and its Derivative

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Initialize Network:

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.rand(self.layers[i], self.layers[i + 1])*0.01
            bias = np.random.rand(1, self.layers[i + 1])
            self.weights.append(weight)
            self.biases.append(bias)

    def feedforward(self, X):
        self.activations = [X]
        for i in range(len(self.layers) - 1):
            net_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            activation = relu(net_input)
            self.activations.append(activation)
        return self.activations[-1]

    def backpropagate(self, X, y):
        y_pred = self.feedforward(X)
        errors = [y_pred - y]
        deltas = [errors[-1] * relu_derivative(y_pred)]

        for i in range(len(self.layers) - 2, 0, -1):
            error = np.dot(deltas[-1], self.weights[i].T)
            delta = error * relu_derivative(self.activations[i])
            errors.append(error)
            deltas.append(delta)

        errors.reverse()
        deltas.reverse()

        grad_weights = []
        grad_biases = []

        for i in range(len(self.weights)):
            grad_weights.append(np.dot(self.activations[i].T, deltas[i]))
            grad_biases.append(np.sum(deltas[i], axis=0, keepdims=True))

        return grad_weights, grad_biases

    def get_weights_and_biases(self):
        params = []
        for w, b in zip(self.weights, self.biases):
            params.extend(w.flatten())
            params.extend(b.flatten())
        return np.array(params)

    def set_weights_and_biases(self, params):
        start = 0
        for i in range(len(self.layers) - 1):
            end = start + self.layers[i] * self.layers[i + 1]
            self.weights[i] = params[start:end].reshape((self.layers[i], self.layers[i + 1]))
            start = end
            end = start + self.layers[i + 1]
            self.biases[i] = params[start:end].reshape((1, self.layers[i + 1]))
            start = end

    def loss(self, params, X, y):
        self.set_weights_and_biases(params)
        y_pred = self.feedforward(X)
        return mean_squared_error(y, y_pred)

    def gradient(self, params, X, y):
        self.set_weights_and_biases(params)
        grad_weights, grad_biases = self.backpropagate(X, y)
        grad = []
        for gw, gb in zip(grad_weights, grad_biases):
            grad.extend(gw.flatten())
            grad.extend(gb.flatten())
        return np.array(grad)

    def fit(self, X, y, maxiter=100, learning_rate=0.01):
        params = self.get_weights_and_biases()
        options = {'maxiter': maxiter, 'disp': True}
        self.loss_history = []

        def callback(params):
            current_loss = self.loss(params, X, y)
            self.loss_history.append(current_loss)
            print(f'Iteration: {len(self.loss_history)}, Loss: {current_loss}')

        result = minimize(fun=self.loss, x0=params, args=(X, y), jac=self.gradient, method='BFGS', options=options, callback=callback)
        self.set_weights_and_biases(result.x)

    def predict(self, X):
        return self.feedforward(X)

# Load Data:

# Example data loading (replace with your own data)
file_path = "/Users/doron/OneDrive/שולחן העבודה/פרויקט גמר/train_ready_for_ANN.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, 1:]

# Shuffle the DataFrame rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)


df_sale_price = df_shuffled[['SalePrice']]
df_without_sale_price = df_shuffled.drop(columns=['SalePrice'])

X_train = df_without_sale_price.iloc[:-200]
y_train = df_sale_price.iloc[:-200]

X_holdout = df_without_sale_price.iloc[-200:]
y_holdout = df_sale_price.iloc[-200:]

# from data frame to array 
X_train = X_train.values.astype(np.float64)
y_train = y_train.values.astype(np.float64)

X_holdout = X_holdout.values.astype(np.float64)
y_holdout = y_holdout.values.astype(np.float64)

# Train the Model:

# Define the layers: input layer size, hidden layer sizes, output layer size
layers = [X_train.shape[1], 10, 10, 10, 10,  1]  # Example: 1 hidden layer with 10 neurons

# Initialize and train the neural network
nn = NeuralNetwork(layers)
nn.fit(X_train, y_train, maxiter=100)

# Evaluate the model on the holdout set
holdout_loss = nn.loss(nn.get_weights_and_biases(), X_holdout, y_holdout)
print(f'Holdout Set Loss: {holdout_loss}')

# Make Predictions:

predictions = nn.predict(X_holdout)
print(predictions)

predictions_frame = pd.DataFrame(predictions, columns=['Values'])

#find MSE
y_holdout_frame = pd.DataFrame(y_holdout, columns=['SalePrice'])
y_holdout_frame.reset_index(drop=True, inplace=True)

df_concat = pd.concat([y_holdout_frame, predictions_frame], axis=1)

def calculate_mse(df, col1, col2):
    
    mse = np.mean((df[col1] - df[col2]) ** 2)
    return mse

mse_value = calculate_mse(df_concat, 'Values', 'SalePrice')
mse_sqrt_value = np.sqrt(mse_value) 