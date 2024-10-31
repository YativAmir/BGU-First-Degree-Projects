# libraries
import numpy as np
import pandas as pd

# set seed
np.random.seed(120)

# Arrays 
Outputs_Matrix = []
Weights_List = []
Biases_List = []


# Dense layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.0001*np.random.rand(n_inputs, n_neurons)
        self.biases = 0.0001*np.random.rand(1, n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases


# ReLU activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Function to create the neural network based on the layer structure
def create_network(layers):
    network = []
    for i in range(len(layers) - 1):
        network.append(Layer_Dense(layers[i], layers[i + 1]))
        network.append(Activation_ReLU())
    return network


# Perform a forward pass through the network
def forward_pass(network, X):
    inputs = X
    for layer in network:
        layer.forward(inputs)
        inputs = layer.output
    return inputs

# Import dataset
file_path = "/Users/doron/OneDrive/שולחן העבודה/פרויקט גמר/train_ready_for_ANN.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, 1:]

# Shuffle the DataFrame rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)

df_sale_price = df_shuffled[['SalePrice']]
df_without_sale_price = df_shuffled.drop(columns=['SalePrice'])

# Create input X and y
X = df_without_sale_price.iloc[:-200]
y = df_sale_price.iloc[:-200]

# Create holdout input holdout & holdout output Y
X_holdout = df_without_sale_price.iloc[-200:]
y_holdout = df_sale_price.iloc[-200:]

# Define the network layers
layers = [X.shape[1], 10, 10, 10, 10, 1]

# Function to create and save multiple networks
def create_and_save_networks(num_networks):
    for _ in range(num_networks):
        # Create the network
        network = create_network(layers)

        # Perform a forward pass through the network
        output = forward_pass(network, df_without_sale_price)

        # Collect weights and biases
        Layer_weights = [layer.get_weights() for layer in network if isinstance(layer, Layer_Dense)]
        Layer_biases = [layer.get_biases() for layer in network if isinstance(layer, Layer_Dense)]

        # Save outputs, weights, and biases
        Outputs_Matrix.append(output)
        Weights_List.append(Layer_weights)
        Biases_List.append(Layer_biases)

# Define number of networks
num_rows= len(X)  # Specific for Square matrix
num_networks = num_rows

# Create and save multiple networks
create_and_save_networks(num_networks)

# to create Matrix
num_Of_Samples = X.shape[0]
Outputs_Matrix_reshaped = [x.reshape(-1) for x in Outputs_Matrix]

# put all samples test and train ANN results in matrix
M_test_and_train = np.array(Outputs_Matrix_reshaped)

# to find M 
num_of_test_sumples=len(X_holdout)
M = M_test_and_train[:, :-num_of_test_sumples]

# to find B
M_transposed = np.transpose(M)
y = np.array(y)
b = np.linalg.solve(M_transposed, y)


# to find test matrix
M_test = M_test_and_train[:, -num_of_test_sumples:]
M_test_transposed = np.transpose(M_test)

# calculation result
result_list = []
for i in range(num_of_test_sumples):
  result_list.append(np.dot( M_test_transposed[i] , b))

result_frame = pd.DataFrame(result_list, columns=['Values'])

#find MSE
y_holdout.reset_index(drop=True, inplace=True)
df_concat = pd.concat([y_holdout, result_frame], axis=1)

def calculate_mse(df, col1, col2):
    
    mse = np.mean((df[col1] - df[col2]) ** 2)
    return mse

mse_value = calculate_mse(df_concat, 'Values', 'SalePrice')
mse_sqrt_value = np.sqrt(mse_value) 