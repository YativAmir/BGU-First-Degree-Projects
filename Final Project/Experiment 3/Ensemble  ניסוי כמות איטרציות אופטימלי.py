# libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
# libraries
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
# libraries
import numpy as np
import pandas as pd
# imports
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Array
mse_sqrt_value_Array = []

#set seed
np.random.seed(6)

# Import dataset
file_path = "/Users/doron/OneDrive/שולחן העבודה/פרויקט גמר/train_ready_for_ANN.csv"
df = pd.read_csv(file_path)

# remove ID
df = df.iloc[:, 1:]

# Shuffle the DataFrame rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)

X_Full = df_shuffled.drop(columns=['SalePrice'])
Y_Full = df_shuffled[['SalePrice']]

# Create input X and y
X = X_Full.iloc[:-200]
Y = Y_Full.iloc[:-200]
# Create vector
Y = np.ravel(Y)

# Create holdout input holdout & holdout output Y
X_holdout = X_Full.iloc[-200:]
y_holdout = Y_Full.iloc[-200:]
# Create vector
Y_Full = np.ravel(Y_Full)

# Define the network layers
def create_and_save_networks(num_networks, X_train, y_train, j):
    for _ in range(num_networks):
        # Create the network
        model = MLPRegressor(
                hidden_layer_sizes=(10, 10, 10, 10),  # 4 layers with 100 neurons each
                activation='relu',  # ReLU activation function
                solver='lbfgs',  # BFGS optimizer
                max_iter=j,  # Maximum number of iterations
            )

        # Fit the network on the training data
        model.fit(X_train, y_train)
        # Save outputs
        Outputs_Matrix.append(model.predict(X_train))

for j in range(5, 66, 10):
 # Array
 Outputs_Matrix = []
 # Define number of networks
 num_rows = len(X)  # Specific for Square matrix
 num_networks = num_rows

 # Create and save multiple networks
 create_and_save_networks(num_networks, X_Full, Y_Full, j)

 # put all samples test and train ANN results in matrix
 M_test_and_train = np.array(Outputs_Matrix)
 M_test_and_train_transposed = M_test_and_train.transpose()

 # to find M
 num_of_test_sumples = len(X_holdout)
 M = M_test_and_train_transposed[:-num_of_test_sumples, :]

 # to find B
 b = np.linalg.solve(M,Y) # i need to chack it

 # to find test matrix
 M_test = M_test_and_train_transposed[-num_of_test_sumples:, :]

 M_test_transposed = np.transpose(M_test)

 # calculation result
 result_list = []
 for i in range(num_of_test_sumples):
    result_list.append(np.dot(M_test_transposed[:, i], b))

 result_frame = pd.DataFrame(result_list, columns=['Values'])

 #find MSE
 y_holdout.reset_index(drop=True, inplace=True)
 df_concat = pd.concat([y_holdout, result_frame], axis=1)

 def calculate_mse(df, col1, col2):
    
    mse = np.mean((df[col1] - df[col2]) ** 2)
    return mse

 mse_value = calculate_mse(df_concat, 'Values', 'SalePrice')
 mse_sqrt_value = np.sqrt(mse_value) 
 mse_sqrt_value_Array.append(mse_sqrt_value)