# libraries
import numpy as np
import pandas as pd
# imports
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

np.random.seed(200)

# Generate a synthetic regression dataset
file_path = "C:/Users/yativ/OneDrive/Desktop/ANN_data.csv"
df = pd.read_csv(file_path)
df = df.iloc[:, 1:]

# Shuffle the DataFrame rows
df_shuffled = df.sample(frac=1).reset_index(drop=True)

df_sale_price = df_shuffled[['SalePrice']]
df_without_sale_price = df_shuffled.drop(columns=['SalePrice'])

X_train = df_without_sale_price.iloc[:-200]
y_train = df_sale_price.iloc[:-200]

X_test = df_without_sale_price.iloc[-200:]
y_test = df_sale_price.iloc[-200:]


# Define the fully connected ANN regression network
mlp = MLPRegressor(
    hidden_layer_sizes=(10, 10, 10, 10, 10, 10, 10, 10, 10, 10 ),  # 4 layers with 100 neurons each
    #                    1   2   3   4   5   6   7   8   9   10
    activation='relu',  # ReLU activation function
    solver='lbfgs',  # BFGS optimizer
    max_iter=100,  # Maximum number of iterations
)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mse_sqrt_value = np.sqrt(mse) 
print(f"Mean Squared Error: {mse_sqrt_value}")