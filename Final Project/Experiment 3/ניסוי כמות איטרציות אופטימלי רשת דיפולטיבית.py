# libraries
import numpy as np
import pandas as pd
# imports
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
# Array
mse_sqrt_value_Array = []

for j in range(0, 10):
    
    np.random.seed(j)

    # Generate a synthetic regression dataset
    file_path = "/Users/doron/OneDrive/שולחן העבודה/פרויקט גמר/train_ready_for_ANN.csv"
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
    
    for i in range(50, 501, 50):
        # Define the fully connected ANN regression network
        mlp = MLPRegressor(
            hidden_layer_sizes=(10, 10, 10, 10),  # 4 layers with 100 neurons each
            activation='relu',  # ReLU activation function
            solver='lbfgs',  # BFGS optimizer
            max_iter=i,  # Maximum number of iterations
        )

        # Train the model
        mlp.fit(X_train, y_train)

        # Make predictions
        y_pred = mlp.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mse_sqrt_value = np.sqrt(mse) 
        print(f"Mean Squared Error: {mse_sqrt_value}")
        mse_sqrt_value_Array.append(mse_sqrt_value)

  
