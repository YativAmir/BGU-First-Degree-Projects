import numpy as np
import pandas as pd

file_path = "/Users/doron/OneDrive/שולחן העבודה/פרויקט גמר/predictions4.xlsx"
df = pd.read_excel(file_path, header=None)
def rmsle(y_true, y_pred):

    print(y_true)
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true) +1
    y_pred = np.array(y_pred) +1
    print(y_true)

    # Compute the log of the true and predicted values plus one
    log_true = np.log(y_true)
    log_pred = np.log(y_pred)

    # # Compute the mean squared error
    # msle = np.mean((log_true - log_pred) ** 2)

    sum_squared_log_diff = 0.0


    # Loop through each element to compute the squared difference
    for i in range(len(y_true)):
        sum_squared_log_diff += (log_true[i] - log_pred[i]) ** 2

    # Calculate the mean of the squared logarithmic differences
    msle = sum_squared_log_diff / len(y_true)

    # Return the square root of the mean squared error
    return np.sqrt(msle)

# Example usage
y_pred = df.iloc[:, 0]
y_true = df.iloc[:, 1]


rmsle_value = rmsle(y_true, y_pred)
print(rmsle_value)
