import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
#%matplotlib inline

from sklearn.tree import DecisionTreeClassifier, plot_tree ## for training and ploting DT
from sklearn.metrics import roc_auc_score  ## for evaluation of the model using accuracy metric

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#lab 3
from tqdm import tqdm ## to show the progress bar of the training process
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from sklearn.neural_network import MLPClassifier

from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyclustering.cluster.kmedoids import kmedoids

# improve model libreries
import umap
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

#lab 3
from tqdm import tqdm ## to show the progress bar of the training process
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier

#--------------------------------------- read csv file:
table = pd.read_csv('table_after_feature_selection- 70-30 -Oversampling.csv')
X_data = table.drop(['fraudulent', 'job_id'], axis=1)
X_train = table.drop(['fraudulent','job_id'], axis=1)
y_train = table['fraudulent']
random_state = 10


##---------------------------------------------- Hyperparameter tuning & Evaluation Methods
### Evaluation Method - Holdout-set

## splitting the data to 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=10)


print(f"Train size: {X_train.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print("Train\n-----------\n", pd.value_counts(y_train)/y_train.shape[0])
print("\nTest\n-----------\n", pd.value_counts(y_test)/y_test.shape[0])

# ------------------------------------------------- 1. Decision Trees -----------------------------------------------------------------------------------


# ---------------------------------------------- Grid Search -------------------------------------------

# initializing a defaultive DT model
DecisionTreeClassifier()

# defining the grid that we want to tune the model according to.
# in this case we will tune max_depth, criterion, max_features as our hiperparameters
# we finally check the number of combinations, which will be the number of different DT models to train.
param_grid = {'max_depth': np.arange(1, 30, 1),
              'criterion': ['entropy', 'gini'],
              'max_features': ['sqrt', 'log2', None]# the method is on the number of features
             }

# Grid search will perform a 10-fold CV (cv=10)
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=10),
                            param_grid=param_grid,
                            scoring='roc_auc',
                            refit=True,
                            cv=10,
                            return_train_score=True)

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
#print(comb)
param_grid.values()

# fit the train set in the Grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score - validation set
best_model = grid_search.best_estimator_
print("Best model:", best_model)
print(grid_search.best_params_, '\n')

model = best_model

gridSearchCombination_DT = pd.DataFrame(grid_search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
print("The best model is: ", best_model)

# Train Score:
print("Train accuracy: ", round(max(gridSearchCombination_DT['mean_train_score']), 4))

# Validation Score:
print("Validation accuracy: ", round(max(gridSearchCombination_DT['mean_test_score']), 4))

# Test Score:
preds = best_model.predict(X_test)
print("Test accuracy: ", round(roc_auc_score(y_test, preds), 3))

#Best DT model Tree
best_model = DecisionTreeClassifier (criterion='entropy', max_depth=8, max_features=None)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
cm_DT = confusion_matrix(y_test, y_pred)
print("DT Confusion Matrix:")
print(cm_DT)

plt.figure(figsize=(20, 8))
plot_tree(best_model,
              filled=True,
              max_depth=3,
              class_names=True,
              feature_names=X_train.columns.values,
              fontsize=8)
plt.show()

# ------------------------------------------ feature importance's

feature_importances = pd.Series(best_model.feature_importances_)
feature_importances.index = [X_test.columns.values]
feature_importances = feature_importances.sort_values(ascending=False)
feature_importances.head(24).plot(kind='bar', figsize=(10, 8))
plt.title('Feature Importance', size=25)
plt.xlabel('Num Of Features')
plt.ylabel('Feature importance', size=20)
plt.show()



# ------------------------------------------------- 2.Artificial Neural Networks -----------------------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=10)


model = MLPClassifier(max_iter=1000)  # Adjust the hidden_layer_sizes as per your requirements
model.fit(X_train, y_train)


y_pred = model.predict(X_train)

accuracy = roc_auc_score(y_train, y_pred, )
print("default selection")
print(f"Training accuracy: {accuracy}")



y_pred = model.predict(X_val)

accuracy = roc_auc_score(y_val, y_pred, )
print("default selection")
print(f"Validation accuracy: {accuracy}")



# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


#
# Define the parameter grid for grid search
param_grid = {
    #'hidden_layer_sizes': [(i, j, k) for i in range(10, 60, 10) for j in range(10, 60, 10) for k in range(10, 30, 10)],
    #'hidden_layer_sizes': [(i) for i in range(50, 120, 10)],
    'hidden_layer_sizes': [(i, j) for i in range(10, 80, 10) for j in range(10, 80, 10)],
    'activation': ['relu', 'tanh','logistic','identity'],
    'learning_rate_init': [0.0001,0.0005, 0.001, 0.005],
    'alpha': [0.00005, 0.0001, 0.0005, 0.001, ],
    'batch_size': [64],
    'solver': ['adam'],
}

# Create the MLPClassifier model
model = MLPClassifier(max_iter=1000)

# Create the GridSearchCV object with scoring method set to 'roc_auc'
grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5, return_train_score=True)

# Fit the grid search on the training data
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Use the best model to make predictions
y_pred = best_model.predict(X_test)

# Calculate accuracy using roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred, )

# Print the best parameters and roc_auc score
print("Best Parameters:", best_params)
print("ROC AUC Score [test]:", roc_auc)

# Use the best model to make predictions
y_pred = best_model.predict(X_val)

# Calculate accuracy using roc_auc_score
roc_auc = roc_auc_score(y_val, y_pred)

# Print the best parameters and roc_auc score
print("Best Parameters:", best_params)
print("ROC AUC Score [validation]:", roc_auc)


# -----------------------------------------activation plot-----------------------------------------------------

# Create empty lists to store the activation values and accuracy scores
activation_values = []
training_scores = []
validation_scores = []

# Iterate over the parameter grid
for param_values in param_grid['activation']:
    # Create the MLPClassifier model with the current activation value
    model = MLPClassifier(activation=param_values, max_iter=1000)

    # Fit the model on the validation data
    model.fit(X_val, y_val)

    # Make predictions on the training and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate the accuracy scores for training and test sets
    training_accuracy = roc_auc_score(y_train, y_pred, )
    validation_accuracy = roc_auc_score(y_test, y_pred, )

    # Append the activation value and accuracy scores to the lists
    activation_values.append(param_values)
    training_scores.append(training_accuracy)
    validation_scores.append(validation_accuracy)

# Convert the lists to numpy arrays
activation_values = np.array(activation_values)
training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)

# Plot the accuracy levels for 'activation'
plt.plot(activation_values, training_scores, label='Training Accuracy')
plt.plot(activation_values, validation_scores, label='validation Accuracy')
plt.xlabel('Activation')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Activation Functions')
plt.legend()
plt.show()



# ---------------------------------- alpha plot --------------------------------------------------------

# Create empty lists to store the hyperparameter values and scores
hyperparameter_values = []
training_scores = []
test_scores = []

# Iterate over the parameter grid
for param_value in param_grid['alpha']:
    # Set the current hyperparameter value
    best_model.alpha = param_value

    # Fit the model and calculate scores
    best_model.fit(X_train, y_train)
    training_score = best_model.score(X_train, y_train)
    testing_score = best_model.score(X_test, y_test)

    # Append the values and scores to the lists
    hyperparameter_values.append(param_value)
    training_scores.append(training_score)
    test_scores.append(testing_score)

# Convert the lists to numpy arrays
hyperparameter_values = np.array(hyperparameter_values)
training_scores = np.array(training_scores)
test_scores = np.array(test_scores)

# Plot the accuracy levels
plt.plot(hyperparameter_values, training_scores, label='Training Accuracy')
plt.plot(hyperparameter_values, test_scores, label='Test Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Alpha Values')
plt.legend()
plt.show()

# ----------------------------------- hidden_layer_sizes plot---------------------------------------------------------
# Create empty lists to store the hidden_layer_sizes values and accuracy scores
hidden_layer_sizes_values = []
training_scores = []
validation_scores = []

# Iterate over the parameter grid
for param_values in param_grid['hidden_layer_sizes']:
    # Create the MLPClassifier model with the current hidden_layer_sizes value
    model = MLPClassifier(hidden_layer_sizes=param_values, max_iter=1000)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the training and test sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Calculate the accuracy scores for training and test sets
    training_accuracy = roc_auc_score(y_train, y_train_pred, )
    validation_accuracy = roc_auc_score(y_val, y_val_pred, )

    # Append the hidden_layer_sizes value and accuracy scores to the lists
    hidden_layer_sizes_values.append(param_values)
    training_scores.append(training_accuracy)
    validation_scores.append(validation_accuracy)

# Convert the lists to numpy arrays
hidden_layer_sizes_values = np.array(hidden_layer_sizes_values)
training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)

# Plot the accuracy levels for 'hidden_layer_sizes'
plt.plot(hidden_layer_sizes_values, training_scores, label='Training Accuracy')
plt.plot(hidden_layer_sizes_values, validation_scores, label='Validation Accuracy')
plt.xlabel('Hidden Layer Sizes')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Hidden Layer Sizes')
plt.legend()
plt.show()

# -------------------------------- learning_rate_init plot ---------------------------------------------------------
#
#
# Create empty lists to store the hyperparameter values and scores
hyperparameter_values = []
training_scores = []
validation_scores = []

# Iterate over the parameter grid
for param_value in param_grid['learning_rate_init']:
    # Set the current hyperparameter value
    best_model.learning_rate_init = param_value

    # Fit the model and calculate scores
    best_model.fit(X_train, y_train)
    training_score = best_model.score(X_train, y_train)
    validation_score = best_model.score(X_val, y_val)

    # Append the values and scores to the lists
    hyperparameter_values.append(param_value)
    training_scores.append(training_score)
    validation_scores.append(validation_score)

# Convert the lists to numpy arrays
hyperparameter_values = np.array(hyperparameter_values)
training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)

# Plot the accuracy levels
plt.plot(hyperparameter_values, training_scores, label='Training Accuracy')
plt.plot(hyperparameter_values, validation_scores, label='Validation Accuracy')
plt.xlabel('Learning Rate Init')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Learning Rate Init Values')
plt.legend()
plt.show()

# ------------------------------batch size plot---------------------------------------------------------------------------

# Create empty lists to store the hyperparameter values and scores
hyperparameter_values = []
training_scores = []
validation_scores = []

# Iterate over the parameter grid
for param_value in param_grid['batch_size']:
    # Set the current hyperparameter value
    best_model.batch_size = param_value

    # Fit the model and calculate scores
    best_model.fit(X_train, y_train)
    training_score = best_model.score(X_train, y_train)
    validation_score = best_model.score(X_val, y_val)

    # Append the values and scores to the lists
    hyperparameter_values.append(param_value)
    training_scores.append(training_score)
    validation_scores.append(validation_score)

# Convert the lists to numpy arrays
hyperparameter_values = np.array(hyperparameter_values)
training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)

# Plot the accuracy levels
plt.plot(hyperparameter_values, training_scores, label='Training Accuracy')
plt.plot(hyperparameter_values, validation_scores, label='Validation Accuracy')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Batch Size Values')
plt.legend()
plt.show()

# ------------------------------------- solver plot----------------------------- ----------------------------------
# Create empty lists to store the hyperparameter values and scores
hyperparameter_values = []
training_scores = []
validation_scores = []

# Iterate over the parameter grid
for param_value in param_grid['solver']:
    # Set the current hyperparameter value
    best_model.solver = param_value

    # Fit the model and calculate scores
    best_model.fit(X_train, y_train)
    training_score = best_model.score(X_train, y_train)
    validation_score = best_model.score(X_val, y_val)

    # Append the values and scores to the lists
    hyperparameter_values.append(param_value)
    training_scores.append(training_score)
    validation_scores.append(validation_score)

# Convert the lists to numpy arrays
hyperparameter_values = np.array(hyperparameter_values)
training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)

# Plot the accuracy levels
plt.plot(hyperparameter_values, training_scores, label='Training Accuracy')
plt.plot(hyperparameter_values, validation_scores, label='Validation Accuracy')
plt.xlabel('Solver')
plt.ylabel('Accuracy')
plt.title('Accuracy Levels for Different Solver Types')
plt.legend()
plt.show()

# ------------------------------------------------- 3.SVM -----------------------------------------------------------------------------------
# Create the LinearSVC model
SVM_model = LinearSVC(random_state=10)
#find best model - find the Hyperparameter - all features
param_grid = {'C': np.arange(1, 120, 6),
              'dual': [True, False],
              'tol': [1e-4, 1e-3, 1e-2],
             }

# Grid search will perform a 10-fold CV (cv=10)
grid_search = GridSearchCV(estimator=SVM_model,
                            param_grid=param_grid,
                            scoring='roc_auc',
                            refit=True,
                            cv=10,
                            return_train_score=True)

comb = 1
for list_ in param_grid.values():
    comb *= len(list_)
#print(comb)
param_grid.values()

# fit the train set in the Grid search
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score - validation set
best_model = grid_search.best_estimator_
print("Best model:", best_model)
print(grid_search.best_params_, '\n')

model = best_model

gridSearchCombination_DT = pd.DataFrame(grid_search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
print("The best model is: ", best_model)

# Train Score:
print("Train accuracy: ", round(max(gridSearchCombination_DT['mean_train_score']), 4))

# Validation Score:
print("Validation accuracy: ", round(max(gridSearchCombination_DT['mean_test_score']), 4))

# Test Score:
preds = best_model.predict(X_test)
print("Test accuracy: ", round(roc_auc_score(y_test, preds), 3))

# Best SVM model:
best_model = LinearSVC(C=13, dual=False, tol=0.01, random_state=10)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
cm_SVM = confusion_matrix(y_test, y_pred)
print("SVM Confusion Matrix:")
print(cm_SVM)


# --------------------------------use the best model for plotting:
best_model = SVM_model = LinearSVC(C=13, dual=False, tol=0.01, random_state=10)
X = X_train[['company_profile_ratio', 'has_company_logo']]
y = y_train
SVM_model.fit(X, y)

# Create a meshgrid to plot the decision boundary
h = 0.02  # step size in the mesh
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the labels for all points in the meshgrid
Z = SVM_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and the data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8)
scatter=plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)

plt.xlabel('company_profile_ratio', size=25)
plt.ylabel('has_company_logo', size=25)
plt.title('SVM Decision Boundary', size=25)
plt.show()

cm_SVM = confusion_matrix(y, y_pred=SVM_model.predict(X))
print("Confusion Matrix:")
print(cm_SVM)

model_coef= SVM_model.coef_
print(SVM_model.coef_)
print(SVM_model.intercept_)

# ------------------------------------------------- 4.Unsupervised Learning - Clustering -----------------------------------------------------------------------------------
x_data = table.drop(['fraudulent','job_id'], axis=1)
y_data = table['fraudulent']

# X_data_s = standard_scaler.fit_transform(x_data)
# standard_scaler.transform(x_data)
# ----------------------------------------PCA
pca = PCA(n_components=2)
pca.fit(x_data)

# Get the feature loadings
loadings = pca.components_

# Create a DataFrame to store the loadings
loadings_df = pd.DataFrame(loadings, columns=x_data.columns)

# Visualize the loadings
plt.figure(figsize=(10, 6))
plt.imshow(loadings_df, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(x_data.columns)), x_data.columns, rotation=90)
plt.yticks(range(len(loadings_df)), range(1, len(loadings_df)+1))
plt.xlabel('Original Features')
plt.ylabel('Principal Components')
plt.title('Feature Loadings')
plt.show()

# variance  explanation:
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())

train_pca = pca.transform(x_data)
train_pca = pd.DataFrame(train_pca, columns=['PC1', 'PC2'])
train_pca['fraudulent'] = table['fraudulent']
train_pca.head(10)

# show real scatter
sns.scatterplot(x='PC1', y='PC2', hue='fraudulent', data=train_pca)
plt.title("Data Scatter by PCA")
plt.show()


# best K=3 plot:

k_medoids = KMedoids(n_clusters=3, metric='euclidean', max_iter=100, method="pam", init="heuristic", random_state=10)
k_medoids.fit(x_data)


train_pca['cluster'] = pd.DataFrame(k_medoids.predict(x_data),index=y_data.index)

sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=train_pca, palette='Accent')
plt.scatter(pca.transform(k_medoids.cluster_centers_)[:, 0], pca.transform(k_medoids.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.title("Clustering Scatter")
plt.show()

# train on train_pca
print("train on train_pca")
train_pca['cluster'] = pd.DataFrame(k_medoids.predict(train_pca),index=y_data.index)

sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=train_pca, palette='Accent')
plt.scatter(pca.transform(k_medoids.cluster_centers_)[:, 0], pca.transform(KMedoids.cluster_centers_)[:, 1], marker='+', s=100 ,color='red')
plt.title("Clustering Scatter")
plt.show()


# ---------------------------------------------- find the best number of clusters
iner_list = []
dbi_list = []
sil_list = []

for n_clusters in tqdm(range(2, 10, 1)):
    k_medoids = KMedoids(n_clusters=n_clusters, max_iter=100, method="pam", init="heuristic", random_state=10)
    k_medoids.fit(x_data)
    assignment = k_medoids.predict(x_data)

    iner = k_medoids.inertia_
    sil = silhouette_score(x_data, assignment)
    dbi = davies_bouldin_score(x_data, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)


plt.plot(range(2, 10, 1), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.show()

plt.plot(range(2, 10, 1), dbi_list, marker='o')
plt.title("Davies-bouldin")
plt.xlabel("Number of clusters")
plt.show()



# ------------------------------------------------- Improvements -----------------------------------------------------------------------------------

import umap

# data improvement only
umap_obj = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.9)

# Perform UMAP dimensionality reduction on training data
X_train_embedded = umap_obj.fit_transform(X_train)

bestModel = MLPClassifier(hidden_layer_sizes=(30, 70), solver='adam', alpha=0.005,   activation='relu', batch_size=64, learning_rate_init=0.0005, max_iter=1000)  # Adjust the hidden_layer_sizes as per your requirements
bestModel.fit(X_train, y_train)

y_pred = bestModel.predict(X_test)

accuracy = roc_auc_score(y_test, y_pred)
print(f"Test accuracy: {accuracy}")

# model improvement only
bestModel = MLPClassifier(hidden_layer_sizes=(30, 70), solver='adam', alpha=0.005,   activation='relu', batch_size=64, learning_rate_init=0.005, max_iter=1000)  # Adjust the hidden_layer_sizes as per your requirements

# Define the learning rate schedule
def learning_rate_schedule(epoch, initial_learning_rate, decay_factor, decay_epochs):
    return initial_learning_rate * decay_factor**(epoch // decay_epochs)

initial_learning_rate = 0.005
decay_factor = 0.8
decay_epochs = 64
batch_size = 64

for epoch in range(int(X_train.size/batch_size)+1):
    # Update the learning rate
    learning_rate = learning_rate_schedule(epoch, initial_learning_rate, decay_factor, decay_epochs)
    bestModel.learning_rate_init = learning_rate



    bestModel.partial_fit(X_train, y_train, classes=np.unique(y_train))
    print(f"Epoch {epoch+1} - Test roc_auc_score: {bestModel}")

y_pred = bestModel.predict(X_test)
accuracy = roc_auc_score(y_test, y_pred, )
print(f"Test roc_auc_score: {accuracy}")

# both model and data improvements
umap_obj = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.9)

# Perform UMAP dimensionality reduction on training data
X_train_embedded = umap_obj.fit_transform(X_train)

bestModel = MLPClassifier(hidden_layer_sizes=(30, 70), solver='adam', alpha=0.005,   activation='relu', batch_size=64, learning_rate_init=0.005, max_iter=1000)  # Adjust the hidden_layer_sizes as per your requirements

# Define the learning rate schedule
def learning_rate_schedule(epoch, initial_learning_rate, decay_factor, decay_epochs):
    return initial_learning_rate * decay_factor**(epoch // decay_epochs)

initial_learning_rate = 0.005
decay_factor = 0.8
decay_epochs = 64
batch_size = 64

for epoch in range(int(X_train.size/batch_size)+1):
    # Update the learning rate
    learning_rate = learning_rate_schedule(epoch, initial_learning_rate, decay_factor, decay_epochs)
    bestModel.learning_rate_init = learning_rate



    bestModel.partial_fit(X_train, y_train, classes=np.unique(y_train))
    print(f"Epoch {epoch+1} - Test roc_auc_score: {bestModel}")

y_pred = bestModel.predict(X_test)
accuracy = roc_auc_score(y_test, y_pred, )
print(f"Test roc_auc_score: {accuracy}")

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

