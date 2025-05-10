#Objective: This notebook aims to make a predictive model using one training dataset. The model will predict values for a second, test dataset. Both datasets are artificially created and share the same data model. We'll gauge success using mean squared error (MSE).

#End-to-end modeling is completed as well too throughout the project 

# Importing necessary libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# Setting random seed for reproducibility
random.seed(42)

#Exploratory data analysis

#Read in the data 
train_data = pd.read_csv('codetest_train.txt', delimiter='\t')
test_data = pd.read_csv('codetest_test.txt', delimiter='\t')

# Display first few rows of training data
print(train_data.head())
print(test_data.head())

#Data Exploration - only look at the training data 

## Summary statistics of the training data
print(train_data.describe())

# Visualizing the distribution of the target variable with histogram
plt.figure(figsize=(10, 6))
sns.histplot(train_data['target'], bins=50, kde=True)
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.show()

#Data Preparation - changing data types 

non_numeric_cols = [col for col, dtype in train_data.dtypes.items() if dtype not in ['float64', 'int64']]

if non_numeric_cols:
    print(f"Columns with non-numeric data types: {non_numeric_cols}")
else:
    print("All columns have float or integer data types.")

train_data['f_61'].head(), train_data['f_121'].head(), train_data['f_215'].head(), train_data['f_237'].head(), 

#Encoding

non_numeric_features = train_data.dtypes[train_data.dtypes == 'object'].index

# Encoding the non-numeric features
encoder = LabelEncoder()
for feature in non_numeric_features:
    train_data[feature] = encoder.fit_transform(train_data[feature].astype(str))

# Confirming that all features are now numeric
train_data.dtypes[train_data.dtypes != 'float64']

train_data['f_61'].head(), train_data['f_121'].head(), train_data['f_215'].head(), train_data['f_237'].head(), 

non_numeric_cols = [col for col, dtype in test_data.dtypes.items() if dtype not in ['float64', 'int64']]

if non_numeric_cols:
    print(f"Columns with non-numeric data types: {non_numeric_cols}")
else:
    print("All columns have float or integer data types.")

non_numeric_features_test = test_data.dtypes[test_data.dtypes == 'object'].index

from sklearn.preprocessing import LabelEncoder

# Encoding the non-numeric features
encoder = LabelEncoder()
for feature in non_numeric_features_test:
    test_data[feature] = encoder.fit_transform(test_data[feature].astype(str))

# Confirming that all features are now numeric
print(test_data.dtypes[test_data.dtypes != 'float64'])

#Missing Values

# Check missing values in train_data
missing_train = train_data.isna().sum()
print("Missing values in train_data:")
print(missing_train[missing_train > 0])

# Check missing values in test_data
missing_test = test_data.isna().sum()
print("\nMissing values in test_data:")
print(missing_test[missing_test > 0])


# Filling missing values in train_data with median
for column in train_data.columns:
    median_value = train_data[column].median()
    train_data[column].fillna(median_value, inplace=True)

# Filling missing values in test_data with median
for column in test_data.columns:
    median_value = test_data[column].median()
    test_data[column].fillna(median_value, inplace=True)

# Check missing values in train_data
missing_train = train_data.isna().sum()
print("Missing values in train_data:")
print(missing_train[missing_train > 0])

# Check missing values in test_data
missing_test = test_data.isna().sum()
print("\nMissing values in test_data:")
print(missing_test[missing_test > 0])

#all missing values have been removed 

#Modeling - Linear Regression 

# Separating features and target variable from training data
X = train_data.drop('target', axis=1)
y = train_data['target']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data)

# Initializing the model
lr_model = LinearRegression()


# Fitting the model to the training data
lr_model.fit(X_train_scaled, y_train)

# Making predictions on the validation set
y_val_pred = lr_model.predict(X_val_scaled)

# Calculating the mean squared error
mse = mean_squared_error(y_val, y_val_pred)
print(mse)

#Random Forest Regressor

# Why is Scaling Not Required? - because they are tree based models 

# Initializing the model
rf_model = RandomForestRegressor(random_state=42)

# Fitting the model to the training data
rf_model.fit(X_train_scaled, y_train)

# Making predictions on the validation set
y_val_pred_rf = rf_model.predict(X_val_scaled)

# Calculating the mean squared error for Random Forest
mse_rf = mean_squared_error(y_val, y_val_pred_rf)
print(mse_rf)

#Comparing Multiple Regression Models
# Initializing the models
lasso_model = Lasso(random_state=42)
ridge_model = Ridge(random_state=42)

# Fitting the models to the training data
lasso_model.fit(X_train_scaled, y_train)
ridge_model.fit(X_train_scaled, y_train)

# Making predictions on the validation set
#Validation set is used to prevent overfitting and ensure the model generalizes well to new, unseen data
y_val_pred_lasso = lasso_model.predict(X_val_scaled)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)

# Calculating the MSE and RMSE for all models
mse_lasso = mean_squared_error(y_val, y_val_pred_lasso)
mse_ridge = mean_squared_error(y_val, y_val_pred_ridge)

rmse_lr = np.sqrt(mse)
rmse_rf = np.sqrt(mse_rf)
rmse_lasso = np.sqrt(mse_lasso)
rmse_ridge = np.sqrt(mse_ridge)

mse_values_all = [mse, mse_rf, mse_lasso, mse_ridge]
rmse_values_all = [rmse_lr, rmse_rf, rmse_lasso, rmse_ridge]

print(mse_values_all, rmse_values_all)

#Comparing RMSE and R-Squared Among Models

# Your color palette
color1 = '#98C5C0'  # Light teal
color2 = '#A880DC'  # Light purple

models_all = ['Linear Regression', 'Random Forest', 'Lasso', 'Ridge']

# Calculating R-squared for all models
r2_lr = lr_model.score(X_val_scaled, y_val)
r2_rf = rf_model.score(X_val_scaled, y_val)
r2_lasso = lasso_model.score(X_val_scaled, y_val)
r2_ridge = ridge_model.score(X_val_scaled, y_val)

r2_values_all = [r2_lr, r2_rf, r2_lasso, r2_ridge]

# Creating the bar plot for RMSE and R-squared
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

ax[0].barh(models_all, rmse_values_all, color=color1)
ax[0].set_xlabel('Root Mean Squared Error (RMSE)')
ax[0].set_title('Comparison of RMSE among Four Models')

ax[1].barh(models_all, r2_values_all, color=color2)
ax[1].set_xlabel('R-Squared ($R^2$)')
ax[1].set_title('Comparison of $R^2$ among Four Models')

plt.tight_layout()
plt.show()