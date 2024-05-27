import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

# Load the CSV data into a Pandas DataFrame
gold_data = pd.read_csv(r"C:\Users\ADMIN\Downloads\gld_price_data.csv")

# Print first 5 rows in the dataframe
print(gold_data.head())

# Number of rows and columns
print(gold_data.shape)

# Getting some basic information about the data
print(gold_data.info())

# Checking the number of missing values
print(gold_data.isnull().sum())

# Getting the statistical measures of the data
print(gold_data.describe())

# Calculating the correlation matrix
correlation = gold_data.corr()

# Constructing a heatmap to understand the correlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()

# Correlation values of GLD
print(correlation['GLD'])

# Checking the distribution of the GLD Price
sns.distplot(gold_data['GLD'], color='green')
plt.show()

# Splitting the data into features and target
X = gold_data.drop(['Date','GLD'], axis=1)
Y = gold_data['GLD']

print(X)
print(Y)

# Splitting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Creating the model
regressor = RandomForestRegressor(n_estimators=100)

# Training the model
regressor.fit(X_train, Y_train)

# Making predictions on the test data
test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)

# Calculating the R squared error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared error : ", error_score)

# Plotting the actual prices vs predicted prices
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label='Actual Value')
plt.plot(test_data_prediction, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()

# Saving the trained model to a pickle file
with open('gold_price_model.pkl', 'wb') as file:
    pickle.dump(regressor, file)
