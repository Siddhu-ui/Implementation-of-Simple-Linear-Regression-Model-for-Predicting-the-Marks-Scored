# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# Developed by: SIDDHARTH S
# RegisterNumber: 212224040317

## AIM: To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries and read the dataset containing hours studied and scores.

2. Split the data into features (X) and target (Y), then into training and testing sets.

3. Train a Linear Regression model using the training data.

4. Predict scores for the test set using the trained model.

5. Visualize results with scatter plots and the regression line.

6.Evaluate model performance using MSE, MAE, and RMSE metrics.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_scores.csv")

# Display the first and last few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

# Separate the independent (X) and dependent (Y) variables
X = df.iloc[:, :-1].values  # 'Hours' column
Y = df.iloc[:, -1].values   # 'Scores' column

# Split the dataset into training and testing sets (1/3 for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test set results
Y_pred = regressor.predict(X_test)

# Display predicted and actual values
print("\nPredicted values:", Y_pred)
print("Actual values:", Y_test)

# Plot the Training set results
plt.scatter(X_train, Y_train, color="red", label="Actual Scores (Train)")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Plot the Testing set results
plt.scatter(X_test, Y_test, color='green', label="Actual Scores (Test)")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print error metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('\n--- Error Metrics ---')
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_scores.csv")

# Display the first and last few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

# Separate the independent (X) and dependent (Y) variables
X = df.iloc[:, :-1].values  # 'Hours' column
Y = df.iloc[:, -1].values   # 'Scores' column

# Split the dataset into training and testing sets (1/3 for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test set results
Y_pred = regressor.predict(X_test)

# Display predicted and actual values
print("\nPredicted values:", Y_pred)
print("Actual values:", Y_test)

# Plot the Training set results
plt.scatter(X_train, Y_train, color="red", label="Actual Scores (Train)")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Plot the Testing set results
plt.scatter(X_test, Y_test, color='green', label="Actual Scores (Test)")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print error metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('\n--- Error Metrics ---')
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)


*/
```

## Output:

![alt text](<Screenshot 2025-04-12 203017.png>)
![alt text](<Screenshot 2025-04-12 203047.png>)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
