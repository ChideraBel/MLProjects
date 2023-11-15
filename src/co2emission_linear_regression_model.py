import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

path = "../../IbmMLProjects/raw_data/FuelConsumptionCo2.csv"

df = pd.read_csv(path)

# view the data in the csv file
print(df.head())

# summarise the data, with means and counts

# print(df.describe())

# select the features to explore more, select the columns

select_features = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# print(select_features.head(9))  view the first 9 rows

# plot each column data as a histogram, (remove commenting to view histogram)
histogram_data = select_features[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
# histogram_data.hist()
# plt.show()

# plot the co2 emissions by the fuelConsumption_comb
# plt.scatter(select_features.CO2EMISSIONS, select_features.FUELCONSUMPTION_COMB)
# plt.xlabel("CO2EMISSIONS")
# plt.ylabel("FUELCONSUMPTION_COMB")
# plt.show()

# plt.scatter(select_features.ENGINESIZE, select_features.CO2EMISSIONS)
# plt.xlabel("ENGINESIZE")
# plt.ylabel("CO2EMISSIONS")
# plt.show()

# Creating train and test data set with train/test split using random.rand generator to create a mask.
mask = np.random.rand(len(df)) < 0.8
train = select_features[mask]
test = select_features[~mask]

# Create object for the linear regression model from scikit and train it with the train values of x and y
regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Train the model
regression.fit(train_x, train_y)

# The co-efficients and intercepts of the model(the line of best fit)
print('Coefficients: ', regression.coef_)
print('Intercept: ', regression.intercept_)
print('\n')

# Evaluate the accuracy of the model using the R^2 score, this is 1 - RMSE(root mean squared error)
# [scikit has as function that does that for you]
test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# Get the values that model predicts for the y values using the test_x
predicted_y = regression.predict(test_x)

# Calculate Mean Absolute Error (MAE) the average of the sum of each data points error
mae = np.mean(np.absolute(predicted_y - test_y))
print("Mean Absolute Error: ", mae)

# Calculate Mean Squared Error (MSE) the average ot the sum of the squares of each data points err
mse = np.mean(np.square(predicted_y - test_y))
print("Mean Squared Error: ", mse)

# Calculate the r^2 score using scikitlearn
r2score = r2_score(test_y, predicted_y)
print("R^2 score: ", r2score)

# Plot the line of best fit using the equation y = mx + c "'-r'" just means draw straight line(-) that is red(r)
plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS)
plt.xlabel("Fuel Consumption")
plt.ylabel("Emissions")
plt.plot(train_x, regression.coef_[0][0]*train_x + regression.intercept_[0], '-r')
plt.show()
