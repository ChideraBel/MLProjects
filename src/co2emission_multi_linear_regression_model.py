import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

path = "../raw_data/FuelConsumptionCo2.csv"

df = pd.read_csv(path)

# view the data in the csv file
print(df.head())

select_features = df[['CO2EMISSIONS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]

# Create Train and test data
mask = np.random.rand(len(df)) < 0.8
train = select_features[mask]
test = select_features[~mask]

# Create model
regression = linear_model.LinearRegression()

# Use multiple independent variables for train data values
train_x = np.asanyarray(train[['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# Train model
regression.fit(train_x, train_y)

# sklearn used Ordinary Least Squares to find the best weights for each independent variables coef
print("Coefficients: ", regression.coef_)
print('')

test_x = np.asanyarray(test[['FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']])
true_y = np.asanyarray(test[['CO2EMISSIONS']])

# Model will predict y values
predicted_y = regression.predict(test_x)

# Get r^2 score
r2score = r2_score(true_y, predicted_y)
print("R^2 score: ", r2score)