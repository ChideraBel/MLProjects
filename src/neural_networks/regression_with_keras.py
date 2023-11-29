import pandas as pd
import numpy as np
from sklearn import preprocessing

path = '../../raw_data/concrete_data.csv'
concrete_data = pd.read_csv(path)

#check for null values and clean if there is
print(concrete_data.isnull().sum())

#Split data into predictors and target
concrete_data_columns = concrete_data.columns 

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

#Normalize the predictors values
predictors_norm = preprocessing.StandardScaler().fit_transform(predictors.astype(float))
print(predictors_norm[:5])

n_cols = predictors_norm.shape[1]

def regression_model():
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=n_cols)) #Add layer with 50 nodes and use the RLU activation function, specify the input shape for the first hidden layer
    model.add(Dense(50, activation='relu')) #Add another hidden layer with 50 nodes
    model.add(Dense(1)) #Add the output layer with one node
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

my_model = regression_model()