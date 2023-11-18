import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, classification_report, log_loss

data_frame = pd.read_csv('../raw_data/ChurnData.csv')

data_frame = data_frame[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless', 'churn']]
data_frame['churn'] = data_frame['churn'].astype('int')

X_data = np.asanyarray(data_frame[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']])
y_data = np.asanyarray(data_frame['churn'])

# Normalize the data
X_data_normalized = preprocessing.StandardScaler().fit_transform(X_data.astype(float))
print(X_data_normalized[0:5])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_data_normalized, y_data, test_size=0.2, random_state=2)

# Create logistic regression object
regression = LogisticRegression(C=0.01, solver='liblinear')
regression.fit(X_train, y_train)

predicted_y = regression.predict(X_test)

# Logistic regression uses probability of the value being 1 to determine the values
probability_y = regression.predict_proba(X_test)

# Use Jaccard index to evaluate model accuracy, pos_label is used to define which value is positive.
# For churn a value of 0 is positive, because customers not leaving is positive for the company
print("Jaccard score of model: ", jaccard_score(y_test, predicted_y, pos_label=0))

# Print the classification report
print("Classification Report: \n", classification_report(y_test, predicted_y))

# Print the log loss
print("Log loss: ", log_loss(y_test, probability_y))
