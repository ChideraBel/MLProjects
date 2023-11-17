import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import sklearn.tree as tree

my_data = pd.read_csv('../raw_data/drug200.csv')
# print(my_data.head())

# Define a mapping for categorical variables
sex_mapping = {'F': 0, 'M': 1}
bp_mapping = {'LOW': 0, 'NORMAL': 1, 'HIGH': 2}
cholesterol_mapping = {'NORMAL': 0, 'HIGH': 1}

# Apply the label encoding to the columns
my_data['Sex'] = my_data['Sex'].map(sex_mapping)
my_data['BP'] = my_data['BP'].map(bp_mapping)
my_data['Cholesterol'] = my_data['Cholesterol'].map(cholesterol_mapping)

# Sklearn decision trees need numerical values so converted categorical values to numerical values
X_data = np.asanyarray(my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']])
# print(X_data[0:5])

y_data = np.asanyarray(my_data['Drug'])
# print(y_data[0:5])

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=4)

# Creating the tree classifier object
drug_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4)

# Train the model
drug_tree.fit(X_train, y_train)

y_predicted = drug_tree.predict(X_test)

print("Y test: \n", y_test[0:5])
print("Predicted y values: \n", y_predicted[0:5])

# Evaluate accuracy of the model

print("Accuracy score: ", accuracy_score(y_test, y_predicted))

# Plot the decision tree
tree.plot_tree(drug_tree)
plt.show()
