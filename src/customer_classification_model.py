import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

data = pd.read_csv('../raw_data/teleCust1000t.csv')
print(data.head())

# Check how many customer categories there are

cat_count = data['custcat'].value_counts()

# print("Category counts ", cat_count)

# Convert from pandas to numpy array
X_data = np.asanyarray(data[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']])
y = np.asanyarray(data[['custcat']])

# Normalize the data
X_normalized = preprocessing.StandardScaler().fit_transform(X_data.astype(float))

print(X_normalized)

# Train test split using sklearn function instead of rand mask
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=4)

# Start with k = 4, going to determine which value K gives more accuracy
k = 4

# Train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train.flatten())
predicted_y = neigh.predict(X_test)

# Evaluate accuracy using Jaccard index
test_score = metrics.accuracy_score(y_test, predicted_y)
print("Test score: ", test_score)
train_score = metrics.accuracy_score(y_train, neigh.predict(X_train))
print("Train score: ", train_score)

Ks = 10
best_k = -1
max_score = 0
mean_acc = np.zeros((Ks-1))

for n in range(1, Ks):

    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train.flatten())
    predicted_y = neigh.predict(X_test)
    test_score = metrics.accuracy_score(y_test, predicted_y)
    mean_acc[n-1] = test_score

    if test_score > max_score:
        max_score = test_score
        best_k = n

print(mean_acc)
print("The best k number of neighbors: ", best_k)

# Model is sensitive to test-split randomness, larger sets of data could help reduce this sensitivity
final_neigh = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train.flatten)
