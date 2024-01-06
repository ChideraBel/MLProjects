import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# this model is to be used to predict and classify handwritten digits
# import the data
from keras.datasets import mnist

# read the data, the Keras library conveniently includes the MNIST dataset as part of its API
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)

# display one of the training datapoints (a handwritten digit character)
plt.imshow(X_train[0])

# flatten images into one-dimensional vector

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes) #number of classes are the numer of digits from 0-9

# Building the neural network
# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training and testing the model, build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1])) #prints an accuracy of 97.7%

model.save('classification_model.h5') #save the model so that it can be used again when needed
