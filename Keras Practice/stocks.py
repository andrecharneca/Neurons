# Keras neural network environment test

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#load the dataset
dataset = loadtxt('stocks_data.txt', delimiter=',')

#split into input X and output Y

x = dataset[:,0:21] #columns 0->7
y = dataset[:,8] #column 8

#define Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) #1a layer, 12 nodes, ReLU activation functions
model.add(Dense(8, activation='relu')) #2a layer, 8 nodes
model.add(Dense(1, activation='sigmoid')) #3a layer, Sigmoid activation function for output in [0,1]

#compile Keras model, Cross Entropy loss function for binary classification, Adam algorithm for optimization
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit (train) the Keras model on the dataset
model.fit(x,y, epochs=150, batch_size=10)

#evaluate Keras model
_, accuracy = model.evaluate(x, y, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))

#make class predictions with the model
predictions = model.predict(x)
rounded = [np.round(p) for p in predictions]
#summarize first 5 cases
for i in range(5,10):
    print('%s -> %d (expected %d)' % (x[i].tolist(), rounded[i], y[i]))