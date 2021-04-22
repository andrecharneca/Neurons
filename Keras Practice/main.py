# Keras neural network environment test

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time

#training data company name
name = 'apple'
#number of days per row
ndays = 10

#load the dataset, using AAPL data from 2013-2018
dataset = loadtxt('stocks_data_processed_' + name + '.txt', delimiter=',')

#split into input X and output Y, ndays is number of days per row

x = dataset[:,0:2*ndays] #columns 0->ndays-1
y = dataset[:,2*ndays] #column ndays

accuracy_mean = 0
n = 50

#measure time in seconds
start = time.time()
for i in range(n):
    #define Keras model
    model = Sequential()
    model.add(Dense(20, input_dim=2*ndays, activation='relu')) #1a layer, 12 nodes, ReLU activation functions
    model.add(Dense(12, activation='relu')) #2a layer, 8 nodes
    model.add(Dense(1, activation='sigmoid')) #3a layer, Sigmoid activation function for output in [0,1]

    #compile Keras model, Cross Entropy loss function for binary classification, Adam algorithm for optimization
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #fit (train) the Keras model on the dataset
    model.fit(x,y, epochs=500, batch_size=10, verbose=0)

    #evaluate Keras model
    _, accuracy = model.evaluate(x, y, verbose=0)
    print('Accuracy (it. %d): %.2f' % (i,accuracy*100))
    accuracy_mean += accuracy

    """#save model to file if acc >98
    if accuracy>=0.98:
        #save model to json file
        model_json = model.to_json()
        with open("stocks_model_ndays2_e500_b10.json", "w") as json_file:
            json_file.write(model_json)

        #save weights to h5 file
        model.save_weights("stocks_model_ndays2_e500_b10.h5")
        print("Model saved to json file.")
        break"""


#end timer
end = time.time()
elapsed_time = end-start #in seconds
accuracy_mean = accuracy_mean/n
print(('Accuracy mean: %.2f\nTime: %d seconds' % (accuracy_mean*100, elapsed_time)))

#make class predictions with the model
predictions = model.predict(x)
rounded = [np.round(p) for p in predictions]
#summarize first 5 cases
for i in range(100,120):
    print('%s -> %d (expected %d)' % (x[i].tolist(), rounded[i], y[i]))