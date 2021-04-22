#Loading Keras model from json file test

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np

#read and create model from json file
json_file = open('stocks_model_ndays2_e500_b10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights into model
loaded_model.load_weights("stocks_model_ndays2_e500_b10.h5")
print("Loaded model from disk")

#evaluate model with data
ndays=2
dataset = np.loadtxt("stocks_data_processed_activision.txt", delimiter=',')
x = dataset[:,0:2*ndays] #columns 0->19
y = dataset[:,2*ndays] #column 20

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(x,y,verbose=0)
predictions = loaded_model.predict(x)
rounded = [np.round(p) for p in predictions]
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

for i in range(100,150):
    print('Sample %d -> %d (expected %d)' % (i, rounded[i], y[i]))
print("No. samples: %d" % (len(y)))