import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# load dataset
dataframe = pandas.read_csv("../Data/iris.txt", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
one_hot_y = np_utils.to_categorical(encoded_Y)

file1 = open(r"../Data/iris_one_hot.txt", "w")

for i in range(len(one_hot_y)):
    file1.write(",".join( repr(e) for e in X [i]) + "," + ",".join( repr(e) for e in one_hot_y [i]) + "\n") #removes the brackets in array representation
file1.close()