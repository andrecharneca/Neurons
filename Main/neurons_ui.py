import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import subprocess
import qdarkstyle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from numpy import loadtxt
import numpy as np
import time as time
from qt_material import apply_stylesheet
import os
import threading
## Console output reading function ##
"""def run(cmd):
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr"""

## Popup Window ##

class NewModelPopupWindow(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        """mainLayout = QGridLayout(widget)
        self.setLayout(mainLayout)"""

        self.name = 'new_model'

        vbox_main = QVBoxLayout(self)
        hbox_newModel = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        label_newModel = QLabel(self)
        label_newModel.setText("Model name:")
        """label_inputNumber = QLabel(self)
        label_inputNumber.setText("No. of input columns:")
        label_outputNumber = QLabel(self)
        label_outputNumber.setText("No. of output columns:")"""
        label_warning = QLabel("Creating new model for selected training file.\n\n - NOTE: The number of input/ouput columns is unalterable after creating model.\n",self)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setText('new_model')
        self.lineEdit.textChanged.connect(self.change_path)
        self.lineEdit.returnPressed.connect(self.create_model)

        pushButton_createModel = QPushButton("Create model", self)
        pushButton_createModel.clicked.connect(self.create_model)

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.pressed.connect(self.close)

        hbox_newModel.addWidget(label_newModel)
        hbox_newModel.addWidget(self.lineEdit)
        hbox_buttons.addWidget(pushButton_createModel)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox_main.addWidget(label_warning)
        vbox_main.addLayout(hbox_newModel)
        vbox_main.addLayout(hbox_buttons)
        #mainLayout.addLayout(vbox_main,0,0,0,0)

    def change_path(self):
        self.name = self.lineEdit.text()

    def create_model(self):
        """create new model"""
        global model_name
        global model
        global hidden_layers
        global output_layer_settingsDict
        global label_modelName

        """INSERT CHECK FOR VALID NAME"""
        """INSERT WARNING IF UNSAVED MODEL"""

        tf.keras.backend.clear_session()

        #Initialize new model
        hidden_layers = []
        model_name = self.lineEdit.text()
        model = Sequential()
        inputDim = inputCol_end - inputCol_start + 1
        outputDim = outputCol_end - outputCol_start + 1

        model.add(Dense(inputDim, input_dim=inputDim, activation='relu', name = 'input'))
        model.add(Dense(12, activation = 'relu', name = 'Layer_1')) ###?

        #Change output settings for Delete and Add buttons
        model.add(Dense(outputDim, activation ='relu', name = 'output')) #name = 'dense' by default, then 'dense_1', etc
        output_layer_settingsDict["units"] = outputDim
        output_layer_settingsDict["activation"] = "ReLu"

        hidden_layers.append(model.layers[1])
        update_hiddenLayersList()

        #update modelName label
        label_modelName.setText("Model: " + model_name)
        self.close()

class EditInputLayerPopupWindow(QWidget):
    """Edit input layer popup window"""
    def __init__(self):
        QWidget.__init__(self)
        """mainLayout = QGridLayout(widget)
        self.setLayout(mainLayout)"""

        vbox = QVBoxLayout(self)
        hbox_activation = QHBoxLayout(self)
        hbox_neurons = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        self.activationFunction = QComboBox(self)
        self.activationFunction.addItems(activationFunctionList)
        self.activationLabel = QLabel("Activation Function:",self)

        self.neurons = QSpinBox(self)
        self.neurons.setValue(10)
        self.neurons.setMinimum(0)
        self.neuronsLabel = QLabel("Neurons:",self)


        self.label = QLabel("Edit Input layer properties.")
        self.label.setAlignment(Qt.AlignCenter)

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.clicked.connect(self.close)

        pushButton_saveEdit = QPushButton("Save Edit", self)
        pushButton_saveEdit.clicked.connect(self.make_changes)

        hbox_activation.addWidget(self.activationLabel)
        hbox_activation.addWidget(self.activationFunction)
        hbox_neurons.addWidget(self.neuronsLabel)
        hbox_neurons.addWidget(self.neurons)
        hbox_buttons.addWidget(pushButton_saveEdit)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox.addWidget(self.label)
        vbox.addLayout(hbox_neurons)
        vbox.addLayout(hbox_activation)
        vbox.addLayout(hbox_buttons)

        #mainLayout.addLayout(vbox,0,0,0,0)

    def make_changes(self):
        global model
        layer = model.get_layer(name = "input")
        layer.activation = activationFunctionDict[self.activationFunction.currentText()]
        layer.units = self.neurons.value()
        print(layer.get_config()) ###

        self.close()

class EditOutputLayerPopupWindow(QWidget):
    """Edit output layer popup window"""
    def __init__(self):
        QWidget.__init__(self)
        """mainLayout = QGridLayout(widget)
        self.setLayout(mainLayout)"""

        vbox = QVBoxLayout(self)
        hbox_activation = QHBoxLayout(self)
        hbox_neurons = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        self.activationFunction = QComboBox(self)
        self.activationFunction.addItems(activationFunctionList)
        self.activationLabel = QLabel("Activation Function:",self)

        self.neuronsLabel = QLabel("Note: Number of neurons in output layer is the output shape.")

        self.label = QLabel("Edit Output layer properties.")
        self.label.setAlignment(Qt.AlignCenter)

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.clicked.connect(self.close)

        pushButton_saveEdit = QPushButton("Save Edit", self)
        pushButton_saveEdit.clicked.connect(self.make_changes)

        hbox_activation.addWidget(self.activationLabel)
        hbox_activation.addWidget(self.activationFunction)
        hbox_neurons.addWidget(self.neuronsLabel)
        hbox_buttons.addWidget(pushButton_saveEdit)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox.addWidget(self.label)
        vbox.addLayout(hbox_neurons)
        vbox.addLayout(hbox_activation)
        vbox.addLayout(hbox_buttons)

        #mainLayout.addLayout(vbox,0,0,0,0)

    def make_changes(self):
        global model
        layer = model.get_layer(name = "output")
        layer.activation = activationFunctionDict[self.activationFunction.currentText()]
        print(layer.get_config()) ###

        self.close()

class EditHiddenLayerPopupWindow(QWidget):
    def __init__(self, listItem):
        QWidget.__init__(self)
        """mainLayout = QGridLayout(widget)
        self.setLayout(mainLayout)"""
        self.itemName = listItem.text()

        global model
        layer = model.get_layer(name=self.itemName)
        current_neurons = layer.units
        current_activationString = layer.get_config()['activation']
        print(current_neurons) ###
        print(current_activationString)###

        self.listItem = listItem

        vbox = QVBoxLayout(self)
        hbox_lineEdit = QHBoxLayout(self)
        hbox_activation = QHBoxLayout(self)
        hbox_neurons = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        self.activationFunction = QComboBox(self)
        self.activationFunction.addItems(activationFunctionList)
        self.activationFunction.setCurrentIndex(activationFunctionList_lowercase.index(current_activationString))
        self.activationLabel = QLabel("Activation Function:",self)

        self.neurons = QSpinBox(self)
        self.neurons.setValue(current_neurons)
        self.neurons.setMinimum(0)
        self.neuronsLabel = QLabel("Neurons:",self)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setText(listItem.text())
        self.lineEditLabel = QLabel("Layer name:")

        #set bold text
        myFont = QFont()
        myFont.setBold(True)
        self.noteName = QLabel("\nNote: layer name must be unique and can't contain spaces.\n")
        self.noteName.setFont(myFont)

        self.label = QLabel("Edit " + self.itemName + " properties.")
        self.label.setAlignment(Qt.AlignCenter)

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.clicked.connect(self.close)

        pushButton_saveEdit = QPushButton("Save Edit", self)
        pushButton_saveEdit.clicked.connect(self.make_changes)

        hbox_lineEdit.addWidget(self.lineEditLabel)
        hbox_lineEdit.addWidget(self.lineEdit)
        hbox_activation.addWidget(self.activationLabel)
        hbox_activation.addWidget(self.activationFunction)
        hbox_neurons.addWidget(self.neuronsLabel)
        hbox_neurons.addWidget(self.neurons)
        hbox_buttons.addWidget(pushButton_saveEdit)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox.addWidget(self.label)
        vbox.addLayout(hbox_lineEdit)
        vbox.addWidget(self.noteName)
        vbox.addLayout(hbox_neurons)
        vbox.addLayout(hbox_activation)
        vbox.addLayout(hbox_buttons)

        #mainLayout.addLayout(vbox,0,0,0,0)

    def make_changes(self):
        global model
        layer = model.get_layer(name = self.itemName)
        if validLayerName(self.lineEdit.text(), self.itemName):
            print("Name was valid!") ###

            layer._name = self.lineEdit.text() #.name is not changeable, ._name is
            layer.activation = activationFunctionDict[self.activationFunction.currentText()]
            layer.units = self.neurons.value()
            self.listItem.setText(self.lineEdit.text())

            print(layer.get_config())###
            self.close()
        else:
            error = QMessageBox.warning(None, "Error", "\n   Please insert valid layer name.   \n")

class AddHiddenLayerPopupWindow(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        vbox = QVBoxLayout(self)
        hbox_lineEdit = QHBoxLayout(self)
        hbox_activation = QHBoxLayout(self)
        hbox_neurons = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        self.activationFunction = QComboBox(self)
        self.activationFunction.addItems(activationFunctionList)
        self.activationFunction.setCurrentIndex(activationFunctionList_lowercase.index("relu"))
        self.activationLabel = QLabel("Activation Function:",self)

        self.neurons = QSpinBox(self)
        self.neurons.setValue(10)
        self.neurons.setMinimum(0)
        self.neuronsLabel = QLabel("Neurons:",self)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setText("Insert new layer name")
        self.lineEditLabel = QLabel("Layer name:")

        #set bold text
        myFont = QFont()
        myFont.setBold(True)
        self.noteName = QLabel("Note: layer name must be unique, can't contain spaces and\n 'output' and 'input' are invalid names.")
        self.noteName.setFont(myFont)

        self.label = QLabel("Add new layer")
        self.label.setAlignment(Qt.AlignCenter)

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.clicked.connect(self.close)

        pushButton_addNewLayer = QPushButton("Add Layer", self)
        pushButton_addNewLayer.clicked.connect(self.add_layer)

        hbox_lineEdit.addWidget(self.lineEditLabel)
        hbox_lineEdit.addWidget(self.lineEdit)
        hbox_activation.addWidget(self.activationLabel)
        hbox_activation.addWidget(self.activationFunction)
        hbox_neurons.addWidget(self.neuronsLabel)
        hbox_neurons.addWidget(self.neurons)
        hbox_buttons.addWidget(pushButton_addNewLayer)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox.addWidget(self.label)
        vbox.addLayout(hbox_lineEdit)
        vbox.addWidget(self.noteName)
        vbox.addLayout(hbox_neurons)
        vbox.addLayout(hbox_activation)
        vbox.addLayout(hbox_buttons)

        #mainLayout.addLayout(vbox,0,0,0,0)

    def add_layer(self):
        global model
        global list_layers
        if validLayerName(self.lineEdit.text()):
            print("Name was valid!") ###
            #First pop output layer
            model.pop()

            #Add new layer
            model.add(Dense(12, activation = 'relu', name = self.lineEdit.text()))
            layer = model.get_layer(name = self.lineEdit.text())
            layer.activation = activationFunctionDict[self.activationFunction.currentText()]
            layer.units = self.neurons.value()
            #Add listItem to list
            hidden_layers.append(model.get_layer(layer.name))
            list_layers.addItem(layer.name)
            update_hiddenLayersList()

            #ReAdd output layer
            model.add(Dense(output_layer_settingsDict["units"], name = 'output'))
            model.get_layer(name = "output").activation = activationFunctionDict[output_layer_settingsDict["activation"]]

            print(model.summary())###
            self.close()
        else:
            error = QMessageBox.warning(None, "Error", "\n   Please insert valid layer name.   \n")

class TrainPopupWindow(QWidget):
    """Train network popup window"""
    def __init__(self):

        QWidget.__init__(self)
        vbox = QVBoxLayout(self)
        hbox_optimizer = QHBoxLayout(self)
        hbox_loss = QHBoxLayout(self)
        hbox_metrics = QHBoxLayout(self)
        hbox_fit = QHBoxLayout(self)
        hbox_buttons = QHBoxLayout(self)

        #for max value of batch size
        trainData = loadtxt(trainFile_path, delimiter=',')

        self.label = QLabel("Training settings (compile and fit)")
        self.label.setAlignment(Qt.AlignCenter)
        self.optimizerCombo = QComboBox(self)
        self.optimizerCombo.addItems(optimizerList)
        self.optimizerCombo.setCurrentIndex(optimizerList.index(previous_train_settingsDict['optimizer']))
        self.optimizerLabel = QLabel("Optimizer algorithm:",self)

        self.lossCombo = QComboBox(self)
        self.lossCombo.addItems(lossFunctionList)
        self.lossCombo.setCurrentIndex(lossFunctionList.index(previous_train_settingsDict['loss']))
        self.lossLabel = QLabel("Loss function:", self)

        self.metricsCombo = QComboBox(self)
        self.metricsCombo.addItems(metricsList)
        self.metricsCombo.setCurrentIndex(metricsList.index(previous_train_settingsDict['metric']))
        self.metricsLabel = QLabel("Show metric:", self)

        self.epochsSpin = QSpinBox(self)
        self.epochsSpin.setRange(1,10000)
        self.epochsSpin.setValue(previous_train_settingsDict['epochs'])
        self.epochsLabel = QLabel("Number of Epochs: ")

        self.batchSpin = QSpinBox(self)
        self.batchSpin.setRange(1,len(trainData))
        self.batchSpin.setValue(previous_train_settingsDict['batch_size'])
        self.batchLabel = QLabel("Batch size: ")

        #set bold text
        """myFont = QFont()
        myFont.setBold(True)
        self.noteName = QLabel("Note: layer name must be unique, can't contain spaces and\n 'output' and 'input' are invalid names.")
        self.noteName.setFont(myFont)

        self.label = QLabel("Add new layer")
        self.label.setAlignment(Qt.AlignCenter)"""

        pushButton_cancel = QPushButton("Cancel", self)
        pushButton_cancel.clicked.connect(self.close)

        pushButton_trainModel = QPushButton("Train", self)
        pushButton_trainModel.clicked.connect(self.train)

        hbox_optimizer.addWidget(self.optimizerLabel)
        hbox_optimizer.addWidget(self.optimizerCombo)
        hbox_loss.addWidget(self.lossLabel)
        hbox_loss.addWidget(self.lossCombo)
        hbox_metrics.addWidget(self.metricsLabel)
        hbox_metrics.addWidget(self.metricsCombo)
        hbox_fit.addWidget(self.epochsLabel)
        hbox_fit.addWidget(self.epochsSpin)
        hbox_fit.addWidget(self.batchLabel)
        hbox_fit.addWidget(self.batchSpin)
        hbox_buttons.addWidget(pushButton_trainModel)
        hbox_buttons.addWidget(pushButton_cancel)

        vbox.addWidget(self.label)
        vbox.addLayout(hbox_optimizer)
        vbox.addLayout(hbox_loss)
        vbox.addLayout(hbox_metrics)
        vbox.addLayout(hbox_fit)
        vbox.addLayout(hbox_buttons)

    def train(self):
        global textBrowser
        textBrowser.clear()

        #update previous settings
        previous_train_settingsDict['optimizer'] = self.optimizerCombo.currentText()
        previous_train_settingsDict['loss'] = self.lossCombo.currentText()
        previous_train_settingsDict['metric'] = self.metricsCombo.currentText()
        previous_train_settingsDict['epochs'] = self.epochsSpin.value()
        previous_train_settingsDict['batch_size'] = self.batchSpin.value()

        self.thread = QThread()

        self.worker = train_model_worker()
        self.worker.set_parameters(self.optimizerCombo.currentText(), lossFunctionList_lowercase[lossFunctionList.index(self.lossCombo.currentText())], metricsList_lowercase[metricsList.index(self.metricsCombo.currentText())], self.epochsSpin.value(), self.batchSpin.value())
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(print_signal)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.print_metric)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.close)

        # start the thread
        self.thread.start()

    def print_metric(self):
        """Prints metric for model"""
        _, columns = read_file(trainFile_path, ',')
        trainData = loadtxt(trainFile_path, delimiter=',')
        input = trainData[:, inputCol_start:inputCol_end + 1]
        output = trainData[:, outputCol_start:outputCol_end + 1]

        _, metric_value = model.evaluate(input, output, verbose=0)

        predictions = model.predict(input)
        rounded = [np.round(p) for p in predictions]

        textBrowser.append(self.metricsCombo.currentText() + ': %.4f' % (metric_value))  ###

        ###Below is temporary
        for i in range(columns):
            textBrowser.append(str(input[i].tolist()) + ' -> ' + str(predictions[i]) + ' (expected ' + str(output[i]) + ')')  ###?



## File functions ##

#Read data file
def read_file(filename, delimiter):
    """
    Lê dados de um ficheiro

    Parameters
    ----------
    filename : str
        Nome do ficheiro com caminho absoluto/relativo completo.
    delimiter : str
        Delimiter between dataa in same line

    Returns
    -------
    output : list of list
        output[0] corresponde ao conteúdo da coluna 1, output[1] ao da coluna
        2,....
    columns : int
        Number of columns in file
    """

    f = open(filename, 'r')
    data = f.readlines()
    temp = data[0].split(delimiter)
    columns = len(temp)

    output = [[] for _ in range(columns)]

    for datum in data:
        temp = datum.split(delimiter)
        for i in range(len(temp)):
            output[i].append(float(temp[i]))

    return output, columns

#Browse file buttons
def open_trainFile():  #open train data file
    global trainFile_path
    path, _ = QFileDialog.getOpenFileName(None,
                        "Train data",
                        "",
                        "Text files (*.txt)",
                        options=QFileDialog.Options())
    trainFile_path = path
    lineEdit_trainFile.setText(trainFile_path)
    """if path.find('.txt') != -1: # .txt filter WITH "OPTIONS" I DONT NEED THIS
        trainFile_path = path
        lineEdit_trainFile.setText(trainFile_path)
    else:
        error = QErrorMessage()
        error.showMessage("Please select a .txt file!")
        error.exec()"""

def open_testFile():  #open test data file
    global testFile_path
    path, _ = QFileDialog.getOpenFileName(None,
                        "Test data",
                        "",
                        "Text files (*.txt)",
                        options=QFileDialog.Options())
    testFile_path = path
    lineEdit_testFile.setText(testFile_path)
    """if path.find('.txt') != -1:  # .txt filter
        testFile_path = path
        lineEdit_testFile.setText(testFile_path)
    else:
        error = QErrorMessage()
        error.showMessage("Please select a .txt file!")
        error.exec()"""

#Checks if path is a file
def validPath(path):
        try:
            file = open(path, 'r')
            s = file.read()
            #print ('read', len(s), 'bytes')
            if path.find('.txt') != -1: #only accepts .txt files
                return True
            else:
                return False
        except:
            return False

#Update file names for lineEdits
def update_trainPath():
    global trainFile_path
    trainFile_path = lineEdit_trainFile.text()

    """#Checks if its valid path
    if validPath(path):
        trainFile_path = path
        lineEdit_trainFile.setText(trainFile_path)
        print(trainFile_path)
    else:
        error = QErrorMessage()
        error.showMessage("Please select .txt file!")
        error.exec()"""

def update_testPath():
    global testFile_path
    testFile_path = lineEdit_testFile.text()

    """#Checks if its valid path
    if validPath(path):
        testFile_path = path
        lineEdit_testFile.setText(testFile_path)
        print(testFile_path)
    else:
        error = QErrorMessage()
        error.showMessage("Please select .txt file!")
        error.exec()"""

## Train/Test ##
def get_inputOutput(path): ###NOT USED
    global inputCol_start, inputCol_end, outputCol_start, outputCol_end

    """Get input and output lists with given input and output columns"""
    data, columns = read_file(path, ',')
    input = [[] for _ in range(inputCol_end-inputCol_start+1) ]
    output = [[] for _ in range(outputCol_end-outputCol_start+1)]
    #input
    k = 0
    for i in range(inputCol_start, inputCol_end+1):
        input[k] = data[i]
        k+=1
    #output
    k = 0
    for i in range(outputCol_start, outputCol_end + 1):
        output[k] = data[i]
        k+=1
    return input, output

def train_model_button():
    """Train button function"""
    ##INSERT CHECK FOR IF MODEL HAS BEEN CREATED, OR IF INPUT/OUTPUT SHAPE HAVE CHANGED
    if validModel():
        try: _, columns = read_file(trainFile_path, ',')
        except:
            pass

        if validPath(trainFile_path) and validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns) \
                and validInputOutputShapes():
            global trainPopup
            trainPopup = TrainPopupWindow()
            trainPopup.setGeometry(QRect(400, 400, 100, 100))
            trainPopup.setWindowTitle("Train current model")
            trainPopup.show()

        elif not validPath(trainFile_path):
            error = QMessageBox.warning(None, "Error", "\n   Please select a training file.   \n")

        elif not validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns):
            error = QMessageBox.warning(None, "Error", "\n   Invalid input or output columns.   \n")

        elif not validInputOutputShapes():
            error = QMessageBox.warning(None, "Error", "\n   Invalid input or output shapes.   \n   Note: Can't change input or output size after\n creating model.   \n")

    else: #invalid model
        error = QMessageBox.warning(None, "Error", "\n   Invalid model.   \n")

def train_model(optimizer, loss, metric, epochs, batch_size):
    """Train the model with given parameters """
    _, columns = read_file(trainFile_path, ',')
    textBrowser.clear()

    is_training = True
    # get input and output columns
    trainData = loadtxt(trainFile_path, delimiter=',')
    input = trainData[:, inputCol_start:inputCol_end + 1]
    output = trainData[:, outputCol_start:outputCol_end + 1]

    # compile Keras model
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics=[metric])

    # fit (train) the Keras model on the dataset
    model.fit(input, output, epochs = epochs, batch_size = batch_size, verbose=0)
    _, metric_value = model.evaluate(input, output, verbose=0)

    is_training = False
    ###Below is temporary
    # make predictions with model
    predictions = model.predict(input)
    rounded = [np.round(p) for p in predictions]

    textBrowser.append(metric + ': %.4f' % (metric_value))  ###
    textBrowser.append("Model Trained!")  ###

    for i in range(columns):
        textBrowser.append('%s -> %d (expected %d)' % (input[i].tolist(), rounded[i], output[i]))  ###?

class train_model_worker(QObject):
    """Train Model Worker class to run on separate QThread"""
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def set_parameters(self, optimizer_in, loss_in, metric_in, epochs_in, batch_size_in):
        """Set parameters to use in training"""
        self.optimizer = optimizer_in
        self.loss = loss_in
        self.metric = metric_in
        self.epochs = epochs_in
        self.batch_size = batch_size_in
    def run(self):
        """Train the model with given parameters """
        _, columns = read_file(trainFile_path, ',')

        is_training = True
        # get input and output columns
        trainData = loadtxt(trainFile_path, delimiter=',')
        input = trainData[:, inputCol_start:inputCol_end + 1]
        output = trainData[:, outputCol_start:outputCol_end + 1]

        # compile Keras model
        self.progress.emit("Compiling model...\n")
        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      metrics=[self.metric])
        self.progress.emit("Model compiled.\n")

        # fit (train) the Keras model on the dataset
        self.progress.emit("Fitting model, could take a while...\n")
        model.fit(input, output, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.progress.emit("Model fit.\n")

        is_training = False
        self.finished.emit()

def print_signal(string):
    """Receives signals worker outputs progress to textBrowser"""
    textBrowser.append(string)


def test_model():
    """Test button function"""
    try: testData, columns = read_file(testFile_path, ',')
    except:
        pass
    if validPath(testFile_path) and validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns):
        """
        code
        """
        textBrowser.clear()
        print("file worked")    ###

    elif not validPath(testFile_path):
        error = QMessageBox.warning(None, "Error", "\n   Please select a testing file.   \n")

    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid input or output columns.   \n")

def validColumns(inputStart, inputEnd,outputStart, outputEnd, totalCols):
    """Checks if input/output columns are valid, given the file"""
    if inputEnd >= inputStart and outputEnd>=outputStart and (outputStart > inputEnd or inputStart > outputEnd) \
            and max([inputStart, inputEnd, outputStart, outputEnd]) <= (totalCols - 1):

        return True
    else:
        return False

def validInputOutputShapes():
    """Checks if current selection of input and output columns
    is valid with the model's"""
    global model
    model_inputDim = model.get_layer(name = 'input').input_shape[1] #number of inputs
    model_outputDim = model.get_layer(name = 'output').units
    current_inputDim = inputCol_end - inputCol_start + 1
    current_outputDim = outputCol_end - outputCol_start + 1

    #Non compatible shapes
    if model_inputDim != current_inputDim or model_outputDim != current_outputDim:
        return False
    else:
        return True

def update_inputCols():
    """Input spinBoxes function"""
    global inputCol_start
    global inputCol_end

    inputCol_start = spinBox_inputStart.value()
    inputCol_end = spinBox_inputEnd.value()

def update_outputCols():
    """Output spinBoxes function"""
    global outputCol_start
    global outputCol_end

    outputCol_start = spinBox_outputStart.value()
    outputCol_end = spinBox_outputEnd.value()

## Layers Functions ##

def edit_inputLayer():
    """Edit input layer function"""
    if validModel():
        global editPopup
        editPopup = EditInputLayerPopupWindow()
        editPopup.setGeometry(QRect(400, 400, 100, 100))
        editPopup.setWindowTitle("Edit Input layer")
        editPopup.show()
    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid model.   \n")

def edit_outputLayer():
    """Edit output layer function"""
    if validModel():
        global editPopup
        editPopup = EditOutputLayerPopupWindow()
        editPopup.setGeometry(QRect(400, 400, 100, 100))
        editPopup.setWindowTitle("Edit Output layer")
        editPopup.show()
    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid model.   \n")

def edit_hiddenLayer():
    """Edit hidden layer function"""
    if validModel() and list_layers.currentItem() != None:
        global editHiddenPopup
        editHiddenPopup = EditHiddenLayerPopupWindow(list_layers.currentItem())
        editHiddenPopup.setGeometry(QRect(400, 400, 100, 100))
        editHiddenPopup.setWindowTitle("Edit " + list_layers.currentItem().text())
        editHiddenPopup.show()
    elif list_layers.currentItem() != None:
        error = QMessageBox.warning(None,"Error","\n   Invalid model.   \n")

def delete_hiddenLayer():
    global model
    if validModel():
        if len(hidden_layers)!=0:
            #Pops output AND last top layer in stack
            model.pop()
            model.pop()
            hidden_layers.pop()
            update_hiddenLayersList()

            #ReAdd output layer
            model.add(Dense(output_layer_settingsDict["units"], name = 'output'))
            model.get_layer(name = "output").activation = activationFunctionDict[output_layer_settingsDict["activation"]]


            print(model.summary())###

    if not validModel():
            error = QMessageBox.warning(None,"Error","\n   Invalid model.   \n")

def add_hiddenLayer():
    """Add hidden layer function"""
    if validModel():
        global addHiddenPopup
        addHiddenPopup = AddHiddenLayerPopupWindow()
        addHiddenPopup.setGeometry(QRect(400, 400, 100, 100))
        addHiddenPopup.setWindowTitle("Add hidden layer")
        addHiddenPopup.show()
    elif list_layers.currentItem() != None:
        error = QMessageBox.warning(None, "Error", "\n   Invalid model.   \n")

def update_hiddenLayersList():
    """Updates hidden layers Qlist with hidden_layers list """
    global hidden_layers

    list_layers.clear()
    for layer in hidden_layers:
        list_layers.addItem(layer.name)

def validLayerName(string, currentName=None):
    """Checks if layer name is valid
    - no spaces
    - no repeated layer names (including 'input' and 'output')
     """
    valid = True

    global hidden_layers
    if string == currentName and currentName!=None:
        #If the name is unchanged it's okay
        return True
    for layer in hidden_layers:
        if string == 'output' or string == 'input':
            valid = False
        if " " in string:
            valid = False
        #Check for other layers other than itself
        elif layer.name != currentName:
            if layer.name == string:
                valid = False
        if valid == False:
            break


    return valid

def validModel():
    """Checks if model is valid (has been created)"""
    global model
    if model == None:
        return False
    return True

## MenuBar Functions ##

def new_model():
    global model_name
    """creates new model"""
    try: trainData, columns = read_file(trainFile_path, ',')
    except:
        pass

    if validPath(trainFile_path) and validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns):

        global popupWindow #Popup window for new model
        popupWindow = NewModelPopupWindow()
        popupWindow.setGeometry(QRect(400, 400, 100, 100))
        popupWindow.setWindowTitle("New model")
        popupWindow.show()

    elif not validPath(trainFile_path):
        error = QMessageBox.warning(None, "Error", "\n   Please select a training file.   \n")
    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid input or output columns.   \n")


#Themes
def set_darkTheme():
    apply_stylesheet(app, theme='dark_custom.xml', extra=extra)
    stylesheet = app.styleSheet()
    with open('../stylesheets/Qt_Material/custom_QtPushButton.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))
def set_lightTheme():
    apply_stylesheet(app, theme='light_custom.xml',invert_secondary=False, extra=extra)
    stylesheet = app.styleSheet()
    with open('../stylesheets/Qt_Material/custom_QtPushButton.css') as file:
        app.setStyleSheet(stylesheet + file.read().format(**os.environ))

### Main ###
app = QApplication([])
app.setApplicationName("Neurons 1.0")

window = QMainWindow()
window.setGeometry(500,200,800,650) #(position xy, size xy)

widget = QWidget(window)

window.setCentralWidget(widget)

## Variables ##

# File paths
trainFile_path = "../Data/diabetes_data.txt"
testFile_path = None

#Input/Output columns
inputCol_start = 0
inputCol_end = 0
outputCol_start = 0
outputCol_end = 0
inputNumber = 0
outputNumber = 0

#Model
model = None
hidden_layers = None
model_name = ''
popupWindow = None

#Auxiliary lists and dicts
activationFunctionList = ["ReLu", "Sigmoid", "SoftMax", "SoftPlus", "SoftSign", "Tanh", "SeLu", "Elu", "Exponential"]
activationFunctionList_lowercase = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]
activationFunctionDict = {"ReLu": tf.keras.activations.relu, "Sigmoid": tf.keras.activations.sigmoid, "SoftMax": tf.keras.activations.softmax,
                          "SoftPlus": tf.keras.activations.softplus, "SoftSign": tf.keras.activations.softsign, "Tanh": tf.keras.activations.tanh,
                          "SeLu": tf.keras.activations.selu, "Elu": tf.keras.activations.elu, "Exponential": tf.keras.activations.exponential}
output_layer_settingsDict = {"units": 0, "activation": "ReLu"} #To save output settings for Add and Delete button methods
optimizerList = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
lossFunctionList = ['Mean Squared Error','Mean Squared Logarithmic Error', 'Huber', 'Binary Crossentropy', 'Categorical Crossentropy']
lossFunctionList_lowercase = ['mean_squared_error','mean_squared_logarithmic_error', 'huber_loss','binary_crossentropy','categorical_crossentropy']
metricsList = ['Accuracy', 'Binary Accuracy','Mean Squared Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error']
metricsList_lowercase = ['accuracy', 'binary_accuracy', 'mean_squared_error','mean_absolute_error','mean_absolute_percentage_error']
previous_train_settingsDict = {'optimizer': 'Adam', 'loss': 'Mean Squared Error', 'metric': 'Accuracy', 'epochs': 100, 'batch_size': 10}
is_training = False
## Items ##

#Train and test
myFont = QFont() #Bold font
myFont.setBold(True)
myFont.setPointSize(15)
label_modelName = QLabel("Model: No model selected")
label_modelName.setFont(myFont)
label_modelName.setAlignment(Qt.AlignCenter)
label_modelName.setMaximumHeight(40)
label_trainFile = QLabel("Train File:", widget)
label_testFile = QLabel("Test File:", widget)
label_inputColumns = QLabel("Input Columns:", widget)
label_outputColumns = QLabel("Output Columns:", widget)
label_ToIn = QLabel("to", widget)
label_ToOut = QLabel("to", widget)

lineEdit_trainFile = QLineEdit("../Data/diabetes_data.txt",widget)
lineEdit_testFile = QLineEdit("Path to .txt testing file",widget)

pushButton_trainFile = QPushButton("Browse", widget)
pushButton_testFile = QPushButton("Browse", widget)
pushButton_test = QPushButton("Test", widget)
pushButton_train = QPushButton("Train", widget)

spinBox_inputStart = QSpinBox(widget)
spinBox_inputEnd = QSpinBox(widget)
spinBox_outputStart = QSpinBox(widget)
spinBox_outputEnd = QSpinBox(widget)

#Text Browser
textBrowser = QTextBrowser(widget)
textBrowser.setMinimumHeight(300)
textBrowserLabel = QLabel("Output:")

#Layers List Widget
list_layers = QListWidget(widget)

list_inputLayer = QListWidget(widget)
list_inputLayer.setFixedHeight(40)
list_inputLayer.addItem("Input")


list_outputLayer = QListWidget(widget)
list_outputLayer.setFixedHeight(40)
list_outputLayer.addItem("Output")

label_layers = QLabel("Layers", widget)
label_layers.setFont(myFont)
label_layers.setAlignment(Qt.AlignCenter)
label_hiddenLayers = QLabel("Hidden Layers:", widget)

pushButton_addLayer = QPushButton("Add", widget)
pushButton_deleteLayer = QPushButton("Delete last", widget)
pushButton_editLayer = QPushButton("Edit Hidden", widget)

## Connections ##
#Train and test file pushButtons
pushButton_trainFile.pressed.connect(open_trainFile) #open file browser
pushButton_testFile.pressed.connect(open_testFile) #open file browser

#Train and test file lineEdits
lineEdit_trainFile.returnPressed.connect(update_trainPath)
lineEdit_trainFile.textChanged.connect(update_trainPath)
lineEdit_testFile.returnPressed.connect(update_testPath)
lineEdit_testFile.textChanged.connect(update_testPath)

#Train and test pushButtons
pushButton_train.pressed.connect(train_model_button)
pushButton_test.pressed.connect(test_model)

#Add, delete, edit pushButtons
list_inputLayer.doubleClicked.connect(edit_inputLayer)
list_outputLayer.doubleClicked.connect(edit_outputLayer)
pushButton_deleteLayer.pressed.connect(delete_hiddenLayer)
pushButton_editLayer.pressed.connect(edit_hiddenLayer) ###Este botão é meio inutil
list_layers.doubleClicked.connect(edit_hiddenLayer)
pushButton_addLayer.pressed.connect(add_hiddenLayer)

#Input Output spinBoxes
spinBox_inputStart.valueChanged.connect(update_inputCols)
spinBox_inputEnd.valueChanged.connect(update_inputCols)
spinBox_outputStart.valueChanged.connect(update_outputCols)
spinBox_outputEnd.valueChanged.connect(update_outputCols)

## HBoxes ##

#Train files Hbox
hbox_trainFile = QHBoxLayout()
hbox_trainFile.addWidget(label_trainFile)
hbox_trainFile.addWidget(lineEdit_trainFile)
hbox_trainFile.addWidget(pushButton_trainFile)

#Test files Hbox
hbox_testFile = QHBoxLayout()
hbox_testFile.addWidget(label_testFile)
hbox_testFile.addWidget(lineEdit_testFile)
hbox_testFile.addWidget(pushButton_testFile)

#Input Columns Hbox
hbox_inputColumns = QHBoxLayout()
hbox_inputColumns.addWidget(label_inputColumns)
hbox_inputColumns.addWidget(spinBox_inputStart)
hbox_inputColumns.addWidget(label_ToIn)
hbox_inputColumns.addWidget(spinBox_inputEnd)

#Output Columns Hbox
hbox_outputColumns = QHBoxLayout()
hbox_outputColumns.addWidget(label_outputColumns)
hbox_outputColumns.addWidget(spinBox_outputStart)
hbox_outputColumns.addWidget(label_ToOut)
hbox_outputColumns.addWidget(spinBox_outputEnd)

#Train and test buttons Hbox
hbox_trainTestButtons = QHBoxLayout()
hbox_trainTestButtons.addWidget(pushButton_train)
hbox_trainTestButtons.addWidget(pushButton_test)

#Layer buttons Hbox
hbox_layerButtons = QHBoxLayout()
hbox_layerButtons.addWidget(pushButton_addLayer)
hbox_layerButtons.addWidget(pushButton_deleteLayer)
hbox_layerButtons.addWidget(pushButton_editLayer)

#Top Hbox
hbox_top = QHBoxLayout()

## VBoxes ##

#Files and Train Test buttons Vbox
vbox_files = QVBoxLayout()
vbox_files.addWidget(label_modelName)
vbox_files.addLayout(hbox_trainFile)
vbox_files.addLayout(hbox_testFile)
vbox_files.addLayout(hbox_inputColumns)
vbox_files.addLayout(hbox_outputColumns)
vbox_files.addLayout(hbox_trainTestButtons)

#Layers List Hbox
vbox_layers = QVBoxLayout()
vbox_layers.addWidget(label_layers)
vbox_layers.addWidget(list_inputLayer)
vbox_layers.addWidget(label_hiddenLayers)
vbox_layers.addWidget(list_layers)
vbox_layers.addWidget(list_outputLayer)
vbox_layers.addLayout(hbox_layerButtons)

#Main VBox
vbox_main = QVBoxLayout()

## MenuBar ##
menuBar = QMenuBar(widget)
menuFile = menuBar.addMenu('&File')
menuFile_newModel = menuFile.addAction("New model")
menuFile_newModel.setShortcut("Ctrl+N")
menuFile_newModel.triggered.connect(new_model)

menuFile_newWeights = menuFile.addAction("New weights")
menuFile_newWeights.setShortcut("Ctrl+Shift+N")

menuFile_saveModel = menuFile.addAction("Save model")
menuFile_saveModel.setShortcut("Ctrl+S")

menuFile_saveWeights = menuFile.addAction("Save weights")
menuFile_saveWeights.setShortcut("Ctrl+Shift+S")

menuTheme = menuBar.addMenu("&Theme")
menuTheme_dark = menuTheme.addAction("&Dark")
menuTheme_dark.triggered.connect(set_darkTheme)
menuTheme_light = menuTheme.addAction("&Light")
menuTheme_light.triggered.connect(set_lightTheme)

## Ordering Boxes ##
hbox_top.addLayout(vbox_files)
hbox_top.addLayout(vbox_layers)

vbox_main.addWidget(menuBar)
vbox_main.addLayout(hbox_top)
vbox_main.addWidget(textBrowserLabel)
vbox_main.addWidget(textBrowser)



## Main Layout ##
mainLayout = QGridLayout(widget)

mainLayout.addLayout(vbox_main, 0,0,0,0)
window.setLayout(mainLayout)



## Testing ##
extra = {
    # Button colors
    'danger': '#dc3545',
    'warning': '#ffc107',
    'success': '#17a2b8',

    # Font
    'font-family': 'Montserrat',
}
apply_stylesheet(app, theme='dark_custom.xml', extra=extra)

stylesheet = app.styleSheet()
with open('../stylesheets/Qt_Material/custom_QtPushButton.css') as file:
    app.setStyleSheet(stylesheet + file.read().format(**os.environ))

#app.setStyleSheet(qdarkstyle.load_stylesheet())

if __name__ == "__main__":
    window.show()
    app.exec_()
