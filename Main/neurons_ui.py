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
        model.add(Dense(outputDim, activation ='relu', name = 'output')) #name = 'dense' by default, then 'dense_1', etc

        hidden_layers.append(model.layers[1])
        update_hiddenLayersList(hidden_layers[0])
        self.close()

class EditInputLayerPopupWindow(QWidget):
    """Edit input layer popup window"""
    def __init__(self, listItem):
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

        self.itemName = listItem.text()
        self.label = QLabel("Edit " + self.itemName + " properties.")
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

class EditHiddenLayerPopupWindow(QWidget):
    def __init__(self, listItem):
        QWidget.__init__(self)
        """mainLayout = QGridLayout(widget)
        self.setLayout(mainLayout)"""

        self.listItem = listItem

        vbox = QVBoxLayout(self)
        hbox_lineEdit = QHBoxLayout(self)
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

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setText(listItem.text())
        self.lineEditLabel = QLabel("Layer name:")

        #set bold text
        myFont = QFont()
        myFont.setBold(True)
        self.noteName = QLabel("Note: layer name must be unique and can't contain spaces.")
        self.noteName.setFont(myFont)

        self.itemName = listItem.text()
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
def train_network():
    """Train button function"""
    try: trainData, columns = read_file(trainFile_path, ',')
    except:
        pass

    if validPath(trainFile_path) and validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns):

        """
        code
        """
        textBrowser.clear()
        textBrowser.append(str(trainData[0])) ####
        textBrowser.append(str(columns))    ###
    elif not validPath(trainFile_path):
        error = QMessageBox.warning(None, "Error", "\n   Please select a training file.   \n")

    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid input or output columns.   \n")


def test_network():
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
    print(outputCol_start, outputCol_end)   ###

## Layers Functions ##

def edit_inputLayer():
    """Edit input layer function"""
    if validModel():
        global editPopup
        editPopup = EditInputLayerPopupWindow(list_inputLayer.currentItem())
        editPopup.setGeometry(QRect(400, 400, 100, 100))
        editPopup.setWindowTitle("Edit " + list_inputLayer.currentItem().text())
        editPopup.show()
    else:
        error = QMessageBox.warning(None, "Error", "\n   Invalid model.   \n")


def edit_hiddenLayer():
    """Edit hidden layer function"""
    if validModel():
        global editHiddenPopup
        editHiddenPopup = EditHiddenLayerPopupWindow(list_layers.currentItem())
        editHiddenPopup.setGeometry(QRect(400, 400, 100, 100))
        editHiddenPopup.setWindowTitle("Edit " + list_layers.currentItem().text())
        editHiddenPopup.show()
    else:
        error = QMessageBox.warning(None,"Error","\n   Invalid model.   \n")

def update_hiddenLayersList(layer):
    """Adds item to hidden layer list"""
    list_layers.addItem(layer.name)

def validLayerName(string, currentName):
    """checks if layer name is valid
    - no spaces
    - no repeated layer names"""
    valid = True

    global hidden_layers
    if string == currentName:
        #If the name is unchanged it's okay
        return True
    for layer in hidden_layers:
        if " " in string:
            valid = False
        #Check for other layers other than itself
        elif layer.name != currentName:
            if layer.name == string:
                valid = False

    return valid

def validModel():
    """Checks if model is valid (has been created)"""
    global model
    if model == None:
        return False
    return True
## MenuBar Functions ##

def new_Model():
    """creates new model"""
    try: trainData, columns = read_file(trainFile_path, ',')
    except:
        pass

    if validPath(trainFile_path) and validColumns(inputCol_start, inputCol_end, outputCol_start, outputCol_end, columns):
        """
        code
        """

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
    app.setStyleSheet(qdarkstyle.load_stylesheet())

def set_defaultTheme():
    app.setStyleSheet("")

### Main ###
app = QApplication([])
app.setApplicationName("Neurons 1.0")

window = QMainWindow()
window.setGeometry(500,500,800,300) #(position xy, size xy)

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
model_name = 'default_model'
popupWindow = None

#Layers
activationFunctionList = ["ReLu", "Sigmoid", "SoftMax", "SoftPlus", "SoftSign", "Tanh", "SeLu", "Elu", "Exponential"]
activationFunctionDict = {"ReLu": tf.keras.activations.relu, "Sigmoid": tf.keras.activations.sigmoid, "SoftMax": tf.keras.activations.softmax,
                          "SoftPlus": tf.keras.activations.softplus, "SoftSign": tf.keras.activations.softsign, "Tanh": tf.keras.activations.tanh,
                          "SeLu": tf.keras.activations.selu, "Elu": tf.keras.activations.elu, "Exponential": tf.keras.activations.exponential}
## Items ##

#Train and test
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

#Layers List Widget
list_layers = QListWidget(widget)

list_inputLayer = QListWidget(widget)
list_inputLayer.setFixedHeight(20)
list_inputLayer.addItem("Input")

list_outputLayer = QListWidget(widget)
list_outputLayer.setFixedHeight(20)
list_outputLayer.addItem("Output")

label_layers = QLabel("Layers", widget)
label_layers.setAlignment(Qt.AlignCenter)
label_hiddenLayers = QLabel("Hidden Layers:", widget)

pushButton_addLayer = QPushButton("Add", widget)
pushButton_deleteLayer = QPushButton("Delete", widget)
pushButton_editLayer = QPushButton("Edit", widget)

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
pushButton_train.pressed.connect(train_network)
pushButton_test.pressed.connect(test_network)

#Add, delete, edit pushButtons
list_inputLayer.doubleClicked.connect(edit_inputLayer)

list_layers.doubleClicked.connect(edit_hiddenLayer)

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
menuFile_newModel.triggered.connect(new_Model)

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
menuTheme_light.triggered.connect(set_defaultTheme)

## Ordering Boxes ##
hbox_top.addLayout(vbox_files)
hbox_top.addLayout(vbox_layers)

vbox_main.addWidget(menuBar)
vbox_main.addLayout(hbox_top)
vbox_main.addWidget(textBrowser)



## Main Layout ##
mainLayout = QGridLayout(widget)

mainLayout.addLayout(vbox_main, 0,0,0,0)
window.setLayout(mainLayout)



## Testing ##
list_layers.addItem("test item")


app.setStyleSheet(qdarkstyle.load_stylesheet())
app.setStyleSheet("")

if __name__ == "__main__":
    window.show()
    app.exec_()
