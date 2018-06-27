import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
import sklearn.metrics as sklm
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix


class Metrics(Callback):
    def __init__(self, validation_data=()):
        super(Callback,self).__init__()

        self.validation_data = validation_data


    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []


    def on_epoch_end(self, epoch, logs={}):
        #score = numpy.asarray(self.model.predict(self.model.validation_data[0]))
        predict = numpy.round(
            numpy.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.recall.append(sklm.recall_score(targ, predict))

        print ' - val_precision: %f - val_recall %f' % (self.precision[-1], self.recall[-1])
        return


def text2hash(df):
    df.columns = columnsHead
    df['protocol_type'] = df['protocol_type'].apply(
        lambda x: hashing_trick(x, 200, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ '))
    df['service'] = df['service'].apply(
        lambda x: hashing_trick(x, 200, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ '))
    df['flag'] = df['flag'].apply(
        lambda x: hashing_trick(x, 200, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ '))

def removeListValues(matrix):
    i = 0
    for row in matrix[:, :]:
        j = 0
        for el in row:
            if isinstance(el, list):
                matrix[i][j] = float(el[0])
            j += 1
        i += 1
    return matrix

def m2Binary(classes):
    classes = [x if x == 'normal' else 'anomaly'  for x in classes]
    return classes

columnsHead = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'outcome'
]

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load the dataset
train = pandas.read_csv(
    '../NSL-KDD-Dataset-master/KDDTrain+.csv',
    usecols=range(0, 42),
    engine='python',
    skipfooter=0)

test = pandas.read_csv(
    '../NSL-KDD-Dataset-master/KDDTest+.csv',
    usecols=range(0, 42),
    engine='python',
    skipfooter=0)

train.columns = columnsHead
test.columns = columnsHead

text2hash(train)
text2hash(test)

train = train.values
XTrain = train[:, 0:41]
XTrain = removeListValues(XTrain)
XTrain = XTrain.astype('float32')
YTrain = train[:, 41]
YTrain = m2Binary(YTrain)

#print XTrain[0:5]

test = test.values
XTest = test[:, 0:41]
XTest = removeListValues(XTest)
XTest = XTest.astype('float32')
YTest = test[:, 41]
YTest = m2Binary(YTest)


#reshaping inputs as expected by lstm with time step size = 100
#XTrain = numpy.reshape(XTrain, (len(XTrain), 100, len(XTrain[0])))
#XTest = numpy.reshape(XTest, (len(XTest), 100, len(XTrain[0])))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.fit_transform(XTest)

encoder = LabelEncoder()
encoder.fit(YTrain)
encoded_YTrain = encoder.transform(YTrain)

# convert integers to dummy variables (i.e. one hot encoded)
#dummyYTrain = np_utils.to_categorical(encoded_YTrain)
dummyYTrain = encoded_YTrain

encoder.fit(YTest)
encoded_YTest = encoder.transform(YTest)
dummyYTest = encoded_YTest
# convert integers to dummy variables (i.e. one hot encoded)
#dummyYTest = np_utils.to_categorical(encoded_YTest)
#print dummyYTest
# normalize the dataset

print len(XTrain), len(XTest)
#print len(dummyYTrain[0]), len(dummyYTest[0])
print len(dummyYTrain), len(dummyYTest)

print dummyYTrain.shape

#obtain validation data

# split into train and valid sets
train_size = int(len(XTrain) * 0.67)
validation_size = len(XTrain) - train_size
XTrain, XValid = XTrain[0:train_size,:], XTrain[train_size:len(XTrain),:]
dummyYTrain, dummyYValid = dummyYTrain[0:train_size], dummyYTrain[
    train_size:len(dummyYTrain)]
print(len(XTrain), len(XValid))


def trainModel():
    # create the model
    model = Sequential()
    model.add(
        Embedding(
            len(XTrain), 80, input_length=41))
    model.add(LSTM(80))
    model.add(Dense(1, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01)
    metrics = Metrics((XValid, dummyYValid))
    model.compile(
        loss='mean_squared_error',
        optimizer=sgd,
        metrics=['accuracy', metrics.recall, metrics.precision])
    print(model.summary())
    model.fit(
        XTrain,
        dummyYTrain,
        epochs=500,
        batch_size=500,
        callbacks=[metrics],
        validation_data=(XValid, dummyYValid),
        )  #Bs 50
    # Final evaluation of the model
    scores = model.evaluate(XTest, dummyYTest, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_binary.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_binary.h5")
    print("Saved model to disk")

def loadAndEvaluate() :
    model = './model_binary_batch500'

    # load json and create model
    json_file = open( model+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model+'.h5')
    print("Loaded model from disk")

    sgd = optimizers.SGD(lr=0.01)
    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    score = loaded_model.evaluate(XTest, dummyYTest, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    print("%s: %.2f%%" % (loaded_model.metrics_names[2], score[2]*100))
    print("%s: %.2f%%" % (loaded_model.metrics_names[3], score[3]*100))
    #print("acc2:  %.2f%%" % score[2])


trainModel()