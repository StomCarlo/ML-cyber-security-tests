import numpy
import matplotlib.pyplot as plt
import pandas
import math
import sklearn.metrics as sklm
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier

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

def text2hash(df,cols):
    df.columns = cols
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

    #sgd = optimizers.SGD(lr=0.01)
    # evaluate loaded model on test data
    #loaded_model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy', precision, recall])
    #score = loaded_model.evaluate(XTest, dummyYTest, verbose=1)
    #print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    #print("precision:  %.2f%%" % score[2])
    #print("recall:  %.2f%%" % score[2])

    predictions = loaded_model.predict(testX, batch_size=500)
    val_preds = numpy.argmax(predictions, axis=-1)
    print val_preds
    val_trues = testY
    cm = metrics.confusion_matrix(val_trues, val_preds)
    print cm
    tp, fp, fn, tn =cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print tp
    print 'accuracy ', (tp+tn)/(tp+tn+fp+fn+0.0)*100
    print metrics.accuracy_score(val_trues,val_preds)*100

    # print("%s: %.2f%%" % ('accuracy',
    # (tp + tn) / (tp + tn + fp + fn + 0.0) * 100))
    #print("precision:  %.2f%%" % score[2])
    #print("recall:  %.2f%%" % score[2])
    print 'precision ', (tp) / (tp + fp + 0.0) * 100
    print metrics.precision_score(val_trues, val_preds)*100
    print 'recall ', (tp) / (tp + fn + 0.0) * 100
    print metrics.recall_score(val_trues, val_preds)*100
    print 'far ', fp / (fp + tn + 0.0) * 100


def loadDataset(path, ignoreList=[]): #ingoreList must be ordered
    #remove undesired features
    colNumbers = range(0, 42)
    cols = columnsHead[:]
    ignoreList.sort()
    for i in reversed(ignoreList):
        del colNumbers[i]
        del cols[i]
    # load the dataset
    dt = pandas.read_csv(
        path,#'../NSL-KDD-Dataset-master/KDDdt+.csv'
        usecols = colNumbers,
        engine='python',
        skipfooter=0)

    text2hash(dt,cols)

    dt = dt.values

    Xdt = dt[:, 0:41-len(ignoreList)]
    Xdt = removeListValues(Xdt)
    Xdt = Xdt.astype('float32')
    Ydt = dt[:, 41-len(ignoreList)]
    Ydt = m2Binary(Ydt)

    #reshaping inputs as expected by lstm with time step size = 100
    #Xdt = numpy.reshape(XTrain, (len(XTrain), 100, len(XTrain[0])))
    #XTest = numpy.reshape(XTest, (len(XTest), 100, len(XTrain[0])))

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    Xdt = scaler.fit_transform(Xdt)

    encoder = LabelEncoder()
    encoder.fit(Ydt)
    encoded_Ydt = encoder.transform(Ydt)

    # convert integers to dummy variables (i.e. one hot encoded)
    dummyYdt = encoded_Ydt  #np_utils.to_categorical(encoded_Ydt)

    print dummyYdt.shape
    return Xdt, Ydt

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

trainX, trainY = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv', [6,19,20])
testX, testY = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv', [6,19,20])

print trainX[0]
clf = RandomForestClassifier(n_estimators= 100, max_features=15, random_state=1)
clf.fit(trainX, trainY)

print clf.score(testX,testY)

predictions = clf.predict(testX)
print predictions
val_trues = testY
cm = metrics.confusion_matrix(val_trues, predictions, labels = ['anomaly', 'normal'])
print cm
tp, fn, fp, tn =cm[0][0], cm[0][1], cm[1][0], cm[1][1]
print tp
print 'accuracy ', (tp+tn)/(tp+tn+fp+fn+0.0)*100

# print("%s: %.2f%%" % ('accuracy',
# (tp + tn) / (tp + tn + fp + fn + 0.0) * 100))
#print("precision:  %.2f%%" % score[2])
#print("recall:  %.2f%%" % score[2])
print 'precision ', (tp) / (tp + fp + 0.0) * 100
print 'recall ', (tp) / (tp + fn + 0.0) * 100
print 'far ', fp / (fp + tn + 0.0) * 100
