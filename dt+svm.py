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
from sklearn import tree
import json
from c45 import C45

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

def text2hash(df,cols,toHash = ['service', 'flag','protocol_type']):
    df.columns = cols
    for el in toHash:
        df[el] = df[el].apply(
            lambda x: hashing_trick(x, 200, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ '))

def toJson(df):
    j={}
    df.columns = columnsHead
    for col in columnsHead:
        j[col] = df[col]
    return j

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


def loadDataset(path):
    #remove undesired features
    colNumbers = range(0, 42)
    # load the dataset
    dt = pandas.read_csv(
        path,#'../NSL-KDD-Dataset-master/KDDdt+.csv'
        usecols = colNumbers,
        engine='python',
        skipfooter=0)

    text2hash(dt,columnsHead)
    j = toJson(dt)

    dt = dt.values

    Xdt = dt[:, 0:41]
    Xdt = removeListValues(Xdt)
    Xdt = Xdt.astype('float32')
    Ydt = dt[:, 41]
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
    return Xdt, Ydt, j

def dt ():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    trainX, trainY = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv')
    testX, testY = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv')

    c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
    c1.fetchData()
    c1.preprocessData()
    c1.generateTree()
    c1.printTree()
    print tree

    clf = tree.DecisionTreeClassifier()
    clf.fit(trainX, trainY)
    tree.export_graphviz(clf, out_file='tree.dot')

    print clf.score(testX,testY)

    predictions = clf.predict(testX)
    pred = ['anomaly' for el in predictions]
    print predictions
    val_trues = testY
    cm = metrics.confusion_matrix(val_trues, pred, labels = ['anomaly', 'normal'])
    print cm
    tp, fn, fp, tn =cm.ravel()

    print tp, fp, fn , tn
    print 'accuracy ', (tp+tn)/(tp+tn+fp+fn+0.0)*100
    print 'precision ', (tp) / (tp + fp + 0.0) * 100
    print 'recall ', (tp) / (tp + fn + 0.0) * 100
    print 'far ', fp / (fp + tn + 0.0) * 100

    return predictions

dt()