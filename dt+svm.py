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
from keras.models import model_from_json
from sklearn import tree
from sklearn import svm
import json

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


def text2hash(df, cols, toHash=['service', 'flag', 'protocol_type']):
    df.columns = cols
    for el in toHash:
        df[el] = df[el].apply(
            lambda x: hashing_trick(x, 200, hash_function='md5', filters='!"#$%&()*+,-./:;<=>?@[\]^`{|}~ '))


def toJson(df):
    j = {}
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
    classes = [x if x == 'normal' else 'anomaly' for x in classes]
    return classes


def loadDataset(path):
    #remove undesired features
    colNumbers = range(0, 42)
    # load the dataset
    dt = pandas.read_csv(
        path,  #'../NSL-KDD-Dataset-master/KDDdt+.csv'
        usecols=colNumbers,
        engine='python',
        skipfooter=0)

    text2hash(dt, columnsHead)

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
    return Xdt, Ydt


def hybrid(v, g):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    trainX, trainY = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv')
    testX, testY = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv')

    d = {}
    split = {}

    clf = tree.DecisionTreeClassifier(min_samples_leaf=0.1)
    clf.fit(trainX, trainY)
    tree.export_graphviz(clf, out_file='tree.dot')
    paths = clf.decision_path(trainX)
    predictions = clf.predict(trainX)
    leaves = clf.apply(trainX)

    for i in range(len(predictions)):
        if predictions[i] == 'normal':
            l = leaves[i]
            if not (l in d):
                d[l] = {}
                #d[l]['classes'] = set([predictions[i]])
                if trainY[i] == 'normal':
                    split[l] = {}
                    split[l]['data'] = []
                    split[l]['data'].append(trainX[i])
                    split[l]['classifier'] = svm.OneClassSVM(
                        nu=v, kernel="rbf", gamma=g)
            else:
                #d[l]['leaves'].add(leaves[i])
                #d[l]['classes'].add(predictions[i])
                if trainY[i] == 'normal':
                    if not (l in split):
                        split[l] = {}
                        split[l]['data'] = []
                        split[l]['classifier'] = svm.OneClassSVM(
                            nu=v, kernel="rbf", gamma=g)
                    split[l]['data'].append(trainX[i])
        """
        path = paths[i] #returns the decision path, the last element is the leaf
        print "node: " + str(numpy.where(path.toarray() == 1)[1][-2]) +" leaf: " + str(clf.apply(testX[i].reshape(1,-1))) + " class: " + predictions[i]
    
    for k in d:
        if len(d[k]['leaves']) > 1 or len(d[k]['classes']) > 1:
            print '...........',k, d[k], '............'
    """

    for k in split:
        print k, len(split[k]['data'])
        split[k]['classifier'].fit(split[k]['data'])

#    print clf.score(testX, testY)

    testPred = clf.predict(testX)
    check = testPred[:]
    testLeaves = clf.apply(testX)
    for i in range(len(testPred)):
        if testPred[i] == 'normal':
            l = split[testLeaves[i]]['classifier'].predict(testX[i].reshape(1, -1))
            if l == -1:
                testPred[i] = 'anomaly'


    print check == testPred

    #pred = ['anomaly' for el in predictions]
    #predictions = clf.predict(testX)
    print testPred
    val_trues = testY
    cm = metrics.confusion_matrix(
        val_trues, testPred, labels=['anomaly', 'normal'])
    print cm
    tp, fn, fp, tn = cm.ravel()

    print tp, fp, fn, tn
    print 'accuracy ', (tp + tn) / (tp + tn + fp + fn + 0.0) * 100
    print 'precision ', (tp) / (tp + fp + 0.0) * 100
    print 'recall ', (tp) / (tp + fn + 0.0) * 100
    print 'far ', fp / (fp + tn + 0.0) * 100

    return testPred


hybrid(0.5, 0.01)
