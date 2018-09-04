import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import sklearn.metrics as sklm
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from keras.preprocessing.text import hashing_trick
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers.embeddings import Embedding
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from statsmodels import robust

columnsHead = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
    'dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate',
    'dst_host_srv_rerror_rate','outcome'
]

outlierThreshold = 0 #to fix
nTrees = 10

def text2hash(df,cols):
    df.columns = cols

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


def loadDataset(path, isTrain, ignoreList=[]): #ingoreList must be ordered
    #remove undesired features
    colNumbers = range(0, 42)
    cols = columnsHead[:]
    ignoreList.sort()
    #remove the features you don't want to use
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
    #dt = dt[0:100,:]
    print dt.shape
    attackLables = dt[:,41]
    print attackLables

    if (isTrain):
        normalDT = np.array([el for el in dt if el[41] == 'normal'])
        dt = normalDT[:,:]

    Ydt = dt[:,1]
    print '--------'
    print Ydt
    print dt.shape
    #delete the column containint the services
    Xdt = np.delete(dt,1,1)
    Xdt = removeListValues(Xdt)
    print Xdt.shape
    #remove the 'attack' or 'normal' flag
    Xdt = np.delete(Xdt,len(Xdt[0])-1,1)
    print Xdt.shape
    Xdt = Xdt.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    Xdt = scaler.fit_transform(Xdt)

    return Xdt, Ydt, attackLables


def proximityCalculator(forest, n, items, predicted, expected):
    proxMatrix = [[0 for x in range(n)] for y in range(n)]
    resMatrix = [[]for x in range(10)]
    z=0
    for tree in forest.estimators_ :
        resMatrix[z] = tree.apply(items)
        z +=1;

    #print resMatrix[0]
    for i in range(n):
        if predicted[i] == expected[i]: #calculate only if the item has been correctly evaluated in terms of protocol type
            for j in range(n):
                for t in range(nTrees):
                    if resMatrix[t][i] == resMatrix[t][j]:
                        proxMatrix[i][j] += 1
                proxMatrix[i][j] = (proxMatrix[i][j]+0.0)/nTrees

    return proxMatrix

def outliernessCount(proximities, predicted, expected): #predicted is a list containing the labels given by the rf
    n = len(proximities[0])
    outlierness = [0 for x in range(n)]
    for i in range(n):
        if predicted[i] == expected[i]:
            k = predicted[i]; #k is the class the item i belongs to
            for j in range(n) :
                if k == predicted[j]:
                    outlierness[i] += pow(proximities[i][j],2)
            outlierness[i] = n/outlierness[i]

    d = {}
    for i in range(n):
        if predicted[i] == expected[i]:
            if predicted[i] in d:
                d[ predicted[i] ].append( outlierness[i] )
            else:
                d[ predicted[i] ] = [ outlierness[i] ]
    #print d
    print(outlierness[0:10])

    medians = {}
    medianDevs = {}
    for k in d:
        medians[k] = np.median(d[k])
        medianDevs[k] = robust.mad(d[k])
    print '-----------'
    print d['icmp']
    print medianDevs
    print '------'
    for i in range(n):
        if predicted[i] == expected[i]:
            outlierness[i] = (outlierness[i] - medians[ predicted[i] ]) / medianDevs[ predicted[i] ]

    print(outlierness[0:10])
    return outlierness

def scoreByThreshold(th, outlierness, labels, labelPrediction, predictions,val_trues):
    for i in range(len(labels)):
        if predictions[i] != val_trues[i]:
            labelPrediction[i] = 'anomaly'
        else:
            out = outlierness[i]
            if out > th:
                labelPrediction[i] = 'anomaly'
            else:
                labelPrediction[i] = 'normal'

# print (labelPrediction[0:50])

    cm = metrics.confusion_matrix(
        labels, labelPrediction, labels=['anomaly', 'normal'])
    print cm
    print '--------------', th, '----------------'
    tp, fn, fp, tn =cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    acc= (tp+tn)/(tp+tn+fp+fn+0.0)*100
    prec= (tp) / (tp + fp + 0.0) * 100
    recall=(tp) / (tp + fn + 0.0) * 100
    far =fp / (fp + tn + 0.0) * 100
    print 'accuracy ',acc
    print 'precision ',prec
    print 'recall ', recall
    print 'far ', far
    return acc, far

def thresholdFinder():
    seed = 7
    np.random.seed(seed)

    trainX, trainY, labels = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv',True)
    testX, testY, labels = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv', False) #label are the label in terms of attack

    labels = m2Binary(labels)

    clf = RandomForestClassifier(n_estimators= nTrees, max_features=5, random_state=1)
    clf.fit(trainX, trainY)

    print clf.score(testX,testY) #classification in terms of protocol type
    predictions = clf.predict(testX)

    val_trues = testY #trues in terms of protocol types
    print len(predictions), len(val_trues), len(labels)

    labelPrediction = ['normal' for x in range(len(labels))]

    proxMatrix = proximityCalculator(clf, len(labels), testX, predictions, val_trues)
    #with open('./proxMatrix.json', 'wb') as outfile:
    #    json.dump(proxMatrix, outfile)

    #with open('./proxMatrix.json') as f:
    #    proxMatrix = json.load(f)

    print ('---------------')
    print proxMatrix[1][1]
    print proxMatrix[2][2]
    print len(proxMatrix), len(proxMatrix[0])


    outlierness = outliernessCount(proxMatrix, predictions, val_trues)


    scaler = MinMaxScaler(feature_range=(0, 1))
    print np.any(np.isnan(np.array(outlierness) ))
    print np.all(np.isfinite(np.array(outlierness) ))
    #scaled_out = np.array(outlierness).reshape(-1, 1)
    #out = scaler.fit_transform( scaled_out )
    accs = {}
    fars = {}
    print min(outlierness), max(outlierness)
    for i in np.arange(min(outlierness),max(outlierness),0.5):
        accs[i], fars[i] = scoreByThreshold( i, outlierness,labels,labelPrediction, predictions, val_trues)

    maxAcc = max(accs, key = accs.get)
    maxFar = min(fars, key = fars.get)
    print maxAcc , ' ' , accs[maxAcc]
    print maxFar , ' ' , fars[maxFar]


def predictByThreshold(th, outlierness, labelPrediction, predictions, val_trues):
    for i in range(len(outlierness)):
        if predictions[i] != val_trues[i]:
            labelPrediction[i] = 'anomaly'
        else:
            out = outlierness[i]
            if out > th:
                labelPrediction[i] = 'anomaly'
            else:
                labelPrediction[i] = 'normal'

    return labelPrediction


def train(n_trees = nTrees, mtry=5):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    trainX, trainY, labels = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv',True)

    clf = RandomForestClassifier(n_estimators= n_trees, max_features=mtry, random_state=1)
    clf.fit(trainX, trainY)
    return clf

def test(clf, testX=[], testY=[]):
    if len(testX) == 0:
        testX, testY, labels = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv', False) #label are the label in terms of attack

    print clf.score(testX,testY) #classification in terms of protocol type
    predictions = clf.predict(testX) #in terms of protocol type

    val_trues = testY #trues in terms of protocol types


    labelPrediction = ['normal' for x in range(len(predictions))]

    proxMatrix = proximityCalculator(clf, len(predictions), testX, predictions, val_trues)
    #with open('./proxMatrix.json', 'wb') as outfile:
    #    json.dump(proxMatrix, outfile)

    #with open('./proxMatrix.json') as f:
    #    proxMatrix = json.load(f)

    print ('---------------')
    print proxMatrix[1][1]
    print proxMatrix[2][2]
    print len(proxMatrix), len(proxMatrix[0])


    outlierness = outliernessCount(proxMatrix, predictions, val_trues)

    sortedOut=outlierness[:]
    sortedOut.sort()
    t = len(sortedOut)/8 #this changes the threshold

    anomalyLabels = predictByThreshold(sortedOut[ len(sortedOut) - (t + 1) ], outlierness,
                                       labelPrediction, predictions, val_trues)
    return anomalyLabels
