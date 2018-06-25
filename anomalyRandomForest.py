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

outlierThreshold = 0.5 #to fix
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

    print dt.shape
    attackLables = dt[:,41]
    print attackLables

    if (isTrain):
        normalDT = numpy.array([el for el in dt if el[41] == 'normal'])
        dt = normalDT[:,:]

    Ydt = dt[:,1]
    print '--------'
    print Ydt
    print dt.shape
    Xdt = numpy.delete(dt,1,1)
    Xdt = removeListValues(Xdt)
    print Xdt.shape
    Xdt = numpy.delete(Xdt,len(Xdt[0])-1,1)
    print Xdt.shape
    Xdt = Xdt.astype('float32')

    #reshaping inputs as expected by lstm with time step size = 100
    #Xdt = numpy.reshape(XTrain, (len(XTrain), 100, len(XTrain[0])))
    #XTest = numpy.reshape(XTest, (len(XTest), 100, len(XTrain[0])))

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    Xdt = scaler.fit_transform(Xdt)

    return Xdt, Ydt, attackLables

def outliernessCount(i): #todo
    return i

def proximityCalculator(forest, n, items, predicted, expected): 
    proxMatrix = [[0 for x in range(n)] for x in range(n)]
    resMatrix = [[]for x in range(10)]
    z=0
    for tree in forest.estimators_ :
        resMatrix[z] = tree.apply(items)
        z +=1; 

    print resMatrix[0]
    for i in range(n):
        if predicted[i] == expected[i]:
            for j in range(n):
                for t in range(nTrees):
                    if resMatrix[t][i] == resMatrix[t][j]:
                        proxMatrix[i][j] += 1
                proxMatrix[i][j] = (proxMatrix[i][j]+0.0)/nTrees
    return proxMatrix

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

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
print ('---------------')
print proxMatrix[0]
print len(proxMatrix), len(proxMatrix[0])


for i in range(len(predictions)):
    if predictions[i] != val_trues[i]:
        labelPrediction[i] = 'anomaly'
    else:
        #code to count the outlierness
        out = outliernessCount(i)
        if out > outlierThreshold:
            labelPrediction[i] = 'anomaly'
        else:
            labelPrediction[i] = 'normal'
