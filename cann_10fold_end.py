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
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

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
    classes = [0 if x == 'normal' else 1  for x in classes] #1 means anomaly
    return classes


def loadDataset(path, ignoreList=[]):
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

    text2hash(dt,cols,['flag','protocol_type'])

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

def newFeatureCalculator(dataset, n): # dataset = the unlabeled dataset, the number of classes-clusters to create

    kmeans = KMeans(n_clusters=n, random_state=0).fit(dataset)
    print kmeans.cluster_centers_
    #print kmeans.transform(dataset)[0:10] # In the new space, each dimension is the distance to the cluster centers. Note that even if X is sparse, the array returned by transform will typically be dense.
    #in transform i have the distance to each cluster center

    clusterIndex = kmeans.predict(dataset)
    clusterData = [[],[]]
    for i in range(len(dataset)):
        clusterData[clusterIndex[i]].append(dataset[i])

    print len(kmeans.transform(dataset)[0])
    transform = kmeans.transform(dataset)
    print "distanze dal centro con transform e calcolate manualmente"
    print kmeans.transform(dataset[0].reshape(1,-1))
    #print (dataset[0])
    print(numpy.linalg.norm(kmeans.cluster_centers_[1] - dataset[0]))
    nbrs0 = NearestNeighbors(n_neighbors=2).fit(clusterData[0])
    nbrs1 = NearestNeighbors(n_neighbors=2).fit(clusterData[1])
    # print distances[0:10][0:10]
    #distances, indices = nbrs.kneighbors(dataset)

    newFeature = [0] * len(dataset)
    for i in range(len(dataset)):
        for c in range(n):
            newFeature[i] += transform[i][c]
        #newFeature[i] += distances[i][1]
# print newFeature[0:10]
        if clusterIndex[i] == 0:
            dist,ind = nbrs0.kneighbors(dataset[i].reshape(1,-1))
        else:
            dist, ind = nbrs1.kneighbors(dataset[i].reshape(1, -1))
        #print "dist", dist;
        newFeature[i] += dist[0][1]
    return newFeature

ignoreFor6 = [
    0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 16, 18, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
]

ignoreFor19 = [0,2,4,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23,29,39,40]

xTrain, yTrain = loadDataset('./NSL-KDD-Dataset/KDDTrain+.csv', ignoreFor19) #todo with the removed features

#xTrain = newFeatureCalculator(xtr, 2)
print len(xTrain), len(yTrain)
print yTrain[0:20]


scores=[0]*25
#k between one and 25
"""for i in range(25):
    k=i+1
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(numpy.array(xTrain).reshape(-1,1), yTrain)
    score = neigh.score(numpy.array(xTest).reshape(-1,1),yTest)
    scores[i] = score
"""

print xTrain.shape

accuracy =[]
precision=[]
recall=[]
far=[]

yTrain = numpy.array(yTrain)

xTrain = newFeatureCalculator(xTrain,2)
xTrain = numpy.array(xTrain)

xte, yTest = loadDataset('./NSL-KDD-Dataset/KDDTest+.csv', ignoreFor19)
print yTest[0:10]
xTest = newFeatureCalculator(xte, 2)

kf = KFold(n_splits=10)
best = 0
bestModel = KNeighborsClassifier(n_neighbors=2)
for train_index, test_index in kf.split(xTrain):
    neigh = KNeighborsClassifier(n_neighbors=2)
    X_train, X_test = xTrain[train_index], xTrain[test_index]
    y_train, y_test = yTrain[train_index], yTrain[test_index]
    neigh.fit(numpy.array(X_train).reshape(-1, 1), y_train)
    s = neigh.score(numpy.array(X_test).reshape(-1,1),y_test)
    print 'score:' ,s

    if s > best:
        best = s
        bestModel = KNeighborsClassifier(n_neighbors=2)
        bestModel.fit(numpy.array(X_train).reshape(-1, 1), y_train)


    predictions = neigh.predict(numpy.array(X_test).reshape(-1, 1))
    val_trues = y_test
    cm = metrics.confusion_matrix(val_trues, predictions, labels = [1, 0])
    print cm
    tp, fn, fp, tn =cm.ravel()
    print tp, fp, fn , tn
    accuracy.append( (tp+tn)/(tp+tn+fp+fn+0.0)*100)
    precision.append(  (tp) / (tp + fp + 0.0) * 100)
    recall.append(  (tp) / (tp + fn + 0.0) * 100)
    far.append(  fp / (fp + tn + 0.0) * 100)

print "avg acc:",  reduce(lambda x, y: x + y, accuracy) / len(accuracy)
print "avg precision:",  reduce(lambda x, y: x + y, precision) / len(precision)
print "avg recall:",  reduce(lambda x, y: x + y, recall) / len(recall)
print "avg far:",  reduce(lambda x, y: x + y, far) / len(far)
#todo usare 10fold
#todo settare le varie feature (gruppo da 6 e da 19)
# todo valutare se puoi usare direttamente cklearn per calculare nn senza fartelo a mamno

#testing the best on the test data

xTest = numpy.array(xTest).reshape(-1, 1)
predictions = bestModel.predict(xTest)
val_trues = yTest
cm = metrics.confusion_matrix(
    val_trues, predictions, labels=[1, 0])
print cm
tp, fn, fp, tn = cm.ravel()
print tp, fp, fn, tn
print 'accuracy ', (tp + tn) / (tp + tn + fp + fn + 0.0) * 100
print 'precision ', (tp) / (tp + fp + 0.0) * 100
print 'recall ', (tp) / (tp + fn + 0.0) * 100
print 'far ', fp / (fp + tn + 0.0) * 100