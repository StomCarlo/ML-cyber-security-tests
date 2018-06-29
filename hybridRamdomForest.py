import misuseRandomForest as mrf
import anomalyRandomForest as arf
import numpy as np
from sklearn import metrics

misusePrediction = mrf.misuse(100,15,[6,19,20])

testX, testY, labels = arf.loadDataset(
    './NSL-KDD-Dataset/KDDTest+.csv',
    False)  #label are the label in terms of attack

labels = mrf.m2Binary(labels)

X = []
Y = []
indices = []
for i in range( len(misusePrediction) ) :
    if misusePrediction[i] == 'normal':
        X.append(testX[i])
        Y.append(testY[i])
        indices.append(i) #array to store the original positions of the items classyfoed as normal by the misuse rf

anomalyRF = arf.train()
anomalyLabels = arf.test(anomalyRF, np.array(X),np.array(Y))

for j in range( len (indices) ):
    misusePrediction[ indices[j] ] = anomalyLabels[j]

cm = metrics.confusion_matrix(
        labels, misusePrediction, labels=['anomaly', 'normal'])

print cm
tp, fn, fp, tn =cm[0][0], cm[0][1], cm[1][0], cm[1][1]
acc= (tp+tn)/(tp+tn+fp+fn+0.0)*100
prec= (tp) / (tp + fp + 0.0) * 100
recall=(tp) / (tp + fn + 0.0) * 100
far =fp / (fp + tn + 0.0) * 100
print 'accuracy ',acc
print 'precision ',prec
print 'recall ', recall
print 'far ', far