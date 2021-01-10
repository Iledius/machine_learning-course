import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt

def unpickle(file):
    with open(file, 'rb') as f:
        picdict = pickle.load(f, encoding="latin1")
    return picdict

#datadict = unpickle('./cifar-10-batches-py/data_batch_1')
datadict = unpickle('./cifar-10-batches-py/test_batch')

X = datadict["data"]
Y = datadict["labels"]

labeldict = unpickle('./cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y = np.array(Y)

def cifar_classifier_random(x):
    return random.choice([1,2,3,4,5,6,7,8,9])


def class_acc(pred, gt):
    hits = 0
    for i in range(len(pred)):
        if pred[i-1] == gt[i-1]:
            hits += 1

    return hits/len(pred)

def euc_distance(data1, data2):
    d = 0.0
    for i in range(len(data1)):
        d += (int(data1[i-1])-int(data2[i-1]))**2
    return sqrt(d)

def cifar10_classifier_1nn(x,trdata,trlabels):
    distances = {}
    labeled = []
    for i in range(len(x)):
        distances[i] = []
        for j in range(len(trdata)):
            distances[i].append([euc_distance(x[i], trdata[j]), trlabels[j]])
        distances[i].sort()
        label = distances[i][0][1]
        labeled.append(label)
    return labeled

traindata = datadict["data"][:100]
testdata = datadict["data"][100:200]
trainlabels = Y[:100]
predicted_data = cifar10_classifier_1nn(testdata, traindata, trainlabels)

# check accuracy
print("\n1NN accuracy:")
print(class_acc(predicted_data, Y[100:200]))

