import numpy as np
from math import sqrt

testdata = np.loadtxt('X_test.txt')
testlabels = np.loadtxt('Y_test.txt')
traindata = np.loadtxt('X_train.txt')
trainlabels = np.loadtxt('Y_train.txt')


def pred_acc(pred, gt):
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


def classifier_1nn(x, trdata, trlabels):
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


predicted_data = classifier_1nn(testdata, traindata, trainlabels)

print("\n1NN accuracy:")
print(pred_acc(predicted_data, testlabels))

# I used only my own old codes from the course exercises for inspiration to this exam exercise.
