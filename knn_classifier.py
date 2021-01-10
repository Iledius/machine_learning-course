
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

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


def knn_classifier(x, trdata, trlabels, knn):
    distances = {}
    labeled = []
    for i in range(len(x)):
        distances[i] = []
        neighbors = []
        for j in range(len(trdata)):
            distances[i].append([euc_distance(x[i], trdata[j]), trlabels[j]])
        distances[i].sort()
        for k in range(knn):
            neighbors.append(distances[i][k][1])
        label = max(neighbors, key=neighbors.count)
        labeled.append(label)

    return labeled


k_list = [1, 2, 3, 5, 10, 20]
k_accuracies = []
for k in k_list:
    acc = pred_acc(knn_classifier(testdata, traindata, trainlabels, k), testlabels)
    k_accuracies.append(acc)
    print(f"{k}-NN accuracy:", acc)

plt.plot(k_list, k_accuracies)
plt.xlabel('K-nn')
plt.ylabel('accuracy')
plt.show()

# I used only my own old codes from the course exercises for inspiration for this exam exercise.
