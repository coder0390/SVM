from SVM import *
import SVM

# 数据处理
print("step 1: load data...")
dataSet = []
labels = []
fileIn = open('iris.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split(' ')
    dataSet.append([float(lineArr[0]),float(lineArr[1])])
    labels.append(float(lineArr[4]))

dataSet = mat(dataSet)
labels = mat(labels).T
train_x = dataSet[0:119, :]
train_y = labels[0:119, :]
test_x = dataSet[120:149, :]
test_y = labels[120:149, :]

# 训练
print("step 2: training...")
C = 1
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('linear', 1.0))
print("b:",svmClassifier.b)

# 测试
print("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

# 结果展示
print("step 4: show the result...")
print('The classify accuracy is: %.3f%%' % (accuracy * 100))
SVM.showSVM(svmClassifier)