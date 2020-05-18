3.3.1创建新项目
创建新项目，添加python文件，SVM.py与test.py。
3.3.2 处理数据集
在test.py的同级目录下添加数据集文件，并修改test.py文件第8行fileIn = open('bloodTransfusion_noduplicated.txt')
打开文件。
Test.py文件第10行lineArr = line.strip().split(',')
可以根据数据集文件不同的分隔符读取数据（‘,’、’ ’、’\t’），请根据所选择的数据集修改分隔符。
Test.py文件第11、12行分别读取数据与标签：
dataSet.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), float(lineArr[3])])
labels.append(float(lineArr[4]))
请根据所选择的数据集的维度，修改代码。
Test.py文件第16、17、18、19行划分训练集与测试集：
train_x = dataSet[0:501, :]
train_y = labels[0:501, :]
test_x = dataSet[501:511, :]
test_y = labels[501:511, :]
同样，根据所选择的数据集情况自行划分即可。
3.3.3 训练
Test.py文件第16、17、18、19行设置训练参数：
C = 0.6
toler = 0.001
maxIter = 100
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('rbf', 20))