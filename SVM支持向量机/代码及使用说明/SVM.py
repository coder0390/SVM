from numpy import *
import time
import matplotlib.pyplot as plt

# 计算核函数
def calcKernelValue(matrix_x, sample_x, kernelOption):
    kernelType = kernelOption[0]    # 选择核函数
    numSamples = matrix_x.shape[0]
    kernelValue = mat(zeros((numSamples, 1)))

    # 对于线性可分的数据，使用线性核（自行设置）
    if kernelType == 'linear':
        kernelValue = matrix_x * sample_x.T
    # 否则，使用高斯核
    elif kernelType == 'rbf':
        sigma = kernelOption[1]     # 设置参数，参数过大容易过拟合
        if sigma == 0:
            sigma = 1.0
        for i in range(numSamples):
            diff = matrix_x[i, :] - sample_x
            kernelValue[i] = exp(diff * diff.T / (-2.0 * sigma ** 2))
    else:
        raise NameError('Not support kernel type! You can use linear or rbf!')
    return kernelValue

# 根据核函数计算得到矩阵
def calcKernelMatrix(train_x, kernelOption):
    numSamples = train_x.shape[0]
    kernelMatrix = mat(zeros((numSamples, numSamples)))
    for i in range(numSamples):
        kernelMatrix[:, i] = calcKernelValue(train_x, train_x[i, :], kernelOption)
    return kernelMatrix

# 数据结构SVMStruct，用于存储全局变量
class SVMStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption):
        self.train_x = dataSet  # 数据集
        self.train_y = labels   # 标签
        self.C = C              # 松弛变量，自行设置
        self.toler = toler      # 满足KKT条件的容忍值，自行设置
        self.numSamples = dataSet.shape[0]              # 样本数量
        self.alphas = mat(zeros((self.numSamples, 1)))  # 样本拉格朗日因子，数组维度：样本数量*1，初始值置0
        self.b = 0              # b初始值置0（顺便一提，可以不计算w）
        self.errorCache = mat(zeros((self.numSamples, 2)))  #这个数组存储了之前计算的E，方便调用，维度为：样本数量*2，因为需要标记是否已优化
        self.kernelOpt = kernelOption
        self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)

# 计算Ek
def calcError(svm, alpha_k):
    output_k = float(multiply(svm.alphas, svm.train_y).T * svm.kernelMat[:, alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k

# 更新Ek
def updateError(svm, alpha_k):
    error = calcError(svm, alpha_k)
    svm.errorCache[alpha_k] = [1, error]

# 选择提升最大的（Ej最大的）aj
def selectAlpha_j(svm, alpha_i, error_i):
    svm.errorCache[alpha_i] = [1, error_i]
    candidateAlphaList = nonzero(svm.errorCache[:, 0].A)[0]  # 候选alpha列表
    maxStep = 0
    alpha_j = 0
    error_j = 0

    # 选择提升最大的aj
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = calcError(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(random.uniform(0, svm.numSamples))
        error_j = calcError(svm, alpha_j)

    return alpha_j, error_j

# 优化 alpha i 和 alpha j 的内循环
def innerLoop(svm, alpha_i):
    error_i = calcError(svm, alpha_i)

    # 判断alpha i样本是否违反KKT条件（KKT条件见实验报告）
    if (svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or \
                    (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.alphas[alpha_i] > 0):

        # 步骤1：选择alpha j
        alpha_j, error_j = selectAlpha_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 步骤2：计算上下界H与L
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        # 步骤3：计算n
        eta = 2.0 * svm.kernelMat[alpha_i, alpha_j] - svm.kernelMat[alpha_i, alpha_i] \
              - svm.kernelMat[alpha_j, alpha_j]
        if eta >= 0:
            return 0

        # 步骤4：更新alpha j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j) / eta

        # 步骤5：根据上下界裁剪alpha j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 步骤6：判断是否收敛
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            updateError(svm, alpha_j)
            return 0

        # 步骤7：优化alpha i
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] \
                               * (alpha_j_old - svm.alphas[alpha_j])

        # 步骤8：更新b
        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_i] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) \
                               * svm.kernelMat[alpha_i, alpha_j] \
             - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) \
               * svm.kernelMat[alpha_j, alpha_j]
        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        # 步骤9：更新存储的E值，方便之后的调用
        updateError(svm, alpha_j)
        updateError(svm, alpha_i)

        return 1
    else:
        return 0

# SVM训练，缺省使用高斯核，参数1.0
def trainSVM(train_x, train_y, C, toler, maxIter, kernelOption=('rbf', 1.0)):
    # 记录训练时间
    startTime = time.time()

    # 初始化全局变量
    svm = SVMStruct(mat(train_x), mat(train_y), C, toler, kernelOption)

    # 开始训练
    entireSet = True
    alphaPairsChanged = 0
    iterCount = 0
    # 终止条件：达到最大训练次数/全部训练样本满足KKT条件
    while (iterCount < maxIter) and ((alphaPairsChanged > 0) or entireSet):
        alphaPairsChanged = 0

        # 遍历所有训练样本
        if entireSet:
            #内外循环寻找 alpha i 和 alpha j
            for i in range(svm.numSamples):
                alphaPairsChanged += innerLoop(svm, i)
            print('---iter:%d entire set, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1
        # 检查所有0<αi<C样本点
        else:
            nonBoundAlphasList = nonzero((svm.alphas.A > 0) * (svm.alphas.A < svm.C))[0]
            for i in nonBoundAlphasList:
                alphaPairsChanged += innerLoop(svm, i)
            print('---iter:%d non boundary, alpha pairs changed:%d' % (iterCount, alphaPairsChanged))
            iterCount += 1

        # 如果遍历了全部样本点，必须重新检查所有0<αi<C样本点；如果没有优化任何alpha，则遍历全部样本点
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True

    print('Congratulations, training complete! Took %fs!' % (time.time() - startTime))
    return svm

# SVM测试
def testSVM(svm, test_x, test_y):
    test_x = mat(test_x)
    test_y = mat(test_y)
    numTestSamples = test_x.shape[0]
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    supportVectors = svm.train_x[supportVectorsIndex]
    supportVectorLabels = svm.train_y[supportVectorsIndex]
    supportVectorAlphas = svm.alphas[supportVectorsIndex]
    matchCount = 0
    for i in range(numTestSamples):
        kernelValue = calcKernelValue(supportVectors, test_x[i, :], svm.kernelOpt)
        predict = kernelValue.T * multiply(supportVectorLabels, supportVectorAlphas) + svm.b
        if sign(predict) == sign(test_y[i]):
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    return accuracy


# 可视化展示，但只能支持二维线性可分的数据集
def showSVM(svm):
    if svm.train_x.shape[1] != 2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
        return 1

     # 画出样本点
    for i in range(svm.numSamples):
        if svm.train_y[i] == -1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'or')
        elif svm.train_y[i] == 1:
            plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'ob')

    # 标记支持向量
    supportVectorsIndex = nonzero(svm.alphas.A > 0)[0]
    for i in supportVectorsIndex:
        plt.plot(svm.train_x[i, 0], svm.train_x[i, 1], 'oy')

    # 画出超平面（只能绘制直线）
    w = zeros((2, 1))
    for i in supportVectorsIndex:
        w += multiply(svm.alphas[i] * svm.train_y[i], svm.train_x[i, :].T)
    min_x = min(svm.train_x[:, 0])[0, 0]
    max_x = max(svm.train_x[:, 0])[0, 0]
    y_min_x = float(-svm.b - w[0] * min_x) / w[1]
    y_max_x = float(-svm.b - w[0] * max_x) / w[1]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.show()