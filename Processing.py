from sklearn.linear_model import LinearRegression
from pandas import DataFrame


def featureExtract(xValues, yValues):
    operation = LinearRegression()
    operation.fit(xValues, yValues)
    result1 = 2 * operation.intercept_ / (1 + operation.intercept_ ** 2)
    result2 = (1 - operation.intercept_ ** 2) / (1 + operation.intercept_ ** 2)
    return result1, result2


def splitImage(image):
    L = image.numpy()
    B = [[] for _ in range(16)]
    for i in range(4):
        for j in range(4):
            for k in range(int(len(L[0]) / 4)):
                B[i * 4 + j].append(
                    L[int(len(L[0]) / 4) * i + k][int(len(L[0]) / 4) * j:int(len(L[0]) / 4) * (j + 1)].tolist())
    return B


def processItem(item):
    numberOfBlackPixels = 0
    X, Y = [], []
    for i in range(len(item)):
        for j in range(len(item)):
            if item[i][j] != 0:
                numberOfBlackPixels += 1
                X.append(i)
                Y.append(j)
    if len(X) > 0:
        df = DataFrame(X, columns=['xvalues'])
        df['yvalues'] = Y
        result2, result3 = featureExtract(df[['xvalues']], df.yvalues)
    else:
        result2, result3 = 0.0, 1.0
    return [numberOfBlackPixels / (len(item)) ** 2, result2, result3]
