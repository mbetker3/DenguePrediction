import numpy as np


def Sigmoid(z):
    return 1 / (1 + np.exp(-z))

class logisticRegression:
    def __init__(self, learningRate=0.1, numIterations=1000):
        self.learningRate = learningRate
        self.numIterations = numIterations
        self.W = None
        self.bias = 0.0


    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros(d)
        for i in range(0, self.numIterations):
            z = X @ self.W + self.bias
            yHat = Sigmoid(z)
            dw = (X.T @ (yHat- y))/n
            db = np.sum(yHat - y)/n
            self.W -= self.learningRate * dw
            self.bias -= self.learningRate * db
        return self

    def predictProb(self, X):
        return Sigmoid(X @ self.W+ self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predictProb(X) >= threshold).astype(int)



