#coding:utf-8
import numpy as np

"""
逻辑回归，python实现
"""

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000, print_cost=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.print_cost = print_cost

    def _initialize_parameters(self, n_features):
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.sqrt(n_features)
        np.random.seed(111)
        weights = np.random.uniform(-limit, limit, (n_features,))

        return weights

    def fit(self, X, Y):
        n_samples, n_features = X.shape

        # 初始化网络参数
        self.weights = self._initialize_parameters(n_features)
        self.bias = 0

        for iter in range(self.n_iters):
            # 前向传播过程
            Z = np.dot(X, self.weights) + self.bias
            A = self._sigmoid(Z)

            # 损失函数
            cost = - (1 / n_samples) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
            if self.print_cost and iter%100 == 0:
                print("iter: {}, cost: {}".format(iter, cost))

            # 反向传播计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (A - Y))
            db = (1 / n_samples) * np.sum(A - Y)

            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(Z)

        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # 获取训练数据
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # LR训练过程
    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000, print_cost=True)
    regressor.fit(X_train, y_train)

    # LR测试过程
    predictions = regressor.predict(X_test)

    print("LR classification accuracy:", accuracy(y_test, predictions))


