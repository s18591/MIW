import numpy as np
import matplotlib.pylab as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from plotka import plot_decision_regions


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
    
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Multiclassifier(object):
    def __init__(self,eta=0.01, n_iter=10,num=2):
        self.eta = eta
        self.n_iter = n_iter
        self.ppn = []
        self.num = num
        for i in range(num-1):
            self.ppn.append(LogisticRegressionGD(eta=eta, n_iter=n_iter))

    def multiFit(self, X, Y, i):
        self.ppn[i].fit(X, Y)

    def predict(self, X):
        res = []
        for p in self.ppn:
            res.append(p.predict(X))
            print(p.activation(p.net_input(X)))
        res1 = res[0].copy()
        res1[(res1 == 1)] = 0
        for j in range(len(res1)):
            if res1[j] == -1:
                for l in range(len(res) - 1):
                    if not res[l+1][j] == -1:
                        res1[j] = res[l+1][j]
                        break
                if res1[j] == -1:
                    res1[j] = self.num-1
        return res1


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    mpp = Multiclassifier(eta=0.01, n_iter=2000, num=3)
    for i in range(len(iris.target_names) - 1):
        X1 = X_train.copy()
        Y1 = y_train.copy()
        X1 = X1[(Y1 == i) | (Y1 == i + 1)]
        Y1 = Y1[(Y1 == i) | (Y1 == i + 1)]
        Y1[(Y1 != i)] = -1
        Y1[(Y1 == i)] = 1
        Y1[(Y1 == -1)] = 0
        mpp.multiFit(X1, Y1, i)


    plot_decision_regions(X=X_test, y=y_test, classifier=mpp)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
