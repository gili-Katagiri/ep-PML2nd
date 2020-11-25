import numpy as np

class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):

        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)

            errors = (y-output)

            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)

if __name__=='__main__':

    import matplotlib.pyplot as plt
    import pandas as pd
    from pathlib import Path

    from plotmap import plot_decision_regions as pdr

    dtpath = Path(__file__, '../iris.data').resolve(strict=True)
    df = pd.read_csv(str(dtpath), header=None)
    y = df.iloc[:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    X = df.iloc[:100, [0,2]].values

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
    ax[0,0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
    ax[0,0].set_xlabel('Epochs')
    ax[0,0].set_ylabel('log(Sum-squared_error)')
    ax[0,0].set_title('Adaline - Learning rate 0.1')
    ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
    ax[0,1].plot(range(1,len(ada2.cost_)+1), np.log10(ada2.cost_), marker='o')
    ax[0,1].set_xlabel('Epochs')
    ax[0,1].set_ylabel('log(Sum-squared_error)')
    ax[0,1].set_title('Adaline - Learning rate 0.0001')

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0]-X[:,0].mean()) / X[:,0].std()
    X_std[:, 1] = (X[:, 1]-X[:,1].mean()) / X[:,1].std()

    ada = AdalineGD(n_iter=15, eta=0.01)
    ada.fit(X_std, y)

    pdr(X_std, y, ada, ax[1,0])

    ax[1,0].set_title('Adaline - Gradient Descent')
    ax[1,0].set_xlabel('sepal length [cm]')
    ax[1,0].set_ylabel('petal length [cm]')
    ax[1,0].legend(loc='upper left')


    ax[1,1].plot(range(1,len(ada.cost_)+1), ada.cost_, marker='o')
    ax[1,1].set_xlabel('Epochs')
    ax[1,1].set_ylabel('Sum-squared_error')

    plt.tight_layout()
    plt.show()
