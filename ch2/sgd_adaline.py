from numpy.random import seed

class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):

        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):

        self._initialize_weight(X.shape[1])

        #rgen = np.random.RandomState(self.random_state)
        #self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []

            for xi, target in zip(X, y):
                cost.append(self._update_weight(xi, target))

            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weight(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X,y):
                self._update_weight(xi, target)
        else:
            self._update_weight(X,y)
            return self

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weight(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1+m)
        self.w_initialized = True

    def _update_weight(self, xi, target):

        output = self.activation(self.net_input(xi))

        error = (target-output)

        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error

        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X))>=0.0, 1, -1)

if __name__=='__main__':

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from pathlib import Path

    from plotmap import plot_decision_regions as pdr

    dtpath = Path(__file__, '../iris.data').resolve(strict=True)
    df = pd.read_csv(str(dtpath), header=None)
    y = df.iloc[:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    
    X = df.iloc[:100, [0,2]].values

    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0]-X[:,0].mean()) / X[:,0].std()
    X_std[:, 1] = (X[:, 1]-X[:,1].mean()) / X[:,1].std()

    ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
    ada.fit(X_std, y)

    fig, ax = plt.subplots( 1, 2, figsize=(10,6))
    pdr(X_std, y, ada, ax[0])

    ax[0].set_title('Adaline - Stochastic Gradient Descent')
    ax[0].set_xlabel('sepal length [cm]')
    ax[0].set_ylabel('petal length [cm]')
    ax[0].legend(loc='upper left')

    ax[1].plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Average Cost')

    plt.tight_layout()
    plt.show()
