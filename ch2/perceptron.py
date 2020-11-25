import numpy as np

class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        # learning rate
        self.eta = eta
        # number of iterate
        self.n_iter = n_iter
        # random seed: reproducibility
        self.random_state = random_state

    def fit(self, X, y):
        # set initial weights
        rgen = np.random.RandomState(self.random_state)
        self.weight_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        # classification errors
        self.errors_ = []
        for _ in range(self.n_iter):
            cnt = 0
            for xrow, teacher in zip(X,y):
                # calc delta_weight and update
                update = self.eta * (teacher-self.predict(xrow))
                self.weight_[1:] += update*xrow
                self.weight_[0] += update
                # count errors
                cnt+=int(update != 0.0)
            
            self.errors_.append(cnt)
        return self

    # cover X*w_colvec and x_rowvec*w_colvec
    def matrix_product(self, X):
        return self.weight_[0] + np.dot(X, self.weight_[1:]) 

    def predict(self, X):
        return np.where( self.matrix_product(X)>=0.0, 1, -1 )

if __name__=='__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    from pathlib import Path

    dtpath = Path(__file__, '../iris.data').resolve(strict=True)
    df = pd.read_csv( str(dtpath), header=None)

    y = df.iloc[:100, 4].values
    y = np.where(y=='Iris-setosa', -1, 1)

    X = df.iloc[0:100, [0,2]].values
    
    # plot
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(1,2,1)

    ax1.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='setosa')
    ax1.scatter(X[50:, 0], X[50:,1], color='blue', marker='x', label='versicolor')

    ax1.set_xlabel('sepal length [cm]')
    ax1.set_ylabel('petal length [cm]')
    
    ax1.legend()
    

    # train
    prcp = Perceptron(eta=0.1, n_iter=10)
    prcp.fit(X, y)

    ax2 = fig.add_subplot(1,2,2)
    ax2.plot(range(1, len(prcp.errors_)+1), prcp.errors_, marker='o')

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Number of update')
    
    plt.tight_layout()

    plt.show()
