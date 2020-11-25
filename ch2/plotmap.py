import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, ax, resolution=0.02):

    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap( colors[:len(np.unique(y))] )

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1
    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

    xx1, xx2 = np.meshgrid( np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution) )

    Z = classifier.predict(np.array( [xx1.ravel(), xx2.ravel()]).T )
    Z = Z.reshape(xx1.shape)
    
    ax.contourf( xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y==cl, 0], y=X[y==cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

if __name__=='__main__':

    from perceptron import Perceptron
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt

    dtpath = Path(__file__, '../iris.data').resolve(strict=True)
    df = pd.read_csv(str(dtpath), header=None)

    y = df.iloc[:100,4].values
    y = np.where(y=='Iris-setosa', -1, 1)
    X = df.iloc[:100,[0,2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    fig = plt.figure()
    ax = fig.add_subplot()
    plot_decision_regions(X, y, ppn, ax)

    plt.show()
