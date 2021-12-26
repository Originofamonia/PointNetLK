"""
train a simple classifier on registered point cloud with uni/bi-polar
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np


def main():
    """
    implement leave one out cross validation
    """
    npzfile = np.load(f'saved_pt_10.npz')
    x0, x1, y0, y1 = npzfile['x0'], npzfile['x1'], npzfile['y0'], npzfile['y1']
    x0 = np.reshape(x0, (x0.shape[0], -1))
    x1 = np.reshape(x1, (x1.shape[0], -1))
    print(x0)


if __name__ == '__main__':
    main()
