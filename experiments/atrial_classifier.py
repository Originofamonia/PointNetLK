"""
train a simple classifier on registered point cloud with uni/bi-polar
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


def main():
    """
    implement leave one out cross validation
    """
    npzfile = np.load(f'saved_pt_10.npz')
    x0, x1, y0, y1 = npzfile['x0'], npzfile['x1'], npzfile['y0'], npzfile['y1']
    x0 = np.reshape(x0, (x0.shape[0], -1))  # [8， -1]
    x1 = np.reshape(x1, (x1.shape[0], -1))  # [8， -1]
    kfold = KFold(n_splits=8, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X=x0, y=y0)):
        x_train = x0[train_ids]
        y_train = y0[train_ids]
        x_test = x0[test_ids]
        y_test = y0[test_ids]
        print(y_test)

if __name__ == '__main__':
    main()
