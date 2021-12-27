"""
train a simple classifier on registered point cloud with uni/bi-polar
"""

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np


def main():
    """
    implement leave one out cross validation
    """
    npzfile = np.load(f'saved_pt_10.npz')
    x0, x1, y0, y1 = npzfile['x0'], npzfile['x1'], npzfile['y0'], npzfile['y1']
    x0 = np.reshape(x0, (x0.shape[0], -1, 4))[:, :, -1]  # [8， -1]
    x1 = np.reshape(x1, (x1.shape[0], -1, 4))[:, :, -1]  # [8， -1]
    y0 = y0 - 1  # values of 1, 2 -> 0, 1
    kfold = KFold(n_splits=8, shuffle=True)

    preds = []
    labels = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X=x0, y=y0)):
        x = x0
        y = y0
        # svm_classifier(labels, preds, x, y, train_ids, test_ids)
        lr_classifier(labels, preds, x, y, train_ids, test_ids)

    acc = accuracy_score(labels, preds)
    print(f'preds: {preds}; labels: {labels}')
    print(f'acc: {acc}')


def svm_classifier(labels, preds, x, y, train_ids, test_ids):
    x_train = x[train_ids]
    y_train = y[train_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(x_train, y_train)
    preds.append(clf.predict(x_test)[0])
    labels.append(y_test[0])


def lr_classifier(labels, preds, x0, y0, train_ids, test_ids):
    """
    logistic regression classifier
    """
    x_train = x0[train_ids]
    y_train = y0[train_ids]
    x_test = x0[test_ids]
    y_test = y0[test_ids]
    clf = LogisticRegression(random_state=444).fit(x_train, y_train)
    preds.append(clf.predict(x_test)[0])
    labels.append(y_test[0])


if __name__ == '__main__':
    main()
