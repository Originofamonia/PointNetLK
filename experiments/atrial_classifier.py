"""
train a simple classifier on registered point cloud with uni/bi-polar
"""
import os
import sys
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import VotingClassifier
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)))

import ptlk

os.environ['CUDA_VISIBLE_DIVICES'] = '1'


class MLP(nn.Module):
    """
    MLP classifier for unipolar or bipolar
    """
    def __init__(self, input_dim=406, dropout=0.4):
        super(MLP, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # nn.Linear(300, 300),
            # nn.Dropout(p=dropout),
            # nn.LeakyReLU(),
            nn.Linear(400, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)


class MLPEnsemble(nn.Module):
    """
    ensemble 2 MLP for unipolar + bipolar
    """
    def __init__(self, input_dim=406, dropout=0.4):
        super(MLPEnsemble, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # nn.Linear(300, 300),
            # nn.Dropout(p=dropout),
            # nn.LeakyReLU(),
            nn.Linear(400, 2),
            # nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            # nn.Linear(300, 300),
            # nn.Dropout(p=dropout),
            # nn.LeakyReLU(),
            nn.Linear(400, 2),
            # nn.Sigmoid()
        )

    def forward(self, x1, x2):
        return self.layer1(x1) + self.layer2(x2)


class RNN(nn.Module):
    def __init__(self, input_dim=306, dropout=0.3):
        super(RNN, self).__init__()
        # self.args = args
        self.rnn = nn.GRU(input_size=1, hidden_size=300,
                          num_layers=2, batch_first=True)
        self.fc = nn.Linear(300, 2)

    def forward(self, x):
        # input size : (batch, seq_len, input_size)
        out, h_n = self.rnn(x)
        out = F.relu(torch.max(out, dim=1)[0])
        out = F.softmax(self.fc(out), dim=-1)
        return out


def train_mlp(labels, preds, x, y, train_ids, test_ids):
    epochs = 30
    device = 'cuda'
    x_train = x[train_ids]
    y_train = y[train_ids]
    x_test = x[test_ids]
    y_test = y[test_ids]
    train_data = ptlk.data.datasets.Voltages(x=x_train, y=y_train)
    test_data = ptlk.data.datasets.Voltages(x=x_test, y=y_test)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1, shuffle=False, num_workers=2)

    model = MLP(input_dim=1006).to(device)  # input_dim=812 for ensemble
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(epochs):
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            data = tuple(item.to(device) for item in data)
            x, y = data
            # x = x.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss.item():.3f}')

    model.eval()
    for i, data in enumerate(test_loader):
        data = tuple(item.to(device) for item in data)
        x, y = data
        # x = x.unsqueeze(-1)
        outputs = model(x)
        pred_cls = outputs.data.max(1)[1].item()
        preds.append(pred_cls)
        labels.append(y[0].item())


def train_ensemble_mlp(labels, preds, x1, x2, y, train_ids, test_ids):
    epochs = 30
    device = 'cuda'
    x1_train = x1[train_ids]
    y_train = y[train_ids]
    x1_test = x1[test_ids]
    y_test = y[test_ids]
    x2_train = x2[train_ids]
    x2_test = x2[test_ids]
    train_data = ptlk.data.datasets.EnsembleVoltages(x1=x1_train, x2=x2_train, y=y_train)
    test_data = ptlk.data.datasets.EnsembleVoltages(x1=x1_test, x2=x2_test, y=y_test)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1, shuffle=False, num_workers=2)

    model = MLPEnsemble().to(device)  # input_dim=812 for ensemble
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(epochs):
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            data = tuple(item.to(device) for item in data)
            x1, x2, y = data
            # x = x.unsqueeze(-1)
            optimizer.zero_grad()
            outputs = model(x1, x2)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss.item():.3f}')

    model.eval()
    for i, data in enumerate(test_loader):
        data = tuple(item.to(device) for item in data)
        x1, x2,  y = data
        # x = x.unsqueeze(-1)
        outputs = model(x1, x2)
        pred_cls = outputs.data.max(1)[1].item()
        preds.append(pred_cls)
        labels.append(y[0].item())


def reorder_by_distance(x):
    """
    reorder points by their coordinates L1 distance to original point and return
    reordered voltages
    """
    coords = x[:, :, 0: 3]  # [B, N, 3]
    voltages = x[:, :, -1]
    abs_coords = np.abs(coords)
    l1_d = np.sum(abs_coords, axis=-1)  # [B, N]
    sorted_indices = np.argsort(l1_d, axis=-1)  # [B, N]
    sorted_voltages = []
    for i, row in enumerate(sorted_indices):
        sorted_voltages.append(voltages[i][row])
    return np.asarray(sorted_voltages)


def main():
    """
    implement leave one out cross validation
    """
    npzfile = np.load(f'saved_pt_10.npz', allow_pickle=True)
    x0, x1, y0, y1 = npzfile['x0'], npzfile['x1'], npzfile['y0'], npzfile['y1']
    x0 = np.reshape(x0, (x0.shape[0], -1, 4))  # [:, :, -1]  # [8， -1]
    x1 = np.reshape(x1, (x1.shape[0], -1, 4))  # [:, :, -1]  # [8， -1]
    y0 = 2 - y0  # values of 1, 2 -> 1, 0, AF type
    y1 = 1 - y1  # 1Y re AF
    # add reorder by L1 distance before classifier
    x0 = reorder_by_distance(x0)  # unipolar
    x1 = reorder_by_distance(x1)  # bipolar
    x_cat = np.concatenate((x0, x1), axis=-1)  # ensemble
    kfold = KFold(n_splits=8, shuffle=False)  # LOOCV

    preds = []
    labels = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X=x1, y=y0)):
        x = x1
        y = y0
        # svm_classifier(labels, preds, x, y, train_ids, test_ids)
        # lr_classifier(labels, preds, x, y, train_ids, test_ids)
        train_mlp(labels, preds, x, y, train_ids, test_ids)
        # train_ensemble_mlp(labels, preds, x0, x1, y, train_ids, test_ids)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    p, r, th = precision_recall_curve(labels, preds)
    pr_auc = auc(r, p)
    roc_auc = roc_auc_score(labels, preds)
    print(f'preds: {preds}; labels: {labels}')
    print(f'acc: {acc}, f1: {f1:.3f}, roc_auc: {roc_auc:.3f}, pr_auc: {pr_auc:.3f}, '
          f'precision: {precision:.3f}, recall: {recall:.3f}')


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


def plot_data_y0():
    """
    plot data & labels
    """
    npzfile = np.load(f'saved_pt_10.npz', allow_pickle=True)
    x0, x1, y0, y1 = npzfile['x0'], npzfile['x1'], npzfile['y0'], npzfile['y1']
    x0 = np.reshape(x0, (x0.shape[0], -1, 4))  # [:, :, -1]  # [8， -1]
    x1 = np.reshape(x1, (x1.shape[0], -1, 4))  # [:, :, -1]  # [8， -1]
    voltage0 = x0[:, :, -1]
    voltage1 = x1[:, :, -1]
    for i in range(len(voltage0)):
        fig = plt.figure(figsize=(8, 8))
        ax0 = fig.add_subplot(211)  # 211: row, col, index
        ax1 = fig.add_subplot(212)

        ax0.plot(voltage0[i], c='b', label=f'x0, y0: {y0[i]}; y1: {y1[i]}')
        ax1.plot(voltage1[i], c='r', label=f'x1, y0: {y0[i]}; y1: {y1[i]}')
        # fig.xlabel(f'{desc}')
        ax0.legend()
        ax1.legend()
        plt.savefig(f'voltage_{i}.jpg')
        plt.close(fig)


if __name__ == '__main__':
    main()
    # plot_data_y0()
