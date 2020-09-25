import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_f
from sklearn.base import (BaseEstimator, RegressorMixin, MetaEstimatorMixin)
import torch


class MLP(nn.Module):
    def __init__(self, n_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features // 2)
        self.fc2 = nn.Linear(n_features // 2, n_features // 4)
        self.fc3 = nn.Linear(n_features // 4, 1)

    def forward(self, x):
        x = nn_f.relu(self.fc1(x))
        x = nn_f.relu(self.fc2(x))
        return self.fc3(x)


class MLPRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    def __init__(self, preprocessor=None, n_epochs=2, batch_size=128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.preprocessor = preprocessor

    def _predict(self, X):
        X = self.preprocessor.transform(X)
        return self.net_(torch.from_numpy(X.todense())
                              .to('cuda', torch.float))

    def fit(self, X, y):
        self.n_features_ = self.preprocessor.transform(X[0:1, :]).todense().shape[1]
        self.net_ = MLP(self.n_features_).to('cuda')
        self.optimizer_ = optim.Adam(self.net_.parameters())
        self.criterion_ = nn.MSELoss()
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        for epoch in range(self.n_epochs):
            acc_loss = 0.0
            for i in range(0, X.shape[0], self.batch_size):
                self.optimizer_.zero_grad()
                output = self._predict(X[i:i + self.batch_size])
                target = torch.from_numpy(y[i:i + self.batch_size])\
                              .to('cuda', torch.float)
                loss = self.criterion_(output, target)
                acc_loss = acc_loss + loss.item()
                loss.backward()
                self.optimizer_.step()
            print('Epoch %d, Acc loss: %.5f' % (epoch, acc_loss))
        return self

    def predict(self, X):
        return self._predict(X).to('cpu').numpy()
