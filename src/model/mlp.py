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

    def _set_net(self, X):
        self.n_features_ = self.preprocessor.transform(X[0:1, :]).todense()\
                                                                 .shape[1]
        self.net_ = MLP(self.n_features_).to('cuda')
        self.n_epochs_fit_ = 0
        self.optimizer_ = optim.Adam(self.net_.parameters())
        self.criterion_ = nn.MSELoss()

    def partial_fit(self,  X, y):
        if not hasattr(self, 'net_'):
            self._set_net(X)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        epoch_id = self.n_epochs_fit_ + 1

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
        self.n_epochs_fit_ = epoch_id
        print('Epoch %d, Acc loss: %.5f' % (epoch_id, acc_loss))
        return self

    def fit(self, X, y):
        self._set_net(X)
        for epoch in range(self.n_epochs):
            self.partial_fit(X, y)
        return self

    def predict(self, X):
        return self._predict(X).to('cpu').detach().numpy().reshape(-1)
