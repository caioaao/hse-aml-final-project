import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_f
from sklearn.base import (BaseEstimator, RegressorMixin)

from ..functional import pipe


class MLP(nn.Module):
    def __init__(self, n_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, n_features // 2)
        self.fc2 = nn.Linear(n_features // 2, n_features // 4)
        self.fc3 = nn.Linear(n_features // 4, 1)

    def forward(self, x):
        device = torch.device('cuda')
        return pipe(x,
                    lambda x: x.to(device, dtype=torch.float),
                    self.fc1,
                    nn_f.relu,
                    self.fc2,
                    nn_f.relu,
                    self.fc3)


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_features, n_epochs=2, batch_size=128):
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = 128

    def _predict(self, X):
        return self.net_(torch.from_numpy(X.to_dense)
                              .to('cuda', torch.float))

    def fit(self, X, y):
        if not hasattr(self, 'net_'):
            self.net_ = MLP(self.n_features).to('cuda')
            self.optimizer_ = optim.Adam(self.net_.parameters())
            self.criterion_ = nn.MSELoss()

        for epoch in range(self.n_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                self.optimizer_.zero_grad()
                output = self._predict(X[i:i + self.batch_size])
                target = torch.from_numpy(y[i:i + self.batch_size])\
                              .to('cuda', torch.float)
                loss = self.criterion_(output, target)
                loss.backward()
                self.optimizer_.step()
        return self

    def predict(self, X):
        return self._predict(X).to('cpu').numpy
