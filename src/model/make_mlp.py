import sys

import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_f
from sklearn.base import (BaseEstimator, RegressorMixin, MetaEstimatorMixin)
import pandas as pd

from ..data import df_to_X_y
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


class MLPRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    def __init__(self, n_features, preprocessor=None, n_epochs=2,
                 batch_size=128):
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = 128
        self.preprocessor = preprocessor

    def _predict(self, X):
        X = self.preprocessor.transform(X)
        return self.net_(torch.from_numpy(X.to_dense)
                              .to('cuda', torch.float))

    def fit(self, X, y):
        if not hasattr(self, 'net_'):
            self.net_ = MLP(self.n_features).to('cuda')
            self.optimizer_ = optim.Adam(self.net_.parameters())
            self.criterion_ = nn.MSELoss()

        for epoch in range(self.n_epochs):
            acc_loss = 0.0
            for i in range(0, X.shape[0], self.batch_size):
                self.optimizer_.zero_grad()
                output = self._predict(X[i:i + self.batch_size])
                target = torch.from_numpy(y[i:i + self.batch_size])\
                              .to('cuda', torch.float)
                loss = self.criterion_(output, target)
                acc_loss = acc_loss + loss.item()
                if i % 2000 == 0:
                    print('Epoch %d, Acc loss: %.5f' % (epoch, acc_loss))
                loss.backward()
                self.optimizer_.step()
        return self

    def predict(self, X):
        return self._predict(X).to('cpu').numpy


if __name__ == '__main__':
    train_set = pd.read_parquet(sys.argv[1])
    preprocessor = joblib.load(sys.argv[2])
    output_path = sys.argv[3]

    print('Loading dataset')
    X, y = df_to_X_y(train_set)
    del train_set

    print('Building final estimator')

    mlp = MLPRegressor(preprocessor=preprocessor, n_features=X.shape[0])

    print('Fitting final estimator')
    mlp.fit(X, y)
    joblib.dump(mlp, output_path)
