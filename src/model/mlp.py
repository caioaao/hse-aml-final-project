import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn_f
from sklearn.base import (BaseEstimator, RegressorMixin, MetaEstimatorMixin)
import torch
import numpy as np


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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class MLPRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    def __init__(self, lr=1e-14, n_epochs=100, batch_size=128):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr

    def _set_net(self, X):
        self.n_features_ = X.todense().shape[1]
        self.net_ = MLP(self.n_features_).to('cuda')
        self.n_epochs_fit_ = 0
        self.optimizer_ = optim.Adam(self.net_.parameters(), lr=self.lr)
        self.criterion_ = nn.MSELoss()

    def partial_fit(self, X, y):
        if not hasattr(self, 'net_'):
            self._set_net(X)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        epoch_id = self.n_epochs_fit_ + 1

        acc_loss = 0.0
        xtensor, ytensor = (torch.from_numpy(X.todense())
                                 .to('cuda', torch.float,
                                     non_blocking=True),
                            torch.from_numpy(y).to('cuda', torch.float,
                                                   non_blocking=True))

        loader = FastTensorDataLoader(
            xtensor, ytensor, batch_size=self.batch_size)
        for inp, target in loader:
            self.optimizer_.zero_grad()
            output = self.net_(inp)
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
        X = torch.from_numpy(X.todense()).to('cuda', torch.float)
        return self.net_(X).to('cpu').detach().numpy().reshape(-1)
