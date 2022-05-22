from sklearn.base import BaseEstimator,TransformerMixin

import numpy as np

class AugmenterAM(BaseEstimator, TransformerMixin):
    # See https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
    def __init__(self, nover2, factor=10):
        self.nover2 = nover2
        self.factor = factor # data augmentation factor

    def fit( self, X, y=None):
        return self 

    def transform( self, X, y=None):
        return X

    def fit_transform(self, X, y):

        if y is None:
            return X

        X_bal = X.copy()
        y_bal = y.copy()
        for i in range(2, self.factor):
            X_bal = np.concatenate([X_bal, X.copy()], axis=0)
            y_bal = np.concatenate([y_bal, y.copy()])

        # Check that we are operating on the train fold
        if X.shape[0] > self.nover2:
            for val in [0, 1]:
                idxs = y == val
                X_temp = X[idxs, :]
                n = X_temp.shape[0]
                n_aug = int(self.factor * n) - n
                lam = np.random.rand(n_aug).reshape([-1, 1])
                idx1 = np.random.choice(n, size=n_aug)
                idx2 = np.random.choice(n, size=n_aug)
                # Take convex combination
                X_aug = lam * X_temp[idx1, :] + (1 - lam) * X_temp[idx2, :]

                X = np.concatenate([X, X_aug], axis=0)
                y = np.concatenate([y, np.repeat(val, n_aug)])
            X = np.concatenate([X, X_bal], axis=0)
            y = np.concatenate([y, y_bal])
            return X, y
        return self.transform(X, y)

class AugmenterRS(BaseEstimator, TransformerMixin):
    # See https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
    def __init__(self, nover2, factor=10):
        self.nover2 = nover2
        self.factor = factor # data augmentation factor

    def fit( self, X, y=None):
        return self 

    def transform( self, X, y=None):
        return X

    def fit_transform(self, X, y):
        if y is None:
            return X

        X_bal = X.copy()
        y_bal = y.copy()
        for _ in range(2, self.factor):
            X_bal = np.concatenate([X_bal, X.copy()], axis=0)
            y_bal = np.concatenate([y_bal, y.copy()])

        # Check that we are operating on the train fold
        if X.shape[0] > self.nover2:
            for val in [0, 1]:
                idxs = y == val
                X_temp = X[idxs, :]
                n = X_temp.shape[0]
                n_aug = int(self.factor * n) - n
                p = np.random.rand(n_aug)
                idx = np.random.choice(n, size=n_aug)
                mask = np.random.binomial(1, p, [X_temp.shape[1], n_aug]).T
                X_new = X_temp[idx, :].copy()
                X_min = X_new.min(axis=1, keepdims=True)
                X_min = X_min * np.ones_like(X_new)
                X_new[mask.astype('bool')] = X_min[mask.astype('bool')]
                X_new = X_new / X_new.sum(axis=1, keepdims=True)
                X = np.concatenate([X, X_new], axis=0)
                y = np.concatenate([y, np.repeat(val, n_aug)])
            X = np.concatenate([X, X_bal], axis=0)
            y = np.concatenate([y, y_bal])
            return X, y
        return self.transform(X, y)

class AugmenterSM(BaseEstimator, TransformerMixin):
    # See https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
    def __init__(self, nover2, factor=10):
        self.nover2 = nover2
        self.factor = factor # data augmentation factor

    def fit( self, X, y=None):
        return self 

    def transform( self, X, y=None):
        return X

    def fit_transform(self, X, y):
        if y is None:
            return X

        X_bal = X.copy()
        y_bal = y.copy()
        for _ in range(2, self.factor):
            X_bal = np.concatenate([X_bal, X.copy()], axis=0)
            y_bal = np.concatenate([y_bal, y.copy()])

        # Check that we are operating on the train fold
        if X.shape[0] > self.nover2:
            for val in [0, 1]:
                idxs = y == val
                X_temp = X[idxs, :]
                n = X_temp.shape[0]
                n_aug = int(self.factor * n) - n
                p = np.random.rand(n_aug)
                idx1 = np.random.choice(n, size=n_aug)
                idx2 = np.random.choice(n, size=n_aug)
                mask = np.random.binomial(1, p, [X_temp.shape[1], n_aug]).T
                X_aug = mask * X_temp[idx1, :] + (1 - mask) * X_temp[idx2, :]
                X_aug = X_aug / X_aug.sum(axis=1, keepdims=True)
                X = np.concatenate([X, X_aug], axis=0)
                y = np.concatenate([y, np.repeat(val, n_aug)])
            X = np.concatenate([X, X_bal], axis=0)
            y = np.concatenate([y, y_bal])
            return X, y
        return self.transform(X, y)
        