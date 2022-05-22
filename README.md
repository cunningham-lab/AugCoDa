# AugCoDa
Code to reproduce the results in [Data Augmentation for Compositional Data: Advancing Predictive Models of the Microbiome](elliottgordonrodriguez.com).

To simply run the augmentations on some data:

```
def aitchison_mixup(X_train, y_train, factor)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y)

    for val in y_train.unique():
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n) - n

        lam = np.random.rand(n_aug).reshape([-1, 1])
        idx1 = np.random.choice(n, size=n_aug)
        idx2 = np.random.choice(n, size=n_aug)

        # Take convex combination
        X_aug = lam * X_temp[idx1, :] + (1 - lam) * X_temp[idx2, :]

        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])
    
    return X, y, w

def random_subcompositions(X_train, y_train, factor)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y)

    for val in [0, 1]:
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n) - n
        X_aug = []
        y_aug = []
        p = np.random.rand(n_aug)
        idx = np.random.choice(n, size= n_aug)
        mask = np.random.binomial(1, p, [X_temp.shape[1], n_aug]).T
        X_new = X_temp[idx, :].copy()
        X_new[mask.astype('bool')] = 1
        X_aug.append(X_new)
        y_aug.append(y_train[idx])
        X_aug = X_new
        y_aug = y_aug
        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    return X, y, w

def compositional_cutmix(X_train, y_train, factor)
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y)

    for val in [0, 1]:
        idxs = y_train == val
        X_temp = X_train[idxs, :]
        n = X_temp.shape[0]
        n_aug = int(factor * n - n

        idx1 = np.random.choice(n, size=n_aug)
        idx2 = np.random.choice(n, size=n_aug)

        p = np.random.rand(n_aug)
        mask = np.random.binomial(1, p, [X_temp.shape[1], n_aug]).T
        X_aug = mask * X_temp[idx1, :] + (1 - mask) * X_temp[idx2, :]

        X_aug = X_aug / X_aug.sum(axis=1, keepdims=True)

        X = np.concatenate([X, X_aug], axis=0)
        y = np.concatenate([y, np.repeat(val, n_aug)])
        w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    return X, y, w

```
