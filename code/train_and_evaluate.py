
# %%

import os
import numpy as np
import scipy
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, \
    RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
# from sklearn.calibration import calibration_curve

import argparse

import tensorflow as tf
from tensorflow import keras

from xgboost import XGBClassifier

from maml.code.sklearn_pipeline import SklearnPipeline
from augmenter import AugmenterAM, AugmenterRS, AugmenterSM # augmenters for maml

from DeepMicro.DM import DeepMicrobiome

# %%
# HELPER FUNCTIONS

def clr(X):
    return np.log(X) - np.mean(np.log(X), axis=1, keepdims=True)

# Taken from sklearn metrics and adapted to compute ECE
# as per https://github.com/scikit-learn/scikit-learn/issues/18268
def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    normalize="deprecated",
    n_bins=5,
    strategy="uniform",
):
    """Compute true and predicted probabilities for a calibration curve.
    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.
    Calibration curves may also be referred to as reliability diagrams.
    Read more in the :ref:`User Guide <calibration>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.
    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.
    pos_label : int or str, default=None
        The label of the positive class.
        .. versionadded:: 1.1
    normalize : bool, default="deprecated"
        Whether y_prob needs to be normalized into the [0, 1] interval, i.e.
        is not a proper probability. If True, the smallest value in y_prob
        is linearly mapped onto 0 and the largest one onto 1.
        .. deprecated:: 1.1
            The normalize argument is deprecated in v1.1 and will be removed in v1.3.
            Explicitly normalizing `y_prob` will reproduce this behavior, but it is
            recommended that a proper probability is used (i.e. a classifier's
            `predict_proba` positive class).
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.
        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.
    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).
    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.
    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9,  1.])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    # y_true = column_or_1d(y_true)
    # y_prob = column_or_1d(y_prob)
    # check_consistent_length(y_true, y_prob)
    # pos_label = _check_pos_label_consistency(pos_label, y_true)

    # TODO(1.3): Remove normalize conditional block.
    if normalize != "deprecated":
        warnings.warn(
            "The normalize argument is deprecated in v1.1 and will be removed in v1.3."
            " Explicitly normalizing y_prob will reproduce this behavior, but it is"
            " recommended that a proper probability is used (i.e. a classifier's"
            " `predict_proba` positive class or `decision_function` output calibrated"
            " with `CalibratedClassifierCV`).",
            FutureWarning,
        )
        if normalize:  # Normalize predicted values into interval [0, 1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    # y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    # Change sklearn func as per https://github.com/scikit-learn/scikit-learn/issues/18268
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
    return prob_true, prob_pred, ece
    # return prob_true, prob_pred

def augment_X(X_train, y_train, params):
    X = X_train.copy()
    y = y_train.copy()
    w = np.ones_like(y)

    if params == {}:
        return X, y, w
    
    if 'weight' in params:
        weight = params['weight']
    else:
        weight = params['factor'] / (1 + params['factor'])

    # Random subcompositions
    if params.get('subc') == True:
        for val in [0, 1]:
            idxs = y_train == val
            X_temp = X_train[idxs, :]
            n = X_temp.shape[0]
            n_large = 10 * n
            n_aug = int(params['factor'] * n) - n
            X_aug = []
            y_aug = []
            p = np.random.rand(n_large)
            idx = np.random.choice(n, size=n_large)
            mask = np.random.binomial(1, p, [X_temp.shape[1], n_large]).T
            X_new = X_temp[idx, :].copy()
            X_new[mask.astype('bool')] = 1
            X_aug.append(X_new)
            y_aug.append(y_train[idx])
            X_aug = X_new[0:n_aug]
            y_aug = y_aug[0:n_aug]
            X = np.concatenate([X, X_aug], axis=0)
            y = np.concatenate([y, np.repeat(val, n_aug)])
            w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    # Multinomial resampling
    if params.get('mult') == True:
        for val in [0, 1]:
            idxs = y_train == val
            X_temp = X_train[idxs, :]
            n = X_temp.shape[0]
            n_large = 10 * n
            n_aug = int(params['factor'] * n) - n
            X_aug = []
            y_aug = []
            for i in range(n_large):
                idx = np.random.choice(n)
                counts = X_temp[idx, :] - 1
                X_aug.append(np.random.multinomial(counts.sum(), counts / counts.sum()))
                y_aug.append(y_train[idx])
            X_aug = X_aug[0:n_aug]
            y_aug = y_aug[0:n_aug]
            X = np.concatenate([X, np.array(X_aug) + 1], axis=0)
            y = np.concatenate([y, np.repeat(val, n_aug)])
            w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    if params['space'] == 'clr':
        X_train = clr(X_train)
        X = clr(X)
    elif params['space'] == 'prop':
        X_train_sums = X_train.sum(axis=1, keepdims=True)
        X_train = X_train / X_train_sums
        X_sums = X.sum(axis=1, keepdims=True)
        X = X / X_sums
    
    # Aitchison mixup
    if params.get('conv') == 'rand':
        for val in [0, 1]:
            idxs = y_train == val
            X_temp = X_train[idxs, :]
            n = X_temp.shape[0]
            n_large = 10 * n
            n_aug = int(params['factor'] * n) - n

            lam = np.random.rand(n_large).reshape([-1, 1])
            idx1 = np.random.choice(n, size=n_large)
            idx2 = np.random.choice(n, size=n_large)

            # Take convex combination
            X_aug = lam * X_temp[idx1, :] + (1 - lam) * X_temp[idx2, :]

            X = np.concatenate([X, X_aug[0:n_aug]], axis=0)
            y = np.concatenate([y, np.repeat(val, n_aug)])
            w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    # Compositional CutMix
    if params.get('comb') == 'rand':
        for val in [0, 1]:
            idxs = y_train == val
            X_temp = X_train[idxs, :]
            n = X_temp.shape[0]
            n_large = 10 * n
            n_aug = int(params['factor']) * n - n

            idx1 = np.random.choice(n, size=n_large)
            idx2 = np.random.choice(n, size=n_large)

            p = np.random.rand(n_large)
            mask = np.random.binomial(1, p, [X_temp.shape[1], n_large]).T
            X_aug = mask * X_temp[idx1, :] + (1 - mask) * X_temp[idx2, :]

            # If in clr space we must mean center each observation to 
            # ensure a valid composition
            if params['space'] == 'clr':
                X_aug = X_aug - X_aug.sum(axis=1, keepdims=True)
            if params['space'] == 'prop':
                X_aug = X_aug / X_aug.sum(axis=1, keepdims=True)

            X = np.concatenate([X, X_aug[0:n_aug]], axis=0)
            y = np.concatenate([y, np.repeat(val, n_aug)])
            w = np.concatenate([w, np.repeat(weight / (1 - weight) * X_train.shape[0] / n_aug, n_aug)])

    if params['space'] == 'clr':
        X = scipy.special.softmax(X, axis=1)

    return X, y, w

def transform_X(X_tr, X_te, param):

    if param.get('space') == 'prop':
        X_tr = X_tr / X_tr.sum(axis=1, keepdims=True)
        X_te = X_te / X_te.sum(axis=1, keepdims=True)

    return X_tr, X_te


def dim_reduction(X_train_aug, X_test, dr_params):
    if dr_params == {}:
        return X_train_aug.copy(), X_test.copy()
    
    if 'PCs' in dr_params:
        PCs = dr_params['PCs']
        pca = PCA()
        pca.fit(X_train_aug)
        X_train_dr = pca.transform(X_train_aug)
        X_test_dr = pca.transform(X_test)
        return X_train_dr[:, :PCs], X_test_dr[:, :PCs]


def evaluate_classifier(X_train, y_train, w, X_test, y_test, params):

    start_time = time.time()

    if params['model'] == 'svm':

        model = SGDClassifier(loss='hinge')
        model.fit(X_train, y_train, sample_weight=w)
        y_pred = 1 / (1 + np.exp(-model.decision_function(X_test)))
        
    if params['model'] == 'ridge':
        # Can use liblinear or saga solvers
        model = RidgeClassifier()
        model.fit(X_train, y_train, sample_weight=w)
        y_pred = 1 / (1 + np.exp(-model.decision_function(X_test)))

    if params['model'] == 'ridgecv':
        # Can use liblinear or saga solvers
        model = RidgeClassifierCV()
        model.fit(X_train, y_train, sample_weight=w)
        y_pred = 1 / (1 + np.exp(-model.decision_function(X_test)))

    if params['model'] == 'rf':
        model = RandomForestClassifier(**rf_par) 
        model.fit(X_train, y_train, sample_weight=w)
        y_pred = model.predict_proba(X_test)[:, 1]

    if params['model'] == 'xgb':
        num_rounds = 100
        model = XGBClassifier(random_state=seed, n_estimators=num_rounds)
        model.fit(X_train, y_train, sample_weight=w)
        y_pred = model.predict_proba(X_test)[:, 1]

    if params['model'] == 'metann':
        tf.random.set_seed(tf_seed)

        METRICS = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
        ]

        def make_model_metann(metrics=METRICS, output_bias=None):
            if output_bias is not None:
                output_bias = tf.keras.initializers.Constant(output_bias)

            sequential = []
            sequential.append(keras.layers.Dense(512))
            sequential.append(keras.layers.Dropout(0.5))
            sequential.append(keras.layers.ReLU())
            sequential.append(keras.layers.Dense(256))
            sequential.append(keras.layers.ReLU())
            sequential.append(keras.layers.Dropout(0.5))
            sequential.append(keras.layers.Dense(1, activation='sigmoid'))
            model = keras.Sequential(sequential)


            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.005),
                loss=keras.losses.BinaryCrossentropy(),
                metrics=metrics)

            return model

        EPOCHS = 200

        BATCH_SIZE = 32

        weighted_model = make_model_metann()

        weighted_model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            # The class weights go here
            # class_weight=class_weight
            sample_weight=w,
        )

        y_pred = weighted_model.predict(X_test).flatten()

    if params['model'] == 'maml':
        # Note that incorporating a data augmentation into maml is tricky,
        # One needs to add a couple of hacks to the Pipeline class in sklearn
        # See https://stackoverflow.com/questions/25539311/custom-transformer-for-sklearn-pipeline-that-alters-both-x-and-y
        from maml.code.sklearn_pipeline_config import SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS

        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        y_train_df = pd.DataFrame(y_train.astype('str'))
        tmp_dir = "tmp/d" + str(data_idx) + "s" + str(np_seed)
        skp = SklearnPipeline(tmp_dir, X_train_df, y_train_df)
        skp.filter_low_prevalence_features()
        skp.over_sampling() # in mAML oversampling is used instead of weights

        if params.get('aug') == 'aitch':
            augmenter = AugmenterAM(nover2=X_train.shape[0]/2)
        elif params.get('aug') == 'subc':
            augmenter = AugmenterRS(nover2=X_train.shape[0]/2)
        elif params.get('aug') == 'comb':
            augmenter = AugmenterSM(nover2=X_train.shape[0]/2)
        else:
            augmenter = None

        All_CLASSIFIERS = Tree_based_CLASSIFIERS + Other_CLASSIFIERS
        skp.select_best_scl_clf(SCALERS, Tree_based_CLASSIFIERS, Other_CLASSIFIERS, augmenter, n_jobs=8)
        skp.hypertune_best_classifier(All_CLASSIFIERS, n_jobs=8)
        if hasattr(skp.best_estimator_, "decision_function"):
            y_pred = skp.best_estimator_.decision_function(X_test_df)
            y_pred = 1 / (1 + np.exp(-y_pred))
        elif hasattr(skp.best_estimator_, "predict_proba"):
            y_pred = skp.best_estimator_.predict_proba(X_test_df)[:, 1]
        print(time.time() - start_time)
        print(y_test)
        print(y_pred)

    if params['model'] == 'deepcoda':
        from keras import backend as K

        tf.random.set_seed(tf_seed)
        epochs = 200
        if 'cl' in params:
            cascade_level = params['cl']
        else:
            cascade_level = 5
        bottle_dim = 1
        if 'hd' in params:
            hidden_dim = params['hd']
        else:
            hidden_dim = 16
        output_dim = 1
        batch_size = 32

        if 'nl' in params:
            num_layers = params['nl']
        else:
            num_layers = 1

        # regularize sum of weights at each cascade to be 0 and regularize weights to be sparse
        class SumZeroL1Reg(keras.regularizers.Regularizer):
            def __init__(self, sumzero_lambda=1e0, l1_lambda=1e-2):
                self.sumzero_lambda = K.cast_to_floatx(sumzero_lambda)
                self.l1_lambda = K.cast_to_floatx(l1_lambda)

            def __call__(self, w):
                sumzero_reg = 0
                sumzero_reg += self.sumzero_lambda * K.square(K.sum(w))

                l1_reg = 0
                l1_reg += self.l1_lambda * K.sum(K.abs(w))

                return sumzero_reg + l1_reg

            def get_config(self):
                return {'sumzero_lambda': float(self.sumzero_lambda),
                        'l1_lambda': float(self.l1_lambda)}


        x = keras.Input(shape=(X_train.shape[1],))
        # concat layer for all z
        concat_z = []
        for _ in range(cascade_level):
            x_log = keras.layers.Lambda(lambda t: K.log(t))(x)
        # if use_weight_constraint == True:
            b = keras.layers.Dense(bottle_dim, activation='linear',
                    kernel_regularizer=SumZeroL1Reg())(x_log)
        # else:
        #     b = Dense(bottle_dim, activation='linear')(x_log)
            z = b
            concat_z.append(z)
        if cascade_level == 1:
            all_z = z
        else:
            all_z = keras.layers.Concatenate()(concat_z)
        h = keras.layers.Dense(hidden_dim, activation='relu')(all_z)
        for i in range(1, num_layers):
            h = keras.layers.Dense(hidden_dim, activation='relu')(h)
        beta = keras.layers.Dense(cascade_level, activation='linear')(h)
        # Decoder
        all_z_beta = keras.layers.Dot(axes=1)([all_z, beta])
        decoder = keras.Sequential([keras.layers.Dense(output_dim, input_dim=output_dim, activation='sigmoid')])
        y_pred = decoder(all_z_beta)
        # train network
        model = keras.Model(inputs=x, outputs=y_pred, name='bottleneck_model')
        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, shuffle=True, sample_weight=w, epochs=epochs, batch_size=batch_size, verbose=0)

        y_pred = model.predict(X_test)

    if params['model'] == 'deepmicro':
        tf.random.set_seed(tf_seed)
        dm = DeepMicrobiome("", seed, "")
        dm.X_train = X_train
        dm.y_train = y_train
        dm.X_test = X_test

        dm.ae([256, 128, 64], epochs=2000, patience=20)
        dm.X_train = dm.encoder.predict(X_train)
        dm.y_train = y_train
        dm.y_test = y_test
        
        # Initialization & wd
        lreg = LogisticRegressionCV(penalty='l2', Cs=50)
        z_train = dm.encoder.predict(X_train)
        lreg.fit(z_train, y_train, sample_weight=w)
        l2 = 1 / (2 * lreg.C_[0] * z_train.shape[0])
        if params['head'] == 'lp':
            y_pred = lreg.predict_proba(dm.encoder.predict(X_test))[:, 1]

        if params['head'] == 'ft':
            inputs = keras.Input(shape=(X_train.shape[1],))
            h = dm.encoder(inputs)
            head = keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(l2),  activation='sigmoid')
            outputs = head(h)
            head.set_weights([lreg.coef_.reshape([-1, 1]), lreg.intercept_])
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)

    if params['model'] == 'contrastive':
        from simclr.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
        from simclr.helpers import get_negative_mask
        tf.random.set_seed(tf_seed)

        BATCH_SIZE = X_train.shape[0]

        class CustomAugment(object):
            def __call__(self, sample, idx1=None, idx2=None):        
                # Random flips
                sample = self._random_apply(self._rs, sample, p=0.8)
                
                # Randomly apply transformation (color distortions) with probability p.
                # sample = self._random_apply(self._color_jitter, sample, p=0.8)
                # sample = self._random_apply(self._color_drop, sample, p=0.2)

                return sample

            def _rs(self, x, s=1):
                # one can also shuffle the order of following augmentations
                # each time they are applied.
                n, d = x.shape
                p = np.random.rand(n)
                mask = np.random.binomial(1, p, [d, n]).T
                X_min = tf.reduce_min(x, axis=1, keepdims=True)
                X_min = X_min * tf.ones_like(x)
                x = tf.where(mask, X_min, x)
                # x[mask.astype('bool')] = X_min[mask.astype('bool')]
                x = x / tf.reduce_sum(x, axis=1, keepdims=True)
                return x
            
            def _color_drop(self, x):
                x = tf.image.rgb_to_grayscale(x)
                x = tf.tile(x, [1, 1, 1, 3])
                return x
            
            def _random_apply(self, func, x, p):
                return tf.cond(
                tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                        tf.cast(p, tf.float32)),
                lambda: func(x),
                lambda: x)

        def _rs(x, idx1, idx2):
            # one can also shuffle the order of following augmentations
            # each time they are applied.
            n, d = x.shape
            p = np.random.rand(n)
            mask = np.random.binomial(1, p, [d, n]).T
            X_min = tf.reduce_min(x, axis=1, keepdims=True)
            X_min = X_min * tf.ones_like(x)
            x = tf.where(mask, X_min, x)
            # x[mask.astype('bool')] = X_min[mask.astype('bool')]
            x = x / tf.reduce_sum(x, axis=1, keepdims=True)
            return x

        def _am(x, idx1, idx2):
            # if np.random.rand() > 0.8:
            #     return tf.gather(x, idx1)
            n = len(idx1)
            x = tf.math.log(x) - tf.reduce_mean(tf.math.log(x), axis=1, keepdims=True)
            x1 = tf.gather(x, idx1)
            x2 = tf.gather(x, idx2)
            p = np.random.rand(n, 1)
            x = p * x1 + (1 - p) * x2
            x = tf.math.softmax(x)
            return x

        def _cc(x, idx1, idx2):
            # if np.random.rand() > 0.8:
            #     return tf.gather(x, idx1)
            n = len(idx1)
            x1 = tf.gather(x, idx1)
            x2 = tf.gather(x, idx2)
            p = np.random.rand(n)
            mask = np.random.binomial(1, p, [x.shape[1], n]).T
            mask = tf.cast(mask, tf.float64)
            x = mask * x1 + (1 - mask) * x2
            x = x / tf.reduce_sum(x, axis=1, keepdims=True)
            return x
        
        def _all(x, idx1, idx2):
            u = np.random.rand() 
            if u < 0.25:
                return tf.gather(x, idx1)
            if u < 0.5:
                return _rs(tf.gather(x, idx1), idx1, idx2)
            if u < 0.75:
                return _am(x, idx1, idx2)
            return _cc(x, idx1, idx2)

        # Build the augmentation pipeline
        # if params['aug'] == 'rs':
        data_augmentation = keras.Sequential([keras.layers.Lambda(CustomAugment())])
        if params.get('aug') == 'am':
            data_augmentation = _am
        if params.get('aug') == 'cc':
            data_augmentation = _cc
        if params.get('aug') == 'all':
            data_augmentation = _all

        inputs = keras.Input(shape=(X_train.shape[1],))
        #[256, 128, 64]
        feature_extractor = keras.models.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64)
        ])
        h = feature_extractor(inputs)

        projection1 = keras.layers.Dense(32, activation='relu')(h)
        projection2 = keras.layers.Dense(16)(projection1)

        model = keras.Model(inputs=inputs, outputs=projection2)

        # @tf.function
        def train_step(xis, xjs, model, optimizer, criterion, temperature):
            with tf.GradientTape() as tape:
                n = xis.shape[0]
                # Mask to remove positive examples from the batch of negative samples
                negative_mask = get_negative_mask(n)

                zis = model(xis)
                zjs = model(xjs)

                # normalize projection feature vectors
                zis = tf.math.l2_normalize(zis, axis=1)
                zjs = tf.math.l2_normalize(zjs, axis=1)

                l_pos = sim_func_dim1(zis, zjs)
                l_pos = tf.reshape(l_pos, (n, 1))
                l_pos /= temperature

                negatives = tf.concat([zjs, zis], axis=0)

                loss = 0

                for positives in [zis, zjs]:
                    l_neg = sim_func_dim2(positives, negatives)

                    labels = tf.zeros(n, dtype=tf.int32)

                    l_neg = tf.boolean_mask(l_neg, negative_mask)
                    l_neg = tf.reshape(l_neg, (n, -1))
                    l_neg /= temperature

                    logits = tf.concat([l_pos, l_neg], axis=1) 
                    loss += criterion(y_pred=logits, y_true=labels)

                loss = loss / (2 * n)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss

        def train_simclr(model, dataset, optimizer, criterion,
                        temperature=0.1, epochs=100):
            step_wise_loss = []
            epoch_wise_loss = []

            for epoch in range(epochs):
                epoch_loss = []
                for image_batch in dataset:
                    n = image_batch.shape[0]
                    idx = np.random.choice(np.arange(n), size=n, replace=False)
                    idx1 = idx[0:int(n/2)]
                    idx2 = idx[int(n/2):(2*int(n/2))]
                    a = data_augmentation(image_batch, idx1, idx2)
                    b = data_augmentation(image_batch, idx1, idx2)
                    # a = data_augmentation(image_batch)
                    # b = data_augmentation(image_batch)

                    loss = train_step(a, b, model, optimizer, criterion, temperature)
                    step_wise_loss.append(loss)
                    epoch_loss.append(loss)

                epoch_wise_loss.append(np.mean(step_wise_loss))
                # wandb.log({"nt_xentloss": np.mean(step_wise_loss)})
                
                if epoch % 10 == 0:
                    print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))
                    print(np.mean(epoch_loss))

            return epoch_wise_loss, model

        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                        reduction=tf.keras.losses.Reduction.SUM)
        decay_steps = 1000
        lr_decayed_fn = tf.keras.experimental.CosineDecay(
            initial_learning_rate=0.1, decay_steps=decay_steps)
        optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        train_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(BATCH_SIZE)
        
        epochs = 2000
        # Hack to run faster on the large (but trivial) dataset - will score 100% anyway
        if X_train.shape[0] > 1000:
            epochs = 100
        # If we're using random initialization, skip contrastive pretraining
        if params.get('init') != 'rand':
            _, model = train_simclr(model, train_ds, optimizer, criterion,
                            temperature=0.1, epochs=epochs)
        
        # classification head
        lreg = LogisticRegressionCV(penalty='l2', Cs=50)
        z_train = feature_extractor(X_train)
        z_test = feature_extractor(X_test)
        lreg.fit(z_train.numpy(), y_train, sample_weight=w)
        l2 = 1 / (2 * lreg.C_[0] * z_train.shape[0])        
        if params['head'] == 'lp':
            y_pred = lreg.predict_proba(z_test.numpy())[:, 1]
        
        if params['head'] == 'ft':
            inputs = keras.Input(shape=(X_train.shape[1],))
            h = feature_extractor(inputs)
            head = keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(l2), activation='sigmoid')
            outputs = head(h)
            head.set_weights([lreg.coef_.reshape([-1, 1]), lreg.intercept_])
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)
            y_pred = model.predict(X_test)

    end_time = time.time()

    acc_bl = np.mean(np.concatenate([y_train, y_test]))
    acc_bl = max(acc_bl, 1 - acc_bl)
    y_pred_bin = np.round(y_pred)
    acc = accuracy_score(y_test, y_pred_bin)
    bacc = balanced_accuracy_score(y_test, y_pred_bin)
    auc = roc_auc_score(y_test, y_pred)
    _, _, ece = calibration_curve(y_test.flatten(), y_pred.flatten())

    res = {
        'acc_bl': [acc_bl],
        'acc': [acc],
        'bacc': [bacc],
        'auc': [auc],
        'ece': [ece],
        'runtime': [end_time - start_time],
    }

    return res

# %%
if __name__ == '__main__':

    # Set up params
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_idx', dest='data_idx', type=int, default=0)
    parser.add_argument('--seed', dest='seed', type=int, default=0)
    parser.add_argument('--method', dest='method', type=str, default='fast')

    args = parser.parse_args()

    data_idx = args.data_idx
    np_seed = args.seed
    method = args.method
    np.random.seed(np_seed)

    seed = 0
    tf_seed = 0
    train_size = 0.8
    rf_par = {'n_estimators': 500, 'n_jobs': 8}

    run_params = [
        {
            'aug_params': [
                {},                
                {'conv': 'rand', 'space': 'clr', 'weight': 0.5, 'factor': 10},
                {'subc': True, 'space': '', 'weight': 0.5, 'factor': 10},
                {'comb': 'rand', 'space': 'prop', 'weight': 0.5, 'factor': 10},
            ],
            'tran_params': [
                {'space': 'prop'},
            ],
            'dr_params': [
                {},
            ],
            'head_params': [
                {'model': 'ridge'},
                {'model': 'svm'},
                {'model': 'rf'},
                {'model': 'xgb'},
                {'model': 'deepcoda'},
                {'model': 'metann'},
            ],
        },
    ]

    if method == 'deepmicro':
        run_params = [
            {
                'aug_params': [
                    {},
                ],
                'dr_params': [
                    {},
                ],
            'tran_params': [
                {'space': 'prop'},
            ],
                'head_params': [
                    {'model': 'deepmicro', 'head': 'lp'},
                    {'model': 'deepmicro', 'head': 'ft'},
                ],
            },
        ]

    if method == 'contrastive':
        run_params = [
            {
                'aug_params': [
                    {},
                ],
                'dr_params': [
                    {},
                ],
            'tran_params': [
                {'space': 'prop'},
            ],
                'head_params': [
                    {'model': 'contrastive', 'head': 'lp'},
                    {'model': 'contrastive', 'head': 'ft'},
                    {'model': 'contrastive', 'head': 'lp', 'init': 'rand'},
                    {'model': 'contrastive', 'head': 'ft', 'init': 'rand'},
                    {'model': 'contrastive', 'head': 'lp', 'aug': 'am'},
                    {'model': 'contrastive', 'head': 'ft', 'aug': 'am'},
                    {'model': 'contrastive', 'head': 'lp', 'aug': 'cc'},
                    {'model': 'contrastive', 'head': 'ft', 'aug': 'cc'},
                    {'model': 'contrastive', 'head': 'lp', 'aug': 'all'},
                    {'model': 'contrastive', 'head': 'ft', 'aug': 'all'},
                ],
            },
        ]

    if method == 'maml':
        run_params = [
            {
                'aug_params': [
                    {},
                ],
                'dr_params': [
                    {},
                ],
                'tran_params': [
                    {'space': 'prop'},
                ],
                'head_params': [
                    {'model': 'maml'},
                    {'model': 'maml', 'aug': 'aitch'},
                    {'model': 'maml', 'aug': 'subc'},
                    {'model': 'maml', 'aug': 'comb'},
                ],
            },
        ]

    # Load data
    data_list = pd.read_csv('./code/mlrepo12.csv', header=None)
    data_name = data_list.iloc[data_idx, 0]
    data_dir = './in/'
    X_df = pd.read_csv(
        data_dir + data_name + '-x.csv',
        index_col=0,
    )
    y_df = pd.read_csv(
        data_dir + data_name + '-y.csv',
        index_col=0,
    )

    # Remove redundant variables
    X_df = X_df.loc[:, X_df.std(axis=0) > 0]

    # Convert to numpy
    X = X_df.to_numpy()
    # If y has more than one column we just keep the response variable
    if y_df.shape[1] > 1:
        y = y_df['Var']
    else:
        y = y_df.iloc[:, 0]
    y = pd.get_dummies(y, drop_first=True).to_numpy().flatten()

    out = pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_size, stratify=y
    )

    for params in run_params:
        print(params)

        for aug_params in params['aug_params']:
            aug_seed = 0
            np.random.seed(aug_seed)
            X_train_aug, y_train_aug, w = augment_X(X_train, y_train, aug_params)

            classes = np.unique(y_train_aug)
            tally = [np.sum(y_train_aug == i) for i in classes]
            weights = np.max(tally) / tally
            weights = weights[y_train_aug.astype(int)]
            w = w * weights

            for tran_param in params['tran_params']:
                X_train_tr, X_test_tr = transform_X(X_train_aug, X_test, tran_param)

                for dr_params in params['dr_params']:
                    # Not used in this work
                    X_train_dr, X_test_dr = dim_reduction(X_train_tr, X_test_tr, dr_params)

                    for head_params in params['head_params']:
                        res = evaluate_classifier(X_train_dr, y_train_aug, w, X_test_dr, y_test, head_params)

                        res = {
                            'seed': [np_seed],
                            'data_idx': [data_idx],
                            'aug_params': [aug_params],
                            'tran_params': [tran_param],
                            'dr_params': [dr_params],
                            'head_params': [head_params],
                            'head': [head_params['model']],
                            'n': X_train_dr.shape[0],
                            'p': X_train_dr.shape[1],
                            **res,
                        }

                        out = pd.concat([out, pd.DataFrame(res)])
                    print(out)

    print(out)

    if method == 'fast':
        folder_name = './out/mlrepo12/'
    elif method == 'maml':
        folder_name = './out/mlrepo12maml/'
    elif method == 'deepmicro':
        folder_name = './out/mlrepo12deepmicro/'
    elif method == 'contrastive':
        folder_name = './out/mlrepo12contrastive/'
    file_name = folder_name + 'd' + str(data_idx) + 's' + str(np_seed)
    out.to_csv(file_name)

