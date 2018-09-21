"""Implementation of an arbitrary order Factorization Machines."""

import numpy as np
import tensorflow as tf
from .core import TFFMCore
from .base import TFFMBaseModel
from .utils import loss_cross_entropy, loss_mse, sigmoid

n_entity = 14951
n_relation = 1345

class TFFMClassifier(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with logistic
    loss and gradient-based optimization.

    Only binary classification with 0/1 labels supported.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):
        assert 'loss_function' not in init_params
        init_params['loss_function'] = loss_cross_entropy
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        assert(set(y_) == set([0, 1]))
        return y_


    def predict(self, Xs, Xr, Xo, pred_batch_size=None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.
        pred_batch_size : int batch size for prediction (default None)

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        raw_output = self.decision_function(Xs, Xr, Xo, pred_batch_size)
        predictions = sigmoid(raw_output)
        return predictions



class TFFMRegressor(TFFMBaseModel):
    """Factorization Machine (aka FM).

    This class implements L2-regularized arbitrary order FM model with MSE
    loss and gradient-based optimization.

    See TFFMBaseModel and TFFMCore docs for details about parameters.
    """

    def __init__(self, **init_params):
        assert 'loss_function' not in init_params
        init_params['loss_function'] = loss_mse
        self.init_basemodel(**init_params)

    def preprocess_target(self, y_):
        return y_

    def predict(self, X, pred_batch_size=None):
        """Predict using the FM model

        Parameters
        ----------
        X : {numpy.array, scipy.sparse.csr_matrix}, shape = (n_samples, n_features)
            Samples.
        pred_batch_size : int batch size for prediction (default None)

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Returns predicted values.
        """
        predictions = self.decision_function(X, pred_batch_size)
        return predictions
