import numpy as np
from sklearn.linear_model import LogisticRegression

import torch.nn as nn
import xgboost as xgb


class Trainer:
    def __init__(self, method):
        self._inputs_train = None
        self._targets_train = None
        self._inputs_valid = None
        self._targets_valid = None
        self.model = None

        if method == "logistic":
            self.train = self._train_logistic
            self.predict = self._predict_logistic
        elif method == "xgb":
            self.train = self._train_xgboost
            self.predict = self._predict_xgboost
        else:
            assert False, "Method {} not found".format(method)

    def set_train_data(self, inputs: np.ndarray, targets: np.ndarray):
        self._inputs_train = inputs
        self._targets_train = targets

    def set_valid_data(self, inputs: np.ndarray, targets: np.ndarray):
        self._inputs_valid = inputs
        self._targets_valid = targets

    def _train_logistic(self, params=None):
        self.model = LogisticRegression().fit(
            self._inputs_train,
            self._targets_train
        )

    def _predict_logistic(self, inputs):
        return self.model.predict_log_proba(inputs)[:, 1]

    def _train_xgboost(self, params=None):
        train_data = xgb.DMatrix(self._inputs_train, label=self._targets_train)
        valid_data = xgb.DMatrix(self._inputs_valid, label=self._targets_valid)

        config = {
            "objective": "binary:logistic",
            "nthread": 8,
            "eval_metric": "auc",
            "tree_method": "gpu_hist",
            "gpu_id": 0
        }
        if params:
            config.update(params)
        eval_list = [(valid_data, "eval"), (train_data, "train")]
        self.model = xgb.train(
            config,
            train_data,
            num_boost_round=100,
            evals=eval_list
        )

    def _predict_xgboost(self, inputs):
        inputs = xgb.DMatrix(inputs)
        return self.model.predict(inputs)


class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim: int, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
