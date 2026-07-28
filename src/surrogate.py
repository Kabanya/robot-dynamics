"""Failure-score surrogate with finite-sample acceptance calibration."""

from itertools import product
import math
import pickle

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


_PARAMETER_GRID = tuple(
    {
        "max_leaf_nodes": leaves,
        "learning_rate": rate,
        "max_iter": iterations,
        "l2_regularization": regularization,
    }
    for leaves, rate, iterations, regularization in product(
        (7, 15, 31), (0.03, 0.08), (150, 300), (0.0, 1.0)
    )
)
_THRESHOLDS = np.linspace(0.0, 1.0, 50)
_RISK_LIMIT = 0.05
_FAMILYWISE_ALPHA = 0.05
_CALIBRATION_ROBOTS = frozenset(("talos", "icub"))


def _clopper_pearson_upper(failures, trials, alpha):
    """Return the exact one-sided ``1-alpha`` binomial upper bound."""
    if not isinstance(failures, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise TypeError("failures and trials must be integers")
    if trials < 0 or failures < 0 or failures > trials:
        raise ValueError("require 0 <= failures <= trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if trials == 0 or failures == trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, trials - failures))


def _validated_data(X, y, robot_names):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    robots = np.asarray(robot_names, dtype=str)
    if X.ndim != 2 or not len(X):
        raise ValueError("X must be a non-empty 2D array")
    if y.ndim != 1 or robots.ndim != 1 or len(y) != len(X) or len(robots) != len(X):
        raise ValueError("X, y, and robot_names must have matching rows")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must use 1 for failure and 0 for feasibility")
    if np.any(robots == ""):
        raise ValueError("robot_names must be non-empty")
    return X, y.astype(np.int8), robots


def _failure_scores(model, X):
    classes = np.asarray(model.classes_)
    if len(classes) == 1:
        return np.full(len(X), float(classes[0] == 1))
    failure_column = np.flatnonzero(classes == 1)
    if len(failure_column) != 1:
        raise RuntimeError("model has no unique failure class")
    return model.predict_proba(X)[:, failure_column[0]]


class RiskCalibratedSurrogate:
    """Pooled failure classifier; low failure scores are accepted."""

    def __init__(self, random_state=0):
        self.random_state = int(random_state)

    def _new_model(self, params):
        return HistGradientBoostingClassifier(random_state=self.random_state, **params)

    def fit(self, X, y, robot_names, *, tune_data=None):
        """Tune by AUPRC, then fit one pooled model without robot-ID features.

        ``tune_data`` may be ``(X_tune, y_tune, robot_names_tune)``. When it is
        omitted, a deterministic stratified quarter of ``X`` is used for tuning.
        """
        X, y, _ = _validated_data(X, y, robot_names)

        if tune_data is None:
            _, counts = np.unique(y, return_counts=True)
            test_rows = math.ceil(0.25 * len(y))
            can_tune = (
                len(counts) == 2
                and counts.min() >= 2
                and test_rows >= 2
                and len(y) - test_rows >= 2
            )
            if can_tune:
                X_train, X_tune, y_train, y_tune = train_test_split(
                    X, y, test_size=0.25, random_state=self.random_state, stratify=y
                )
            else:
                X_train = X_tune = X
                y_train = y_tune = y
            final_X, final_y = X, y
        else:
            if len(tune_data) != 3:
                raise ValueError("tune_data must be (X, y, robot_names)")
            X_tune, y_tune, _ = _validated_data(*tune_data)
            if X_tune.shape[1] != X.shape[1]:
                raise ValueError("training and tuning feature counts differ")
            X_train, y_train = X, y
            final_X = np.vstack((X, X_tune))
            final_y = np.concatenate((y, y_tune))
            can_tune = len(np.unique(y_train)) == 2

        best_score = -np.inf
        best_params = None
        parameter_grid = _PARAMETER_GRID if can_tune else _PARAMETER_GRID[:1]
        for params in parameter_grid:
            candidate = self._new_model(params).fit(X_train, y_train)
            if np.any(y_tune):
                score = average_precision_score(y_tune, _failure_scores(candidate, X_tune))
            else:
                score = 0.0
            if score > best_score:
                best_score, best_params = float(score), params

        self.best_params_ = dict(best_params)
        self.tune_auprc_ = best_score
        self.model_ = self._new_model(best_params).fit(final_X, final_y)
        self.n_features_in_ = X.shape[1]
        self.threshold_ = None
        self.calibration_ = None
        return self

    def predict_failure_score(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before prediction")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("X has invalid shape or non-finite values")
        return _failure_scores(self.model_, X)

    def calibrate_threshold(self, X, y, robot_names):
        X, y, robots = _validated_data(X, y, robot_names)
        scores = self.predict_failure_score(X)
        robot_values = np.unique(robots)
        if set(robot_values) != _CALIBRATION_ROBOTS:
            raise ValueError(
                "calibration requires exactly the prespecified Talos and iCub groups"
            )
        selection_alpha = _FAMILYWISE_ALPHA / (
            len(_THRESHOLDS) * len(robot_values)
        )
        best = None

        for threshold in _THRESHOLDS:
            accepted = scores <= threshold
            details = {}
            valid = True
            for robot in robot_values:
                selected = accepted & (robots == robot)
                trials = int(selected.sum())
                failures = int(y[selected].sum())
                upper = _clopper_pearson_upper(failures, trials, selection_alpha)
                details[robot] = {
                    "accepted_count": trials,
                    "false_safe_count": failures,
                    "upper_bound": upper,
                }
                valid &= trials > 0 and upper <= _RISK_LIMIT
            candidate = (int(accepted.sum()), float(threshold), details)
            if valid and (best is None or candidate[:2] > best[:2]):
                best = candidate

        if best is None:
            self.threshold_ = None
            accepted_count = 0
            details = {
                robot: {
                    "accepted_count": 0,
                    "false_safe_count": 0,
                    "upper_bound": 1.0,
                }
                for robot in robot_values
            }
        else:
            accepted_count, self.threshold_, details = best

        self.calibration_ = {
            "threshold": self.threshold_,
            "coverage": accepted_count / len(X),
            "accepted_count": accepted_count,
            "sample_count": len(X),
            "risk_limit": _RISK_LIMIT,
            "familywise_alpha": _FAMILYWISE_ALPHA,
            "selection_alpha": selection_alpha,
            "threshold_count": len(_THRESHOLDS),
            "robots": details,
        }
        return self.threshold_

    def accept(self, X):
        scores = self.predict_failure_score(X)
        if getattr(self, "threshold_", None) is None:
            return np.zeros(len(scores), dtype=bool)
        return scores <= self.threshold_

    def save(self, path):
        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before save")
        with open(path, "wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as stream:
            surrogate = pickle.load(stream)
        if not isinstance(surrogate, cls):
            raise TypeError("pickle does not contain a RiskCalibratedSurrogate")
        return surrogate
