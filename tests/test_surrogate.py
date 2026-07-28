import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.surrogate import RiskCalibratedSurrogate, _clopper_pearson_upper


class _FixedFailureScores:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        failure = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack((1.0 - failure, failure))


def _fixed_surrogate():
    surrogate = RiskCalibratedSurrogate()
    surrogate.model_ = _FixedFailureScores()
    surrogate.n_features_in_ = 1
    return surrogate


def test_clopper_pearson_upper_boundaries():
    assert _clopper_pearson_upper(0, 0, 0.05) == 1.0
    assert np.isclose(_clopper_pearson_upper(0, 1, 0.05), 0.95)
    assert _clopper_pearson_upper(1, 1, 0.05) == 1.0
    assert 0.0 < _clopper_pearson_upper(3, 100, 0.001) < 1.0


def test_shared_threshold_is_constrained_by_worse_robot():
    surrogate = _fixed_surrogate()
    safe = np.full(200, 0.10)
    risky = np.full(20, 0.15)
    X = np.concatenate((safe, safe, risky))[:, None]
    y = np.concatenate((np.zeros(400, dtype=int), np.ones(20, dtype=int)))
    robots = np.array(["talos"] * 200 + ["icub"] * 220)

    threshold = surrogate.calibrate_threshold(X, y, robots)

    assert 0.10 <= threshold < 0.15
    assert surrogate.accept(np.array([[0.10], [0.15]])).tolist() == [True, False]
    assert surrogate.calibration_["selection_alpha"] == 0.05 / (50 * 2)
    assert surrogate.calibration_["robots"]["talos"]["false_safe_count"] == 0
    assert surrogate.calibration_["robots"]["icub"]["false_safe_count"] == 0


def test_calibration_accepts_nothing_when_no_threshold_is_certified():
    surrogate = _fixed_surrogate()
    X = np.array([[0.1], [0.2], [0.1], [0.2]])
    y = np.zeros(4, dtype=int)
    robots = np.array(["talos", "talos", "icub", "icub"])

    assert surrogate.calibrate_threshold(X, y, robots) is None
    assert not surrogate.accept(X).any()


def test_calibration_requires_both_prespecified_robot_groups():
    surrogate = _fixed_surrogate()
    try:
        surrogate.calibrate_threshold(
            np.array([[0.1], [0.2]]),
            np.zeros(2, dtype=int),
            np.array(["talos", "talos"]),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("single-robot calibration was accepted")


def test_fit_is_deterministic_and_does_not_add_robot_identity_feature():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(240, 3))
    y = (X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.5, size=240) > 0).astype(int)
    robots = np.where(np.arange(240) % 2, "talos", "icub")

    first = RiskCalibratedSurrogate(random_state=23).fit(X, y, robots)
    second = RiskCalibratedSurrogate(random_state=23).fit(X, y, robots[::-1])

    assert first.n_features_in_ == X.shape[1]
    assert first.best_params_ == second.best_params_
    assert np.array_equal(first.predict_failure_score(X), second.predict_failure_score(X))


def test_single_class_fit_returns_the_constant_failure_score():
    X = np.array([[0.0], [1.0]])
    robots = np.array(["talos", "icub"])
    feasible = RiskCalibratedSurrogate().fit(X, np.zeros(2, dtype=int), robots)
    failed = RiskCalibratedSurrogate().fit(X, np.ones(2, dtype=int), robots)

    assert np.array_equal(feasible.predict_failure_score(X), np.zeros(2))
    assert np.array_equal(failed.predict_failure_score(X), np.ones(2))


def test_tiny_two_class_smoke_fit_is_well_defined():
    X = np.array([[0.0], [1.0]])
    model = RiskCalibratedSurrogate().fit(X, np.array([0, 1]), ["talos", "icub"])
    scores = model.predict_failure_score(X)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_acceptance_direction_and_pickle_round_trip():
    surrogate = _fixed_surrogate()
    surrogate.threshold_ = 0.5
    surrogate.calibration_ = {"threshold": 0.5}
    X = np.array([[0.2], [0.8]])
    assert surrogate.accept(X).tolist() == [True, False]

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "surrogate.pkl")
        surrogate.save(path)
        restored = RiskCalibratedSurrogate.load(path)

    assert np.array_equal(restored.predict_failure_score(X), surrogate.predict_failure_score(X))
    assert np.array_equal(restored.accept(X), surrogate.accept(X))
    assert restored.calibration_ == surrogate.calibration_


def test_explicit_tune_split_is_supported():
    X = np.array([[0.0], [1.0], [0.1], [0.9]])
    y = np.array([0, 1, 0, 1])
    robots = np.array(["talos", "talos", "icub", "icub"])
    fitted = RiskCalibratedSurrogate(3).fit(
        X, y, robots, tune_data=(X.copy(), y.copy(), robots.copy())
    )
    assert fitted.n_features_in_ == 1
    assert fitted.predict_failure_score(X).shape == (4,)


if __name__ == "__main__":
    test_clopper_pearson_upper_boundaries()
    test_shared_threshold_is_constrained_by_worse_robot()
    test_calibration_accepts_nothing_when_no_threshold_is_certified()
    test_calibration_requires_both_prespecified_robot_groups()
    test_fit_is_deterministic_and_does_not_add_robot_identity_feature()
    test_single_class_fit_returns_the_constant_failure_score()
    test_tiny_two_class_smoke_fit_is_well_defined()
    test_acceptance_direction_and_pickle_round_trip()
    test_explicit_tune_split_is_supported()
    print("surrogate tests passed")
