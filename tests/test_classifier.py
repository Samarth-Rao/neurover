import numpy as np
import pytest

import config
from pipeline.classifier import GestureClassifier, Trainer

CLASSES = config.GESTURES


def blobs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for i, c in enumerate(CLASSES):
        centre = np.zeros(8)
        centre[i] = 10.0
        X.append(centre + rng.normal(0, 1.0, (n, 8)))
        y += [c] * n
    return np.vstack(X), np.array(y)


def test_fit_predict_with_confidence_and_probs_for_every_class():
    X, y = blobs()
    clf = GestureClassifier(CLASSES, k=5)
    report = clf.fit(X, y)
    assert report.n_samples == len(X) and report.cv_accuracy > 0.95
    label, conf, probs = clf.predict(X[0])
    assert label == "rest" and conf == pytest.approx(1.0)
    assert set(probs) == set(CLASSES) and sum(probs.values()) == pytest.approx(1.0)


def test_confidence_is_the_share_of_agreeing_neighbours():
    X = np.array([[0.0] * 8] * 3 + [[1.0] * 8] * 3 + [[10.0] * 8] * 3)
    y = ["fist"] * 3 + ["spread"] * 3 + ["rest"] * 3
    clf = GestureClassifier(CLASSES, k=5)
    clf.fit(X, y)
    label, conf, probs = clf.predict([0.4] * 8)   # 3 fist + 2 spread neighbours
    assert label == "fist"
    assert conf == pytest.approx(0.6)
    assert probs["spread"] == pytest.approx(0.4) and probs["rest"] == 0.0


def test_needs_two_gestures():
    X, y = blobs()
    with pytest.raises(ValueError):
        GestureClassifier(CLASSES).fit(X[y == "fist"], y[y == "fist"])


def test_k_shrinks_for_tiny_datasets():
    clf = GestureClassifier(CLASSES, k=5)
    clf.fit(np.array([[0.0] * 8, [10.0] * 8]), ["rest", "fist"])
    assert clf.predict([9.0] * 8)[0] == "fist"


def test_save_and_load_roundtrip(tmp_path):
    X, y = blobs()
    tr = Trainer("unit", CLASSES, models_dir=tmp_path, meta={"channels": 8})
    tr.data.extend(X, y)
    path = tr.save()
    assert path.exists() and tr.data_path.exists()

    fresh = Trainer("unit", CLASSES, models_dir=tmp_path, meta={"channels": 8})
    msg = fresh.load()
    assert "loaded" in msg
    assert len(fresh.data) == len(X)
    assert fresh.clf.predict(X[-1])[0] == y[-1]


def test_corrupt_model_file_is_refit_from_saved_data(tmp_path):
    X, y = blobs()
    tr = Trainer("unit", CLASSES, models_dir=tmp_path)
    tr.data.extend(X, y)
    tr.save()
    tr.model_path.write_bytes(b"not a pickle")
    fresh = Trainer("unit", CLASSES, models_dir=tmp_path)
    assert "loaded" in fresh.load()
    assert fresh.clf.is_fitted


def test_incompatible_saved_model_is_ignored(tmp_path):
    X, y = blobs()
    tr = Trainer("unit", CLASSES, models_dir=tmp_path, meta={"channels": 8})
    tr.data.extend(X, y)
    tr.save()
    other = Trainer("unit", CLASSES, models_dir=tmp_path, meta={"channels": 4})
    assert "ignored" in other.load()
    assert not other.clf.is_fitted


def test_nothing_saved_loads_nothing(tmp_path):
    assert Trainer("none", CLASSES, models_dir=tmp_path).load() is None
