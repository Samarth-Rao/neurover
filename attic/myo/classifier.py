"""Gesture classifier (standardized features -> KNN) plus training data and save/load.

Files (per source, so a mock-trained model never drives the real Myo):
    models/<name>_model.joblib   fitted sklearn pipeline + metadata
    models/<name>_data.npz       the labelled feature windows it was trained on
If the model file can't be unpickled (e.g. different scikit-learn version) it is
refit from the saved data, which is always the source of truth.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import config


@dataclass(frozen=True)
class FitReport:
    n_samples: int
    counts: Mapping[str, int]
    cv_accuracy: Optional[float]

    def describe(self) -> str:
        acc = f", accuracy ~{self.cv_accuracy * 100:.0f}% (held-out folds)" if self.cv_accuracy is not None else ""
        return f"fitted on {self.n_samples:,} windows{acc}"


class GestureClassifier:
    def __init__(self, classes: Sequence[str], k: int = config.KNN_K) -> None:
        self.classes = tuple(classes)
        self.k = k
        self.model: Optional[Pipeline] = None

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    def _make(self, n: int) -> Pipeline:
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=max(1, min(self.k, n))))

    def fit(self, X: np.ndarray, y: Sequence[str], cv_folds: int = config.TRAIN_CV_FOLDS) -> FitReport:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if len(X) != len(y) or len(X) == 0:
            raise ValueError("no training data")
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise ValueError("need examples of at least 2 gestures")
        unknown = set(labels) - set(self.classes)
        if unknown:
            raise ValueError(f"unknown labels: {sorted(unknown)}")
        cv = None
        folds = min(cv_folds, int(counts.min()))
        if folds >= 2:
            # Unshuffled folds keep each fold a contiguous chunk of the recording, so the estimate
            # isn't inflated by overlapping neighbouring windows landing in train and test.
            scores = cross_val_score(self._make(len(X)), X, y, cv=StratifiedKFold(n_splits=folds))
            cv = float(np.mean(scores))
        self.model = self._make(len(X)).fit(X, y)
        return FitReport(len(X), {str(l): int(c) for l, c in zip(labels, counts)}, cv)

    def predict(self, features: np.ndarray) -> tuple[str, float, dict[str, float]]:
        """Returns (label, confidence, per-class probabilities for every class in ``classes``)."""
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        proba = self.model.predict_proba(np.asarray(features, dtype=float).reshape(1, -1))[0]
        fitted = self.model.classes_
        probs = {c: 0.0 for c in self.classes}
        for c, p in zip(fitted, proba):
            probs[str(c)] = float(p)
        i = int(np.argmax(proba))
        return str(fitted[i]), float(proba[i]), probs

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("classifier is not fitted")
        return self.model.predict(np.asarray(X, dtype=float))


class TrainingSet:
    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = tuple(classes)
        self.clear()

    def clear(self) -> None:
        self._X: list[np.ndarray] = []
        self._y: list[str] = []

    def add(self, features: np.ndarray, label: str) -> None:
        if label not in self.classes:
            raise ValueError(f"unknown label {label!r}")
        self._X.append(np.asarray(features, dtype=float))
        self._y.append(label)

    def extend(self, X: np.ndarray, y: Sequence[str]) -> None:
        for x, label in zip(X, y):
            self.add(x, str(label))

    def __len__(self) -> int:
        return len(self._y)

    def counts(self) -> dict[str, int]:
        out = {c: 0 for c in self.classes}
        for label in self._y:
            out[label] += 1
        return out

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._y:
            return np.empty((0, 0)), np.empty((0,), dtype=str)
        return np.vstack(self._X), np.asarray(self._y)


class Trainer:
    """Training data + classifier for one source, with save/load to models/."""

    def __init__(self, name: str, classes: Sequence[str], *, models_dir: Path = config.MODELS_DIR,
                 k: int = config.KNN_K, meta: Optional[dict] = None) -> None:
        self.name = name
        self.models_dir = Path(models_dir)
        self.data = TrainingSet(classes)
        self.clf = GestureClassifier(classes, k)
        self.meta = dict(meta or {})
        self.report: Optional[FitReport] = None
        self.dirty = False  # data changed since the last fit
        self.saved_meta: Optional[dict] = None   # meta of the model last loaded

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.name}_model.joblib"

    @property
    def data_path(self) -> Path:
        return self.models_dir / f"{self.name}_data.npz"

    def add(self, features: np.ndarray, label: str) -> None:
        self.data.add(features, label)
        self.dirty = True

    def clear(self) -> None:
        self.data.clear()
        self.clf.model = None
        self.report = None
        self.dirty = False

    def fit(self) -> FitReport:
        X, y = self.data.arrays()
        self.report = self.clf.fit(X, y)
        self.dirty = False
        return self.report

    def save(self) -> Path:
        """Fit if needed, then write model + training data. Returns the model path."""
        if not self.clf.is_fitted or self.dirty:
            self.fit()
        import joblib

        self.models_dir.mkdir(parents=True, exist_ok=True)
        X, y = self.data.arrays()
        meta = {**self.meta, "name": self.name, "classes": list(self.data.classes), "k": self.clf.k,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        np.savez_compressed(self.data_path, X=X, y=y.astype(str), meta=json.dumps(meta))
        joblib.dump({"model": self.clf.model, "meta": meta}, self.model_path)
        return self.model_path

    def load(self) -> Optional[str]:
        """Load saved data + model if present. Returns a human-readable message, or None if nothing saved."""
        if not self.data_path.exists():
            return None
        try:
            with np.load(self.data_path, allow_pickle=False) as z:
                X, y = z["X"], z["y"]
                saved_meta = json.loads(str(z["meta"]))
        except Exception as exc:
            return f"couldn't read {self.data_path.name}: {exc}"
        for key in ("channels", "features", "window_s", "hop_s"):
            if key in self.meta and key in saved_meta and saved_meta[key] != self.meta[key]:
                return f"ignored saved {self.name} model: {key} changed ({saved_meta[key]} -> {self.meta[key]})"
        if "classes" in saved_meta and list(saved_meta["classes"]) != list(self.data.classes):
            return f"ignored saved {self.name} model: it was trained on other gestures ({saved_meta['classes']})"
        self.saved_meta = saved_meta
        self.data.clear()
        try:
            self.data.extend(X, y)
        except ValueError as exc:
            self.data.clear()
            return f"ignored saved {self.name} model: {exc}"
        model = None
        try:
            import joblib
            model = joblib.load(self.model_path)["model"]
        except Exception:
            model = None
        try:
            if model is not None:
                self.clf.model = model
                labels, counts = np.unique(y, return_counts=True)
                self.report = FitReport(len(y), {str(l): int(c) for l, c in zip(labels, counts)}, None)
            else:
                self.fit()  # refit from the saved data
        except Exception as exc:
            return f"loaded {len(y)} {self.name} training windows, but fitting failed: {exc}"
        self.dirty = False
        return f"loaded {self.name} model ({len(y):,} windows, saved {saved_meta.get('saved_at', '?')})"
