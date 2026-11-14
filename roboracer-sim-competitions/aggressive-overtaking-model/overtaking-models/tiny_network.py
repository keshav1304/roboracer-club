"""Small, fully local neural maneuver classifier with safe NumPy checkpoints.

This module imports no network clients. The trained model produces maneuver
probabilities; the separate trajectory controller converts the selected feasible
maneuver to steering and target speed. Neither probabilities nor teacher advice
are certificates of collision avoidance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _targets(y, count):
    y = np.asarray(y)
    if not np.issubdtype(y.dtype, np.number) or not np.isfinite(y).all():
        raise ValueError("Labels must be finite numeric values.")
    if y.ndim == 1:
        if not np.issubdtype(y.dtype, np.integer):
            if not np.all(np.equal(y, np.floor(y))):
                raise ValueError("Class labels must be integers or probability rows.")
        if np.any(y < 0) or np.any(y >= count):
            raise ValueError("Label outside the declared classes.")
        return np.eye(count, dtype=np.float32)[y.astype(int)]
    if y.ndim != 2 or y.shape[1] != count or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Soft labels must be finite nonnegative class probability rows.")
    if not np.allclose(y.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Each soft label row must sum to one.")
    return y.astype(np.float32)


class TinyNetwork:
    """Two-hidden-layer ReLU MLP. All deployment operations are NumPy only."""

    def __init__(self, weights, biases, mean, scale, classes, feature_names, temperature=1.):
        self.weights = [np.ascontiguousarray(w, dtype=np.float32) for w in weights]
        self.biases = [np.ascontiguousarray(b, dtype=np.float32) for b in biases]
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.classes = tuple(classes)
        self.feature_names = tuple(feature_names)
        self.temperature = float(temperature)
        self.metadata = {}
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape or np.any(self.scale <= 0):
            raise ValueError("Invalid normalization parameters.")
        if len(self.feature_names) != len(self.mean) or len(set(self.feature_names)) != len(self.mean):
            raise ValueError("Feature schema must contain distinct names for every input.")
        if len(set(self.classes)) != len(self.classes) or len(self.classes) < 2:
            raise ValueError("At least two distinct classes are required.")
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError("Invalid layer count.")
        n = len(self.mean)
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or w.shape[0] != n or b.shape != (w.shape[1],):
                raise ValueError("Invalid network dimensions.")
            n = w.shape[1]
        if n != len(self.classes) or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid output or temperature.")
        if not all(np.isfinite(a).all() for a in self.weights + self.biases + [self.mean, self.scale]):
            raise ValueError("Nonfinite model parameters.")

    def _forward(self, X, activations=False):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim not in (1, 2) or X.shape[-1] != len(self.mean) or not np.isfinite(X).all():
            raise ValueError("Features must be finite and match the model schema.")
        a = (X - self.mean) / self.scale
        saved = [a]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            a = a @ w + b
            if i < len(self.weights) - 1:
                a = np.maximum(a, 0)
            saved.append(a)
        return (a, saved) if activations else a

    def predict_proba(self, X):
        return _softmax(self._forward(X) / self.temperature)

    def proba(self, x):
        """One observation in, one probability vector out (LEFT/RIGHT/WAIT[/PROBE])."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("proba expects one feature vector.")
        return self.predict_proba(x)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.metadata)
        meta.update(metadata or {})
        meta.update(format_version=1, classes=list(self.classes), feature_names=list(self.feature_names),
                    temperature=self.temperature, layers=len(self.weights),
                    architecture=[len(self.mean)] + [w.shape[1] for w in self.weights])
        arrays = {"mean": self.mean, "scale": self.scale, "metadata": np.array(json.dumps(meta))}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path, expected_feature_names=None):
        with np.load(path, allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
            if meta.get("format_version") != 1:
                raise ValueError("Unsupported checkpoint format.")
            names = tuple(meta["feature_names"])
            if expected_feature_names is not None and names != tuple(expected_feature_names):
                raise ValueError("Checkpoint feature schema/order mismatch.")
            model = cls([saved[f"w{i}"] for i in range(meta["layers"])],
                        [saved[f"b{i}"] for i in range(meta["layers"])],
                        saved["mean"], saved["scale"], meta["classes"], names,
                        meta.get("temperature", 1.))
        model.metadata = meta
        return model


def _measure(model, X, Y):
    probabilities = model.predict_proba(X)
    loss = -np.mean(np.sum(Y * np.log(np.clip(probabilities, 1e-8, 1)), axis=1))
    actual, predicted = Y.argmax(axis=1), probabilities.argmax(axis=1)
    confusion = np.zeros((Y.shape[1], Y.shape[1]), dtype=int)
    np.add.at(confusion, (actual, predicted), 1)
    return {"loss": float(loss), "accuracy": float(np.mean(actual == predicted)),
            "confusion_matrix": confusion.tolist(), "samples": len(X)}


def fit_network(X, y, output_path, *, validation=None, classes=("LEFT", "RIGHT", "WAIT"),
                feature_names=None, epochs=80, seed=0, hidden=(32, 24), warmstart=None,
                batch_size=128, learning_rate=.002, weight_decay=1e-5, metadata=None):
    """Fit soft-label cross entropy with Adam; return learning curves and metrics.

    Validation is supplied by the caller, ideally split by entire scenario/track
    sector. It selects a checkpoint and calibrates temperature; it must never be
    described as an independent final test. Training examples are not silently
    split row-wise here. All checkpoint files avoid pickle.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
        raise ValueError("Need at least two finite feature rows.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid training settings.")
    classes = tuple(classes)
    Y = _targets(y, len(classes))
    if len(Y) != len(X):
        raise ValueError("Feature/label count mismatch.")
    feature_names = tuple([f"feature_{i}" for i in range(X.shape[1])] if feature_names is None else feature_names)
    rng = np.random.default_rng(seed)
    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), .01)
    sizes = [X.shape[1], *hidden, len(classes)]
    weights = [(rng.standard_normal((a, b)) * np.sqrt(2 / a)).astype(np.float32)
               for a, b in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
    warmstart_used = None
    if warmstart is not None:
        prior = TinyNetwork.load(warmstart, feature_names)
        if [w.shape for w in prior.weights[:-1]] != [w.shape for w in weights[:-1]]:
            raise ValueError("Warmstart hidden architecture does not match.")
        if prior.weights[-1].shape[0] != weights[-1].shape[0]:
            raise ValueError("Warmstart final hidden size does not match.")
        for j in range(len(weights) - 1):
            weights[j], biases[j] = prior.weights[j].copy(), prior.biases[j].copy()
        # Re-express the inherited first layer under the new stage's scaler.
        # Formerly constant inputs (e.g. target speed in static-only training)
        # have no learned contribution and start at zero instead of becoming
        # enormous normalized inputs when motion is introduced.
        inherited = prior.weights[0].copy()
        inherited[prior.scale <= .010001] = 0.
        weights[0] = inherited * (scale / prior.scale)[:, None]
        biases[0] = prior.biases[0] + ((mean - prior.mean) / prior.scale) @ inherited
        for j, name in enumerate(classes):
            if name in prior.classes:
                old = prior.classes.index(name)
                weights[-1][:, j], biases[-1][j] = prior.weights[-1][:, old], prior.biases[-1][old]
            else:
                biases[-1][j] = -2.
        warmstart_used = str(Path(warmstart))
    model = TinyNetwork(weights, biases, mean, scale, classes, feature_names)
    model.metadata = dict(metadata or {})
    model.metadata.update(training_seed=seed, training_samples=len(X), warmstart=warmstart_used)
    val = None
    if validation is not None:
        VX, VY = np.asarray(validation[0], dtype=np.float32), _targets(validation[1], len(classes))
        if VX.ndim != 2 or len(VX) != len(VY) or not len(VX) or VX.shape[1] != X.shape[1]:
            raise ValueError("Invalid validation dimensions.")
        if not np.isfinite(VX).all():
            raise ValueError("Validation features contain nonfinite values.")
        val = VX, VY
    parameters = model.weights + model.biases
    moments = [np.zeros_like(p) for p in parameters]
    variances = [np.zeros_like(p) for p in parameters]
    history, step, best_loss, best = [], 0, float("inf"), None
    output_path = Path(output_path)
    if output_path.suffix != ".npz":
        raise ValueError("output_path must end in .npz.")
    checkpoints = {"early": output_path.with_name(output_path.stem + ".early.npz"),
                   "mid": output_path.with_name(output_path.stem + ".mid.npz"), "final": output_path}
    for epoch in range(1, epochs + 1):
        indices = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            ix = indices[start:start + batch_size]
            logits, activations = model._forward(X[ix], activations=True)
            delta = (_softmax(logits) - Y[ix]) / len(ix)
            gws, gbs = [None] * len(model.weights), [None] * len(model.weights)
            for layer in reversed(range(len(model.weights))):
                gws[layer] = activations[layer].T @ delta + weight_decay * model.weights[layer]
                gbs[layer] = delta.sum(axis=0)
                if layer:
                    delta = (delta @ model.weights[layer].T) * (activations[layer] > 0)
            step += 1
            for p, grad, m, v in zip(parameters, gws + gbs, moments, variances):
                m *= .9
                m += .1 * grad
                v *= .999
                v += .001 * grad * grad
                p -= learning_rate * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        train_metrics = _measure(model, X, Y)
        validation_metrics = _measure(model, *val) if val else None
        record = {"epoch": epoch, "train_loss": train_metrics["loss"],
                  "train_accuracy": train_metrics["accuracy"]}
        if val:
            record.update(validation_loss=validation_metrics["loss"],
                          validation_accuracy=validation_metrics["accuracy"])
        history.append(record)
        selection_loss = (validation_metrics or train_metrics)["loss"]
        if selection_loss < best_loss:
            best_loss = selection_loss
            best = ([p.copy() for p in model.weights], [p.copy() for p in model.biases], epoch)
        for name, checkpoint_epoch in (("early", 1), ("mid", max(1, epochs // 2))):
            if epoch == checkpoint_epoch:
                model.save(checkpoints[name], {"training_epoch": epoch, "checkpoint": name})
    model.weights, model.biases, selected_epoch = best
    if val:
        logits = model._forward(val[0])
        temperatures = np.linspace(.6, 2.5, 39)
        losses = [-np.mean(np.sum(val[1] * np.log(np.clip(_softmax(logits / t), 1e-8, 1)), axis=1))
                  for t in temperatures]
        model.temperature = float(temperatures[int(np.argmin(losses))])
    model.save(output_path, {"training_epoch": selected_epoch, "checkpoint": "final",
                            "temperature_calibration": "validation" if val else "none"})
    report = {"classes": list(classes), "features": list(feature_names), "epochs": epochs,
              "selected_epoch": selected_epoch, "temperature": model.temperature,
              "train": _measure(model, X, Y), "validation": _measure(model, *val) if val else None,
              "validation_is_independent_test": False, "history": history,
              "checkpoints": {k: str(v) for k, v in checkpoints.items()},
              "warmstart": warmstart_used, "metadata": dict(metadata or {})}
    output_path.with_suffix(".history.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
