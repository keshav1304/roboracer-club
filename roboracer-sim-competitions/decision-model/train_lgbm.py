"""Step 3: LightGBM baseline, features → teacher state.

Trains on the episode split from dataset_stats (700 / 150 / 150), early-stops
on val, reports test accuracy, per-state precision/recall, the confusion
matrix and the top features. Rare states are weighted up (inverse frequency,
square-rooted so they don't swamp the common ones).

  python train_lgbm.py data/v1                  # all 90 features → models/lgbm_state.txt
  python train_lgbm.py data/v1 --no-prev        # memoryless: drop the prev_* one-hot
"""
import argparse
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset_stats import load, split
from features import STATES, FeatureExtractor

HERE = pathlib.Path(__file__).resolve().parent


def feature_list(no_prev=False):
    return [f for f in FeatureExtractor.names() if not (no_prev and f.startswith("prev_"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data/v1")
    ap.add_argument("--no-prev", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df, man = load(args.data)
    feats = feature_list(args.no_prev)
    sp = dict(zip(man.seed, split(man.seed)))
    part = df.episode.map(sp)
    y = df.state.map({s: i for i, s in enumerate(STATES)}).to_numpy()
    X = df[feats].to_numpy(np.float32)
    tr, va, te = (part == "train").to_numpy(), (part == "val").to_numpy(), (part == "test").to_numpy()

    freq = np.bincount(y[tr], minlength=len(STATES)) / tr.sum()
    cw = np.sqrt(freq.max() / np.maximum(freq, 1e-6))
    print("class weights:", dict(zip(STATES, cw.round(1))))

    params = dict(objective="multiclass", num_class=len(STATES), learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, num_threads=8, verbose=-1, seed=0)
    dtr = lgb.Dataset(X[tr], y[tr], weight=cw[y[tr]], feature_name=feats)
    dva = lgb.Dataset(X[va], y[va], weight=cw[y[va]], reference=dtr)
    t0 = time.time()
    model = lgb.train(params, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    print(f"trained {model.best_iteration} trees in {time.time() - t0:.0f}s")

    pred = model.predict(X[te], num_iteration=model.best_iteration).argmax(1)
    acc = float((pred == y[te]).mean())
    macro = float(f1_score(y[te], pred, average="macro"))
    print(f"\ntest: accuracy {100 * acc:.2f}%, macro-F1 {macro:.3f} ({te.sum():,} rows, 150 unseen episodes)\n")
    print(classification_report(y[te], pred, labels=range(len(STATES)), target_names=STATES, digits=3,
                                zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y[te], pred, labels=range(len(STATES))), index=STATES, columns=STATES)
    print("confusion (rows = teacher, cols = model):\n", cm, "\n")
    imp = pd.Series(model.feature_importance("gain"), feats).sort_values(ascending=False)
    imp = (100 * imp / imp.sum()).round(1)
    print("top features (% of gain):", imp.head(15).to_dict())

    t1 = time.time()
    for i in range(500):
        model.predict(X[te][i:i + 1], num_threads=1)
    lat = (time.time() - t1) / 500 * 1e3

    out = pathlib.Path(args.out or HERE / "models" / ("lgbm_state_noprev.txt" if args.no_prev else "lgbm_state.txt"))
    out.parent.mkdir(exist_ok=True)
    model.save_model(str(out), num_iteration=model.best_iteration)
    report = {"data": str(args.data), "features": feats, "trees": model.best_iteration, "test_accuracy": acc,
              "test_macro_f1": macro, "single_row_latency_ms": round(lat, 3),
              "per_state": classification_report(y[te], pred, labels=range(len(STATES)), target_names=STATES,
                                                 output_dict=True, zero_division=0),
              "confusion": cm.to_dict(), "top_features": imp.head(25).to_dict()}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"\nsingle-row predict {lat:.2f} ms; model → {out}")


if __name__ == "__main__":
    main()
