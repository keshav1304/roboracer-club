"""Sanity report for a generated dataset: size, label balance, outcomes,
feature ranges, and the train/val/test split by episode.

  python dataset_stats.py data/v1
"""
import pathlib
import sys

import numpy as np
import pandas as pd

from features import STATES, FeatureExtractor


def load(root):
    root = pathlib.Path(root)
    return pd.concat([pd.read_parquet(f) for f in sorted((root / "episodes").glob("ep_*.parquet"))],
                     ignore_index=True), pd.read_parquet(root / "manifest.parquet")


def split(episodes, val=0.15, test=0.15):
    """Split by episode (never by row: neighbouring rows are near-duplicates)."""
    h = (np.asarray(episodes) * 2654435761 % 2 ** 32) / 2 ** 32
    return np.where(h < test, "test", np.where(h < test + val, "val", "train"))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data/v1"
    df, man = load(root)
    feats = FeatureExtractor.names()
    print(f"{root}: {len(man)} episodes, {len(df):,} rows, {len(feats)} features, "
          f"{man.ticks.sum() / 100 / 3600:.2f} h of driving\n")

    print("episodes by kind:", man.kind.value_counts().to_dict())
    print(f"collisions: {man.collision.notna().sum()} ({man.collision.value_counts().to_dict()}); "
          f"passes: {man.passes.sum()}\n")

    print("teacher state share (rows):")
    vc = df.state.value_counts(normalize=True)
    for s in STATES:
        print(f"  {s:12s} {100 * vc.get(s, 0):5.1f}%")
    print("\nstate share by scenario kind (%):")
    print((pd.crosstab(df.kind, df.state, normalize="index") * 100).round(1).reindex(columns=STATES, fill_value=0))

    ok = df[df.y_horizon_ok]
    print(f"\noutcomes (rows with full look-ahead: {len(ok):,}):")
    print(f"  y_collide_1s  {100 * ok.y_collide_1s.mean():.2f}% positive")
    print(f"  y_pass_3s     {100 * ok.y_pass_3s.mean():.2f}% positive")
    print(f"  y_progress_2s mean {ok.y_progress_2s.mean():.2f} m, p5 {ok.y_progress_2s.quantile(.05):.2f} m")
    g = ok.y_min_gap_2s.dropna()
    print(f"  y_min_gap_2s  p1 {g.quantile(.01):.2f} m, median {g.median():.2f} m")
    print("  mean progress by state (m / 2 s):", ok.groupby("state").y_progress_2s.mean().round(2).to_dict())

    X = df[feats]
    bad = X.columns[~np.isfinite(X).all()].tolist()
    print(f"\nnon-finite feature columns: {bad or 'none'}")
    const = X.columns[X.std() < 1e-9].tolist()
    print(f"constant feature columns: {const or 'none'}")

    sp = split(man.seed)
    print("\nsplit by episode:", pd.Series(sp).value_counts().to_dict())


if __name__ == "__main__":
    main()
