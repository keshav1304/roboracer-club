"""Step 4: train RaceDecider-XS on the step-2 data.

Same episode split as LightGBM (700 / 150 / 150) and the same rare-state
weighting. Loss = state cross-entropy + auxiliary outcome heads (collision,
progress, pass). Training corrupts the prev_* one-hot on prev_noise of rows,
so the model can't lean on "keep the previous state" alone: in closed loop it
sees its own, possibly wrong, previous decision.

  python train_nn.py data/v2                 # → models/race_decider_xs.pt
  python train_nn.py data/v2 --extra data/dagger1 --init models/race_decider_xs.pt --epochs 6 \
      --out models/race_decider_xs_dagger.pt  # step 5: fine-tune on DAgger + counterfactual labels
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

from dataset_stats import load, split
from features import STATES
from race_decider import NAMES, PREV, RaceDecider

HERE = pathlib.Path(__file__).resolve().parent


def tensors(df, rows):
    d = df[rows]
    ok = d.y_horizon_ok.to_numpy(bool)
    lab = d["label"] if "label" in d else d.state    # DAgger rows carry their own (possibly counterfactual) label
    return {
        "x": torch.tensor(d[NAMES].to_numpy(np.float32)),
        "y": torch.tensor(lab.map({s: i for i, s in enumerate(STATES)}).to_numpy()),
        "w": torch.tensor(np.where(d["cf_gain"].fillna(0).to_numpy() > 0, 2.0, 1.0) if "cf_gain" in d
                          else np.ones(len(d)), dtype=torch.float32),
        "ok": torch.tensor(ok),
        "col": torch.tensor(d.y_collide_1s.to_numpy(np.float32)),
        "prog": torch.tensor(d.y_progress_2s.to_numpy(np.float32)),
        "pas": torch.tensor(d.y_pass_3s.to_numpy(np.float32)),
    }


@torch.no_grad()
def predict(m, x, bs=65536):
    m.eval()
    outs = [m(x[i:i + bs]) for i in range(0, len(x), bs)]
    return [torch.cat([o[k] for o in outs]) for k in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data/v1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--prev-noise", type=float, default=0.15)
    ap.add_argument("--extra", action="append", default=[], help="DAgger dataset dir(s), all used for training")
    ap.add_argument("--init", default=None, help="start from this checkpoint (fine-tune)")
    ap.add_argument("--out", default=str(HERE / "models" / "race_decider_xs.pt"))
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)

    df, man = load(args.data)
    part = df.episode.map(dict(zip(man.seed, split(man.seed))))
    tr, va, te = (tensors(df, (part == p).to_numpy()) for p in ("train", "val", "test"))
    for e in args.extra:
        ed, _ = load(e)
        ex = tensors(ed, np.ones(len(ed), bool))
        print(f"+ {e}: {len(ed):,} rows ({int((ed.label != ed.teacher_label).sum()):,} counterfactual labels)")
        tr = {k: torch.cat([tr[k], ex[k]]) for k in tr}

    if args.init:
        from race_decider import load as load_model
        m = load_model(args.init)
        args.width = m.width
        print(f"fine-tuning from {args.init}")
    else:
        m = RaceDecider(args.width)
    x = tr["x"]
    if not args.init:
        m.mu.copy_(x.mean(0))
        m.sd.copy_(x.std(0).clamp_min(1e-3))
        for i in (NAMES.index(n) for n in PREV + ["opp0_present"]):   # leave one-hots / flags raw
            m.mu[i], m.sd[i] = 0.0, 1.0
    n_par = sum(p.numel() for p in m.parameters())
    print(f"RaceDecider-XS width {args.width}: {n_par:,} parameters; train {len(x):,} rows")

    freq = torch.bincount(tr["y"], minlength=len(STATES)).float() / len(x)
    cw = (freq.max() / freq.clamp_min(1e-6)).sqrt()
    col_pw = torch.tensor(min(200.0, float((1 - tr["col"][tr["ok"]].mean()) / tr["col"][tr["ok"]].mean().clamp_min(1e-6))))
    prev_idx = torch.tensor([NAMES.index(n) for n in PREV])

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(x) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.1)
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(len(x))
        tot = 0.0
        for b in range(len(x) // args.batch):
            i = perm[b * args.batch:(b + 1) * args.batch]
            xb = x[i].clone()
            noisy = torch.rand(len(i)) < args.prev_noise
            if noisy.any():   # replace prev_* with a random one-hot
                oh = F.one_hot(torch.randint(len(STATES), (int(noisy.sum()),)), len(STATES)).float()
                xb[noisy.nonzero(as_tuple=True)[0].unsqueeze(1), prev_idx] = oh
            logit, col, prog, pas = m(xb)
            ok = tr["ok"][i]
            loss = (F.cross_entropy(logit, tr["y"][i], weight=cw, reduction="none") * tr["w"][i]).sum() \
                / (cw[tr["y"][i]] * tr["w"][i]).sum()
            if ok.any():
                loss = loss + 0.2 * F.binary_cross_entropy_with_logits(col[ok], tr["col"][i][ok], pos_weight=col_pw) \
                    + 0.05 * F.smooth_l1_loss(prog[ok], tr["prog"][i][ok] / 10.0) \
                    + 0.2 * F.binary_cross_entropy_with_logits(pas[ok], tr["pas"][i][ok])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        pv = predict(m, va["x"])[0].argmax(1)
        f1 = f1_score(va["y"], pv, average="macro")
        tag = ""
        if f1 > best:
            best, best_state, tag = f1, {k: v.clone() for k, v in m.state_dict().items()}, " *"
        print(f"epoch {ep + 1:2d} loss {tot / (len(x) // args.batch):.4f} val macro-F1 {f1:.4f} "
              f"acc {100 * float((pv == va['y']).float().mean()):.2f}% ({time.time() - t0:.0f}s){tag}", flush=True)

    m.load_state_dict(best_state)
    logit, col, prog, pas = predict(m, te["x"])
    pred, y = logit.argmax(1).numpy(), te["y"].numpy()
    acc, macro = float((pred == y).mean()), float(f1_score(y, pred, average="macro"))
    print(f"\ntest: accuracy {100 * acc:.2f}%, macro-F1 {macro:.3f} ({len(y):,} rows, 150 unseen episodes)\n")
    print(classification_report(y, pred, labels=range(len(STATES)), target_names=STATES, digits=3, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=range(len(STATES))), index=STATES, columns=STATES)
    print("confusion (rows = teacher, cols = model):\n", cm, "\n")
    ok = te["ok"].numpy()
    aux = {
        "collide_auc": float(roc_auc_score(te["col"][ok], col[ok])) if te["col"][ok].sum() > 0 else None,
        "pass_auc": float(roc_auc_score(te["pas"][ok], pas[ok])),
        "progress_mae_m": float((prog[ok] * 10 - te["prog"][ok]).abs().mean()),
    }
    print("aux heads:", {k: None if v is None else round(v, 3) for k, v in aux.items()})

    torch.set_num_threads(1)
    m.eval()
    with torch.no_grad():
        xb = te["x"][:1]
        for _ in range(50):
            m(xb)
        t1 = time.time()
        for _ in range(1000):
            m(xb)
    lat = (time.time() - t1)
    print(f"single-row forward {lat:.3f} ms (1 CPU thread)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(exist_ok=True)
    torch.save({"model": m.state_dict(), "width": args.width, "features": NAMES, "states": STATES}, out)
    out.with_suffix(".json").write_text(json.dumps({
        "data": args.data, "params": n_par, "args": vars(args), "test_accuracy": acc, "test_macro_f1": macro,
        "aux": aux, "single_row_latency_ms": round(lat, 3),
        "per_state": classification_report(y, pred, labels=range(len(STATES)), target_names=STATES,
                                           output_dict=True, zero_division=0),
        "confusion": cm.to_dict()}, indent=1))
    print(f"model → {out}")


if __name__ == "__main__":
    main()
