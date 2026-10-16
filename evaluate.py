"""Eğitilmiş ajanları ve karşılaştırma yöntemlerini aynı test kutularında karşılaştırır.

Örnek:
    python evaluate.py --runs results/q_learning_parallel_jaw_s0 results/sarsa_parallel_jaw_s0
    python evaluate.py --gripper vacuum --runs results/q_learning_vacuum_s0

Rastgele ve sezgisel (en yüksek nokta) yöntemler her zaman eklenir. Tüm yöntemler
aynı tohumlu (seed) test kutularında, keşif kapalı (ε=0) çalışır.
Çıktılar: results/eval_<tutucu>/summary.csv, summary.md, bars.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from binpick import ExperimentConfig, build, load_run, make_agent, run_episode  # noqa: E402

TEST_SEED0 = 2_000_000
METRICS = ["return", "success_rate", "collision_rate", "fail_rate", "cleared", "steps"]
LABELS = {"return": "Getiri", "success_rate": "Başarı oranı", "collision_rate": "Çarpışma oranı",
          "fail_rate": "Başarısız kavrama oranı", "cleared": "Boşaltılan parça oranı", "steps": "Adım sayısı"}


def evaluate(cfg, agent, n: int) -> dict:
    env, fe = build(cfg, seed=0)
    rows = [run_episode(env, agent, fe, learn=False, seed=TEST_SEED0 + i) for i in range(n)]
    out = {}
    for k in METRICS:
        vals = np.array([r[k] for r in rows], dtype=float)
        out[k] = vals.mean()
        out[k + "_std"] = vals.std()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="eğitim klasörleri")
    p.add_argument("--gripper", choices=["parallel_jaw", "vacuum"], default=None,
                   help="koşu verilmezse karşılaştırma yöntemleri için tutucu")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    entries = []
    base_cfg = None
    for run in a.runs:
        cfg, agent = load_run(run, seed=0)
        base_cfg = base_cfg or cfg
        if cfg.gripper.kind != base_cfg.gripper.kind:
            raise SystemExit("Tüm koşular aynı tutucu tipini kullanmalı.")
        entries.append((Path(run).name, cfg, agent))
    if base_cfg is None:
        base_cfg = ExperimentConfig()
        base_cfg.gripper.kind = a.gripper or "parallel_jaw"

    for name in ("random", "heuristic"):
        entries.insert(0 if name == "random" else 1, (name, base_cfg, make_agent(name, 0, seed=0)))

    out_dir = Path(a.out or f"results/eval_{base_cfg.gripper.kind}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, cfg, agent in entries:
        r = evaluate(cfg, agent, a.episodes)
        results.append((name, r))
        print(f"{name:35s} başarı={r['success_rate']:.3f}  çarpışma={r['collision_rate']:.3f}  "
              f"boşaltma={r['cleared']:.3f}  getiri={r['return']:.2f}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [m for k in METRICS for m in (k, k + "_std")])
        for name, r in results:
            w.writerow([name] + [round(r[m], 4) for k in METRICS for m in (k, k + "_std")])

    lines = ["| Yöntem | " + " | ".join(LABELS[k] for k in METRICS) + " |",
             "|---" * (len(METRICS) + 1) + "|"]
    for name, r in results:
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.3f} ± {r[k + '_std']:.3f}" for k in METRICS) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    keys = ["success_rate", "collision_rate", "fail_rate", "cleared"]
    x = np.arange(len(keys))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    for i, (name, r) in enumerate(results):
        ax.bar(x + i * width - 0.4 + width / 2, [r[k] for k in keys], width, label=name)
    ax.set_xticks(x, [LABELS[k] for k in keys])
    ax.set_ylim(0, 1)
    ax.set_title(f"Test kutularında karşılaştırma ({a.episodes} bölüm, {base_cfg.gripper.kind})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bars.png")
    print(f"Kaydedildi: {out_dir}")


if __name__ == "__main__":
    main()
