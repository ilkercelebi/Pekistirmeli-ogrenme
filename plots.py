"""Eğitim eğrilerini çizer.

Örnek:
    python plots.py results/q_learning_parallel_jaw_s0 results/sarsa_parallel_jaw_s0 --window 100

Karşılaştırma değerlendirmesi (evaluate.py) yapıldıysa rastgele ve sezgisel
yöntemlerin test başarımı yatay çizgi olarak eklenir.
Çıktılar: results/plots/learning_curves.png, results/plots/greedy_eval.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from binpick.viz import moving_average  # noqa: E402

PANELS = [("return", "Bölüm getirisi"), ("success_rate", "Başarı oranı"),
          ("collision_rate", "Çarpışma oranı"), ("cleared", "Boşaltılan parça oranı")]


def read_csv(path: Path) -> dict:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in (rows[0].keys() if rows else [])}


def baselines(gripper: str) -> dict:
    path = Path("results") / f"eval_{gripper}" / "summary.csv"
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["method"] in ("random", "heuristic"):
                out[r["method"]] = {k: float(r[k]) for k, _ in PANELS}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--out", default="results/plots")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    gripper = json.loads((Path(a.runs[0]) / "config.json").read_text())["gripper"]["kind"]
    base = baselines(gripper)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120)
    for run in a.runs:
        m = read_csv(Path(run) / "metrics.csv")
        for ax, (k, title) in zip(axes.flat, PANELS):
            ax.plot(m["episode"], moving_average(m[k], a.window), label=Path(run).name)
    for ax, (k, title) in zip(axes.flat, PANELS):
        for name, style in (("random", ":"), ("heuristic", "--")):
            if name in base:
                ax.axhline(base[name][k], color="gray", ls=style, label=f"{name} (test)")
        ax.set_title(title)
        ax.set_xlabel("bölüm")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Eğitim eğrileri (hareketli ortalama, pencere={a.window})")
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=120)
    for run in a.runs:
        e = read_csv(Path(run) / "eval.csv")
        if not e:
            continue
        for ax, k in zip(axes, ("success_rate", "collision_rate", "cleared")):
            ax.plot(e["episode"], e[k], marker="o", label=Path(run).name)
    for ax, (k, title) in zip(axes, [("success_rate", "Başarı"), ("collision_rate", "Çarpışma"), ("cleared", "Boşaltma")]):
        ax.set_title(f"Açgözlü politika: {title}")
        ax.set_xlabel("eğitim bölümü")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "greedy_eval.png")
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
