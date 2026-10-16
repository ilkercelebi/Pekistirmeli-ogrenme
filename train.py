"""Tablo tabanlı ajanı (Q-learning veya SARSA) eğitir.

Örnek:
    python train.py --algo q_learning --gripper parallel_jaw --episodes 3000
    python train.py --algo sarsa --gripper vacuum --episodes 3000 --name sarsa_vakum

Çıktılar results/<isim>/ altına yazılır:
    config.json   tüm parametreler
    metrics.csv   her eğitim bölümünün istatistikleri (keşifli politika)
    eval.csv      belirli aralıklarla açgözlü (ε=0) politikanın sabit kutulardaki başarımı
    agent.npz     öğrenilen Q tablosu ve ziyaret sayıları
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from binpick import ExperimentConfig, build
from binpick.training import train


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["q_learning", "sarsa"], default="q_learning")
    p.add_argument("--gripper", choices=["parallel_jaw", "vacuum"], default="parallel_jaw")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--alpha-mode", choices=["constant", "visit"], default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eps-start", type=float, default=None)
    p.add_argument("--eps-end", type=float, default=None)
    p.add_argument("--eps-decay", type=int, default=None, help="ε'nun eps-end'e indiği bölüm sayısı")
    p.add_argument("--n-angles", type=int, default=None)
    p.add_argument("--stride", type=int, default=None, help="aday kavrama noktası ızgara adımı")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-episodes", type=int, default=30)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", default="results")
    p.add_argument("--name", default=None)
    return p.parse_args()


def make_config(a) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.gripper.kind = a.gripper
    cfg.agent.algo = a.algo
    for attr, val in [("alpha", a.alpha), ("alpha_mode", a.alpha_mode), ("gamma", a.gamma),
                      ("eps_start", a.eps_start), ("eps_end", a.eps_end), ("eps_decay_episodes", a.eps_decay)]:
        if val is not None:
            setattr(cfg.agent, attr, val)
    if a.n_angles is not None:
        cfg.gripper.n_angles = a.n_angles
    if a.stride is not None:
        cfg.features.candidate_stride = a.stride
    return cfg


def main():
    a = parse_args()
    cfg = make_config(a)
    name = a.name or f"{a.algo}_{a.gripper}_s{a.seed}"
    run_dir = Path(a.out) / name

    _, fe = build(cfg)
    print(f"[{name}] durum-eylem tablo boyutu: {fe.n_states}  ({' x '.join(map(str, fe.dims))})")

    t0 = time.time()
    recent = []

    def log(ep, row, agent, eval_row):
        recent.append(row)
        if (ep + 1) % a.log_every == 0:
            m = {k: np.mean([r[k] for r in recent]) for k in ("return", "success_rate", "collision_rate", "cleared")}
            print(f"bölüm {ep + 1:5d}  ε={agent.epsilon:.2f}  getiri={m['return']:6.2f}  "
                  f"başarı={m['success_rate']:.2f}  çarpışma={m['collision_rate']:.2f}  "
                  f"boşaltma={m['cleared']:.2f}  ({time.time() - t0:.0f} sn)")
            recent.clear()
        if eval_row:
            print(f"  ↳ açgözlü değerlendirme: başarı={eval_row['success_rate']:.2f}  "
                  f"çarpışma={eval_row['collision_rate']:.2f}  boşaltma={eval_row['cleared']:.2f}")

    train(cfg, a.episodes, seed=a.seed, run_dir=run_dir, eval_every=a.eval_every,
          eval_episodes=a.eval_episodes, callback=log)
    print(f"Bitti. Sonuçlar: {run_dir}")


if __name__ == "__main__":
    main()
