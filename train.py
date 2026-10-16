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
import csv
import json
import time
from pathlib import Path

import numpy as np

from binpick import ExperimentConfig, build, make_agent, run_episode

FIELDS = ["episode", "epsilon", "return", "steps", "parts", "success", "collision", "fail",
          "success_rate", "collision_rate", "fail_rate", "cleared"]
EVAL_FIELDS = ["episode", "return", "success_rate", "collision_rate", "fail_rate", "cleared"]

EVAL_SEED0 = 1_000_000  # eğitimde hiç görülmeyen kutular


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


def greedy_eval(cfg, agent, n: int, seed: int) -> dict:
    env, fe = build(cfg, seed=seed)
    rows = [run_episode(env, agent, fe, learn=False, seed=EVAL_SEED0 + i) for i in range(n)]
    return {k: float(np.mean([r[k] for r in rows])) for k in EVAL_FIELDS if k != "episode"}


def main():
    a = parse_args()
    cfg = make_config(a)
    name = a.name or f"{a.algo}_{a.gripper}_s{a.seed}"
    run_dir = Path(a.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))

    env, fe = build(cfg, seed=a.seed)
    agent = make_agent(a.algo, fe.n_states, cfg.agent, seed=a.seed)
    print(f"[{name}] durum-eylem tablo boyutu: {fe.n_states}  ({' x '.join(map(str, fe.dims))})")

    t0 = time.time()
    with open(run_dir / "metrics.csv", "w", newline="") as mf, open(run_dir / "eval.csv", "w", newline="") as ef:
        mw, ew = csv.DictWriter(mf, FIELDS), csv.DictWriter(ef, EVAL_FIELDS)
        mw.writeheader()
        ew.writeheader()
        recent = []
        for ep in range(a.episodes):
            agent.set_episode(ep)
            s = run_episode(env, agent, fe, learn=True, seed=a.seed * 100_000 + ep)
            mw.writerow({"episode": ep, "epsilon": round(agent.epsilon, 4), "return": s["return"],
                         "steps": s["steps"], "parts": s["parts"], "success": s["success"],
                         "collision": s["collision"], "fail": s["fail"],
                         **{k: round(s[k], 4) for k in ("success_rate", "collision_rate", "fail_rate", "cleared")}})
            recent.append(s)

            if (ep + 1) % a.log_every == 0:
                m = {k: np.mean([r[k] for r in recent]) for k in ("return", "success_rate", "collision_rate", "cleared")}
                print(f"bölüm {ep + 1:5d}  ε={agent.epsilon:.2f}  getiri={m['return']:6.2f}  "
                      f"başarı={m['success_rate']:.2f}  çarpışma={m['collision_rate']:.2f}  "
                      f"boşaltma={m['cleared']:.2f}  ({time.time() - t0:.0f} sn)")
                recent = []
                mf.flush()

            if a.eval_every and ((ep + 1) % a.eval_every == 0 or ep + 1 == a.episodes):
                e = greedy_eval(cfg, agent, a.eval_episodes, seed=a.seed)
                ew.writerow({"episode": ep + 1, **{k: round(v, 4) for k, v in e.items()}})
                ef.flush()
                agent.save(run_dir / "agent.npz")
                print(f"  ↳ açgözlü değerlendirme: başarı={e['success_rate']:.2f}  "
                      f"çarpışma={e['collision_rate']:.2f}  boşaltma={e['cleared']:.2f}")

    agent.save(run_dir / "agent.npz")
    print(f"Bitti. Sonuçlar: {run_dir}")


if __name__ == "__main__":
    main()
