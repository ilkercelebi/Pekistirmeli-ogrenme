"""Eğitim döngüsü: hem `train.py` (komut satırı) hem arayüz bunu kullanır."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .agents import make_agent
from .runner import run_episode

FIELDS = ["episode", "epsilon", "return", "steps", "parts", "success", "collision", "fail",
          "success_rate", "collision_rate", "fail_rate", "cleared"]
EVAL_FIELDS = ["episode", "return", "success_rate", "collision_rate", "fail_rate", "cleared"]

EVAL_SEED0 = 1_000_000  # eğitimde hiç görülmeyen kutular


def build(cfg, seed=None):
    from . import build as _build

    return _build(cfg, seed=seed)


def greedy_eval(cfg, agent, n: int, seed: int = 0) -> dict:
    """Keşif kapalı (ε=0) politikanın sabit test kutularındaki ortalama başarımı."""
    env, fe = build(cfg, seed=seed)
    rows = [run_episode(env, agent, fe, learn=False, seed=EVAL_SEED0 + i) for i in range(n)]
    return {k: float(np.mean([r[k] for r in rows])) for k in EVAL_FIELDS if k != "episode"}


def train(cfg, episodes: int, *, seed: int = 0, run_dir: str | Path | None = None,
          eval_every: int = 250, eval_episodes: int = 30, callback=None):
    """Ajanı eğitir; `run_dir` verilirse config.json, metrics.csv, eval.csv ve agent.npz yazar.

    callback(episode, stats, agent, eval_row) her bölümden sonra çağrılır;
    eval_row yalnızca değerlendirme yapılan bölümlerde doludur. callback False
    döndürürse eğitim erken durdurulur.
    """
    env, fe = build(cfg, seed=seed)
    agent = make_agent(cfg.agent.algo, fe.n_states, cfg.agent, seed=seed)

    mf = ef = mw = ew = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        mf = open(run_dir / "metrics.csv", "w", newline="")
        ef = open(run_dir / "eval.csv", "w", newline="")
        mw, ew = csv.DictWriter(mf, FIELDS), csv.DictWriter(ef, EVAL_FIELDS)
        mw.writeheader()
        ew.writeheader()

    history, evals = [], []
    try:
        for ep in range(episodes):
            agent.set_episode(ep)
            s = run_episode(env, agent, fe, learn=True, seed=seed * 100_000 + ep)
            row = {"episode": ep, "epsilon": round(agent.epsilon, 4), "return": s["return"],
                   "steps": s["steps"], "parts": s["parts"], "success": s["success"],
                   "collision": s["collision"], "fail": s["fail"],
                   **{k: round(s[k], 4) for k in ("success_rate", "collision_rate", "fail_rate", "cleared")}}
            history.append(row)
            if mw:
                mw.writerow(row)

            eval_row = None
            if eval_every and ((ep + 1) % eval_every == 0 or ep + 1 == episodes):
                e = greedy_eval(cfg, agent, eval_episodes, seed=seed)
                eval_row = {"episode": ep + 1, **{k: round(v, 4) for k, v in e.items()}}
                evals.append(eval_row)
                if ew:
                    ew.writerow(eval_row)
                    mf.flush()
                    ef.flush()
                    agent.save(run_dir / "agent.npz")

            if callback is not None and callback(ep, row, agent, eval_row) is False:
                break
    finally:
        if run_dir is not None:
            agent.save(run_dir / "agent.npz")
            mf.close()
            ef.close()
    return agent, history, evals
