"""Bir ajanın kutuyu boşaltmasını GIF olarak kaydeder (README ve video için).

Örnek:
    python make_gif.py --run results/q_learning_parallel_jaw_s0 --seed 7 --out results/gifs/qlearning.gif
    python make_gif.py --agent random --seed 7 --out results/gifs/random.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from binpick import ExperimentConfig, build, load_run, make_agent, run_episode
from binpick.viz import render, save_gif


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=None, help="eğitim klasörü (Q tablosu)")
    p.add_argument("--agent", choices=["random", "heuristic"], default=None)
    p.add_argument("--gripper", choices=["parallel_jaw", "vacuum"], default="parallel_jaw")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    if a.run:
        cfg, agent = load_run(a.run, seed=a.seed)
        label = Path(a.run).name
    else:
        cfg = ExperimentConfig()
        cfg.gripper.kind = a.gripper
        agent = make_agent(a.agent or "random", 0, seed=a.seed)
        label = agent.name

    env, fe = build(cfg, seed=a.seed)
    frames = []
    obs = env.reset(seed=a.seed)
    frames.append(render(env, agent=agent, feats=fe.compute(obs), title=f"{label} · başlangıç"))

    def on_step(env, obs, feats, action, reward, info):
        frames.append(render(env, agent=agent, feats=feats, action=action, info=info,
                             title=f"{label} · adım {info['step']} · kalan parça {info['parts_left']}"))

    stats = run_episode(env, agent, fe, learn=False, seed=a.seed, on_step=on_step)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, a.out, fps=a.fps)
    print(f"{a.out}: {stats['steps']} adım, başarı={stats['success']}, çarpışma={stats['collision']}, "
          f"başarısız={stats['fail']}, boşaltma={stats['cleared']:.2f}")


if __name__ == "__main__":
    main()
