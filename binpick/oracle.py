"""Kâhin (oracle) politika: ulaşılabilecek başarımın üst sınırı.

Kâhin, ajanın göremediği gizli bilgiyi kullanır: parça kimlikleri, hangi parçanın üstü kapalı olduğu
ve her kavramanın gerçek kayma olasılığı. Her adımda, başarılı olabilecek bütün kavramalar arasından
kayma olasılığı en düşük olanı seçer; alınabilir parça kalmayınca durur. Öğrenen ajan aynı eylem
kümesiyle (aynı nokta ızgarası ve bilek açıları) bundan daha iyisini yapamaz; sonuç tablosundaki
"kâhin" satırı bu yüzden "ne kadar iyileştirme payı kaldı?" sorusunun cevabıdır.
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .env import COLLISION, FAIL, STOP, SUCCESS, Action


class _NoSlip:
    """Kayma çekilişini devre dışı bırakır: yalnızca kavramanın fiziksel olarak mümkün olup olmadığını sorarız."""

    def random(self):
        return 1.0


def slip_probability(env, action: Action) -> float:
    g, center = env.g, (action.row, action.col)
    p = env.parts[int(env.id_map[center])]
    if g.kind == "vacuum":
        cup = geometry.disk_mask(env.shape, center, g.cup_radius)
        same = (env.id_map[cup] == env.id_map[center]) & (np.abs(env.heightmap[cup] - env.heightmap[center]) < g.surface_tol)
        seal = float(same.mean())
        return g.slip_base + g.slip_seal * (1 - seal) / max(1 - g.min_seal, 1e-6) + g.slip_offcenter * env._offcenter(p, center) ** 2
    th = float(env.angles[action.angle_idx])
    diff = abs((th - (p.angle + np.pi / 2)) % np.pi)
    mis = min(diff, np.pi - diff) / (np.pi / 2)
    return (g.slip_base + g.slip_misalign * (1 - p.width / p.length) * mis ** 2
            + g.slip_offcenter * env._offcenter(p, center) ** 2)


def best_action(env) -> tuple[float, Action] | None:
    """Başarılı olabilecek kavramalar arasında kayma olasılığı en düşük olanı; yoksa None."""
    real, env.rng = env.rng, _NoSlip()
    best = None
    try:
        rows, cols = np.nonzero((env.id_map >= 0) & env.interior)
        for r, c in zip(rows, cols):
            if env.is_covered(int(env.id_map[r, c])):
                continue
            for k in range(len(env.angles)):
                out = env._grasp_vacuum((r, c)) if env.g.kind == "vacuum" else env._grasp_jaw((r, c), float(env.angles[k]))
                if out[0] == SUCCESS:
                    a = Action(k, int(r), int(c))
                    ps = slip_probability(env, a)
                    if best is None or ps < best[0]:
                        best = (ps, a)
    finally:
        env.rng = real
    return best


def run_oracle_episode(env, seed=None) -> dict:
    """run_episode ile aynı istatistikleri döndürür."""
    env.reset(seed=seed)
    stats = {"return": 0.0, "steps": 0, SUCCESS: 0, COLLISION: 0, FAIL: 0, STOP: 0, "parts": env.n_initial}
    while env.n_alive and env.steps < env.max_steps:
        b = best_action(env)
        if b is None:
            stats[STOP] += 1
            break
        _, reward, term, trunc, info = env.step(b[1])
        stats["return"] += reward
        stats["steps"] += 1
        stats[info["outcome"]] += 1
        if term or trunc:
            break
    n = max(stats["steps"], 1)
    stats["attempts"] = stats["steps"]
    stats["success_rate"] = stats[SUCCESS] / n
    stats["collision_rate"] = stats[COLLISION] / n
    stats["fail_rate"] = stats[FAIL] / n
    stats["cleared"] = stats[SUCCESS] / max(stats["parts"], 1)
    return stats
