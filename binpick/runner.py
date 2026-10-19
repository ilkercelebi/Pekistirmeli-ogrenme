"""Bir bölümü (kutuyu boşaltma denemesi) çalıştıran ortak döngü."""

from __future__ import annotations

from .env import COLLISION, FAIL, STOP, SUCCESS


def run_episode(env, agent, fe, *, learn: bool, seed=None, on_step=None) -> dict:
    """Bölümü sonuna kadar oynatır; `learn=True` ise ajan her adımda güncellenir.

    on_step(env, obs, feats, action, reward, info) her adımdan sonra çağrılır
    (GIF ve arayüz bunu kullanır). `feats`, eylemin seçildiği durumun özellikleridir.
    """
    obs = env.reset(seed=seed)
    agent.allow_stop = env.cfg.allow_stop
    agent.begin_episode()
    feats = fe.compute(obs, env.fail_map, env.fail_streak)
    action, sid = agent.act(feats, explore=learn)

    stats = {"return": 0.0, "steps": 0, SUCCESS: 0, COLLISION: 0, FAIL: 0, STOP: 0, "parts": env.n_initial}
    while True:
        prev_feats = feats
        obs, reward, terminated, truncated, info = env.step(action)
        stats["return"] += reward
        stats["steps"] += 1
        stats[info["outcome"]] += 1
        agent.observe(info)

        next_feats = next_action = next_sid = None
        if not terminated:
            next_feats = fe.compute(obs, env.fail_map, env.fail_streak)
            next_action, next_sid = agent.act(next_feats, explore=learn)
        if learn:
            agent.learn(sid, reward, next_feats, next_sid, terminated)
        if on_step is not None:
            on_step(env, obs, prev_feats, action, reward, info)
        if terminated or truncated:
            break
        feats, action, sid = next_feats, next_action, next_sid

    # "dur" bir kavrama denemesi değildir: oranlar yalnızca kavrama denemeleri üzerinden hesaplanır
    stats["attempts"] = stats["steps"] - stats[STOP]
    stats["steps"] = stats["attempts"]
    n = max(stats["attempts"], 1)
    stats["success_rate"] = stats[SUCCESS] / n
    stats["collision_rate"] = stats[COLLISION] / n
    stats["fail_rate"] = stats[FAIL] / n
    stats["cleared"] = stats[SUCCESS] / max(stats["parts"], 1)
    return stats
