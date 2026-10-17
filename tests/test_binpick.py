"""Ortam kuralları, özellik çıkarıcı ve ajanlar için temel testler."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binpick import (  # noqa: E402
    COLLISION,
    FAIL,
    SUCCESS,
    Action,
    ExperimentConfig,
    Part,
    build,
    make_agent,
    run_episode,
)


def make_env(kind="parallel_jaw", no_slip=True):
    cfg = ExperimentConfig()
    cfg.gripper.kind = kind
    cfg.env.obs_noise = 0.0
    if no_slip:
        cfg.gripper.slip_base = cfg.gripper.slip_misalign = cfg.gripper.slip_offcenter = cfg.gripper.slip_seal = 0.0
    env, fe = build(cfg, seed=0)
    return cfg, env, fe


def box(cy, cx, length=8, width=4, angle=0.0, thickness=3):
    return Part("box", cy, cx, length, width, angle, thickness)


def angle_idx(env, deg):
    return int(np.argmin(np.abs(np.degrees(env.angles) - deg)))


def test_reset_generates_parts_inside_bin():
    _, env, _ = make_env()
    env.reset(seed=3)
    assert env.n_alive >= 1
    for p in env.parts:
        assert not (p.mask & env.wall_mask).any()


def test_jaw_success_on_isolated_part():
    _, env, _ = make_env()
    env.reset(parts=[box(21, 21, angle=0.0)])
    # parça x yönünde uzun; kısa eksen boyunca (90°) sıkıştır
    _, r, term, _, info = env.step(Action(angle_idx(env, 90), 21, 21))
    assert info["outcome"] == SUCCESS and r == 1.0 and term


def test_jaw_too_wide_when_closing_along_long_axis():
    _, env, _ = make_env()
    env.reset(parts=[box(21, 21, length=12, width=4, angle=0.0)])
    _, r, _, _, info = env.step(Action(angle_idx(env, 0), 21, 21))
    assert info["outcome"] == FAIL and math.isclose(r, -0.2)


def test_wall_collision():
    _, env, _ = make_env()
    env.reset(parts=[box(4, 21, length=6, width=3, angle=0.0)])
    _, r, _, _, info = env.step(Action(angle_idx(env, 90), 3, 21))
    assert info["outcome"] == COLLISION and r == -1.0


def test_empty_grasp_fails():
    _, env, _ = make_env()
    env.reset(parts=[box(10, 10)])
    _, r, _, _, info = env.step(Action(0, 30, 30))
    assert info["outcome"] == FAIL and info["part"] is None


def test_covered_part_cannot_be_lifted():
    _, env, _ = make_env("vacuum")
    env.reset(parts=[box(21, 21, length=12, width=6), box(21, 26, length=6, width=6)])
    # alttaki parçanın açıkta kalan sol ucu
    _, _, _, _, info = env.step(Action(0, 21, 17))
    assert info["outcome"] == FAIL and "üstünde" in info["reason"]
    # üstteki parça alınabilir
    _, _, _, _, info = env.step(Action(0, 21, 26))
    assert info["outcome"] == SUCCESS


def test_vacuum_face_feature_sees_cover_beyond_neighbor_radius():
    _, env, fe = make_env("vacuum")
    obs = env.reset(parts=[box(21, 21, length=14, width=6), box(21, 26, length=6, width=6)])
    raw = fe.compute(obs).raw
    # alttaki parçanın üstteki parçadan uzak ucu: yerel "yakında yüksek" görmez, yüzey özelliği görür
    assert raw["higher"][21, 16] == 0
    assert raw["face_higher"][21, 16] == 1
    # üstteki parçanın yüzü açık
    assert raw["face_higher"][21, 26] == 0


def test_vacuum_face_feature_flag_sets_table_size():
    cfg = ExperimentConfig()
    cfg.gripper.kind = "vacuum"
    assert build(cfg)[1].n_states == 864           # + hafıza (denendi)
    cfg.features.face_feature = False
    _, fe = build(cfg)
    assert fe.n_states == 432 and "yüzey_kapalı" not in fe.names
    # sezgisel + yüzey bilgisi için ham harita yine hesaplanır
    assert fe.compute(build(cfg, seed=0)[0].reset(seed=1)).raw["face_higher"] is not None


def test_heuristic_face_skips_covered_face():
    """Küçük bir parça (vantuz oturmaz) büyük parçanın üstünde: sezgisel büyük parçayı seçip başarısız olur,
    yüzey bilgisi verilen sezgisel açıktaki alçak parçayı seçer."""
    _, env, fe = make_env("vacuum")

    def scene():
        return env.reset(parts=[box(21, 21, length=14, width=8, thickness=5), box(21, 17, length=2, width=1, thickness=2),
                                box(10, 34, length=6, width=6, thickness=2)])

    feats = fe.compute(scene())
    outcomes = {}
    for name in ("heuristic", "heuristic_face"):
        scene()
        action, _ = make_agent(name, 0, seed=0).act(feats)
        outcomes[name] = env.step(action)[4]["outcome"]
    assert outcomes == {"heuristic": FAIL, "heuristic_face": SUCCESS}


@pytest.mark.parametrize("kind", ["parallel_jaw", "vacuum"])
def test_feature_ids_in_range(kind):
    _, env, fe = make_env(kind)
    feats = fe.compute(env.reset(seed=1))
    assert feats.state_ids.shape == (len(env.angles),) + env.shape
    ids = feats.state_ids[feats.valid]
    assert ids.min() >= 0 and ids.max() < fe.n_states


def test_feature_wall_matches_env_collision():
    """Özellik çıkarıcının 'duvara çarpar' dediği her eylem, ortamda da çarpışmadır."""
    _, env, fe = make_env()
    env.reset(parts=[])
    feats = fe.compute(env.observe())
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(len(env.angles)))
        r, c = (int(v) for v in rng.integers(2, 42, size=2))
        predicted = feats.raw["wall"][k][r, c] == 0
        assert predicted == env._wall_hit((r, c), float(env.angles[k]))


@pytest.mark.parametrize("algo", ["q_learning", "sarsa", "random", "heuristic", "heuristic_face"])
def test_agents_run_and_learn(algo):
    cfg, env, fe = make_env(no_slip=False)
    agent = make_agent(algo, fe.n_states, cfg.agent, seed=0)
    for ep in range(3):
        agent.set_episode(ep)
        stats = run_episode(env, agent, fe, learn=True, seed=ep)
        assert stats["steps"] >= 1
    if agent.learns:
        assert agent.N.sum() > 0


@pytest.mark.parametrize("kind", ["parallel_jaw", "vacuum"])
def test_heuristic_baseline_is_meaningful(kind):
    """Sezgisel karşılaştırma yöntemi aynı başarısız kavramayı tekrarlayıp takılmamalı."""
    cfg, env, fe = make_env(kind, no_slip=False)
    agent = make_agent("heuristic", 0, seed=0)
    rows = [run_episode(env, agent, fe, learn=False, seed=2_000_000 + i) for i in range(6)]
    assert np.mean([r["success_rate"] for r in rows]) > 0.4
    assert np.mean([r["collision_rate"] for r in rows]) == 0.0


def test_jaw_alignment_and_axis_center_features():
    """Kaymayı belirleyen iki etken gözlemden ölçülür: kısa kenardan sıkma ve uzun eksende ortalanma."""
    _, env, fe = make_env()
    assert fe.names[-4:] == ["hizalama", "eksen_ortası", "yüzey_kapalı", "denendi"]
    feats = fe.compute(env.reset(parts=[box(21, 21, length=12, width=4, angle=0.0)]))
    k0, k90 = angle_idx(env, 0), angle_idx(env, 90)

    def bins(k, r, c):
        return fe.describe(int(feats.state_ids[k, r, c]))

    # parça x yönünde uzun: 90° kısa kenardan sıkar, 0° uzun kenardan
    assert bins(k90, 21, 21)["hizalama"] == 0 and bins(k0, 21, 21)["hizalama"] == 1
    # merkez "ortada", uç "uçta"
    assert bins(k90, 21, 21)["eksen_ortası"] == 0 and bins(k90, 21, 16)["eksen_ortası"] == 2


def test_old_runs_without_alignment_flag_still_load():
    cfg = ExperimentConfig.from_dict({"gripper": {"kind": "parallel_jaw"}, "features": {}})
    assert cfg.features.alignment_features is False
    assert build(cfg)[1].n_states == 1296


def test_memory_marks_failures_and_clears_on_success():
    cfg, env, _ = make_env("vacuum")
    cfg.features.memory_feature = True
    env, fe = build(cfg, seed=0)
    env.reset(parts=[box(21, 21, length=10, width=6), box(10, 10, length=6, width=6)])
    assert "denendi" in fe.names and not env.fail_map.any()
    env.step(Action(0, 30, 30))                         # boş nokta: başarısız
    assert env.fail_map[30, 30] and not env.fail_map[21, 21]
    feats = fe.compute(env.observe(), env.fail_map)
    assert fe.describe(int(feats.state_ids[0, 30, 30]))["denendi"] == 1
    assert fe.describe(int(feats.state_ids[0, 21, 21]))["denendi"] == 0
    env.step(Action(0, 21, 21))                         # başarılı alım hafızayı siler
    assert not env.fail_map.any()
