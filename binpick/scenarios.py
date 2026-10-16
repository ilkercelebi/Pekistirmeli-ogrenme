"""Anlatım için elle kurulmuş örnek kavrama durumları (arayüzün giriş sayfası ve testler)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .env import Action, Part


@dataclass
class Scenario:
    key: str
    title: str
    explanation: str
    parts: list
    action: tuple          # (açı derecesi, satır, sütun)
    kind: str = "parallel_jaw"


def _box(cy, cx, length, width, angle_deg=0.0, thickness=3.0):
    return Part("box", cy, cx, length, width, float(np.radians(angle_deg)), thickness)


SCENARIOS = [
    Scenario(
        "success", "Doğru nokta, doğru açı",
        "Parmaklar parçanın kısa kenarı boyunca kapanıyor, etraf boş, üstünde başka parça yok. Parça kalkar.",
        [_box(21, 21, 10, 4), _box(30, 12, 6, 6, 30, 2.5), _box(12, 31, 7, 3, 70, 2.0)],
        (90, 21, 21),
    ),
    Scenario(
        "collision", "Duvara fazla yakın",
        "Parça duvarın dibinde. Bu açıda parmaklardan biri kutu duvarına iniyor. Çarpışma: ekipman zarar görür.",
        [_box(5, 21, 10, 3), _box(24, 20, 8, 5, 20, 3.5)],
        (90, 4, 21),
    ),
    Scenario(
        "too_wide", "Yanlış açı",
        "Parmaklar parçanın uzun ekseni boyunca açılıyor. Parça açıklıktan geniş olduğu için parmaklar parçanın üstüne oturuyor.",
        [_box(21, 21, 13, 4), _box(32, 30, 6, 4, 60, 2.5)],
        (0, 21, 21),
    ),
    Scenario(
        "covered", "Üstü kapalı parça",
        "Noktanın altındaki parçanın üzerine başka bir parça binmiş. Alttaki parça yerinden kalkamaz.",
        [_box(21, 21, 13, 6), _box(21, 27, 6, 6, 0, 2.5)],
        (90, 21, 16),
    ),
]


def run_scenario(sc: Scenario):
    """Senaryoyu kayma olasılığı ve gürültü kapalı çalıştırır.

    (kavramadan önceki ortam, eylem, ödül, info) döndürür.
    """
    from . import build

    cfg = ExperimentConfig()
    cfg.gripper.kind = sc.kind
    cfg.env.obs_noise = 0.0
    g = cfg.gripper
    g.slip_base = g.slip_misalign = g.slip_offcenter = g.slip_seal = 0.0
    env, _ = build(cfg, seed=0)
    env.reset(parts=copy.deepcopy(sc.parts))
    deg, r, c = sc.action
    k = int(np.argmin(np.abs(np.degrees(env.angles) - deg)))
    action = Action(k, r, c)
    before = copy.deepcopy(env)
    _, reward, _, _, info = env.step(action)
    return before, action, reward, info
