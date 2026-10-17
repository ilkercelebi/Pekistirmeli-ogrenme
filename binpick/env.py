"""Bin picking simülasyonu (kuşbakışı 2.5B yükseklik haritası).

Kutuya rastgele parçalar sırayla bırakılır; her parça altındaki en yüksek
noktanın üstüne oturur, böylece üst üste binen bir yığın oluşur. Ajan yalnızca
gürültülü yükseklik haritasını (derinlik kamerası) görür. Kavramanın sonucu ise
ortamın bildiği gerçek parça bilgisiyle, aşağıdaki kurallarla belirlenir:

  1. Tutucu ayak izi kutu duvarına değiyorsa         -> ÇARPIŞMA  (-1)
  2. Kavrama noktasında parça yoksa                  -> BAŞARISIZ (-0.2)
  3. Parmaklı: parmaklar inemiyorsa (parça çok geniş
     ya da komşu parça engel)                         -> BAŞARISIZ
     Vakumlu: yüzey sızdırmazlığı yetersizse          -> BAŞARISIZ
  4. Parçanın üstünde başka parça varsa               -> BAŞARISIZ
  5. Kavrama kalitesine bağlı olasılıkla kayıp düşme  -> BAŞARISIZ
  6. Aksi halde parça kaldırılır                      -> BAŞARILI  (+1)

Arayüz Gymnasium'a benzer: reset() -> obs, step(a) -> obs, r, terminated, truncated, info
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry
from .config import EnvConfig, GripperConfig

SUCCESS = "success"
COLLISION = "collision"
FAIL = "fail"

REASONS = {
    "ok": "parça başarıyla kaldırıldı",
    "wall": "tutucu kutu duvarına çarptı",
    "empty": "boş noktaya kavrama",
    "too_wide": "parça tutucu açıklığından geniş",
    "finger_blocked": "parmaklar komşu parçaya takıldı",
    "covered": "parçanın üstünde başka parça var",
    "slip": "parça tutucudan kayıp düştü",
    "seal": "vakum sızdırıyor (parça kenarı)",
}


@dataclass
class Action:
    angle_idx: int
    row: int
    col: int


@dataclass
class Part:
    kind: str           # "box" veya "disc"
    cy: float
    cx: float
    length: float
    width: float
    angle: float        # uzun kenarın yönü
    thickness: float
    base: float = 0.0
    mask: np.ndarray | None = None
    alive: bool = True

    @property
    def top(self) -> float:
        return self.base + self.thickness


class BinPickingEnv:
    def __init__(self, env_cfg: EnvConfig | None = None, gripper_cfg: GripperConfig | None = None, seed=None):
        self.cfg = env_cfg or EnvConfig()
        self.g = gripper_cfg or GripperConfig()
        self.rng = np.random.default_rng(seed)
        self.angles = geometry.angle_values(self.g.angle_count)

        n, t = self.cfg.size, self.cfg.wall_thickness
        self.shape = (n, n)
        self.wall_mask = np.ones(self.shape, dtype=bool)
        self.wall_mask[t : n - t, t : n - t] = False
        self.interior = ~self.wall_mask

        self.parts: list[Part] = []
        self.heightmap = np.zeros(self.shape)
        self.id_map = np.full(self.shape, -1, dtype=int)
        self.steps = 0
        self.max_steps = 0
        self.n_initial = 0
        # Robotun bu bölümdeki başarısız kavrama noktaları (kendi hafızası; kutu değişince silinir).
        self.fail_map = np.zeros(self.shape, dtype=bool)

    # ------------------------------------------------------------ kurulum

    def reset(self, seed=None, parts: list[Part] | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.parts = []
        if parts is None:
            n_parts = int(self.rng.integers(self.cfg.n_parts_min, self.cfg.n_parts_max + 1))
            self._rebuild()
            for _ in range(n_parts):
                part = self._spawn_part()
                if part is not None:
                    self.parts.append(part)
                    self._rebuild()
        else:
            self._rebuild()
            for p in parts:
                self._drop(p)
                self.parts.append(p)
                self._rebuild()
        self.n_initial = len(self.parts)
        self.steps = 0
        self.fail_map = np.zeros(self.shape, dtype=bool)
        self.max_steps = max(1, math.ceil(self.n_initial * self.cfg.max_steps_per_part))
        return self.observe()

    def _part_mask(self, p: Part) -> np.ndarray:
        if p.kind == "disc":
            return geometry.disk_mask(self.shape, (p.cy, p.cx), p.length / 2)
        return geometry.rect_mask(self.shape, (p.cy, p.cx), p.length, p.width, p.angle)

    def _drop(self, p: Part) -> None:
        """Parçayı altındaki en yüksek noktanın üzerine oturtur."""
        p.mask = self._part_mask(p)
        p.base = float(self.heightmap[p.mask].max()) if p.mask.any() else 0.0

    def _spawn_part(self, tries: int = 50) -> Part | None:
        c = self.cfg
        mid = (c.size - 1) / 2
        for _ in range(tries):
            if self.rng.random() < c.box_prob:
                length = self.rng.uniform(*c.box_length)
                width = min(self.rng.uniform(*c.box_width), length)
                kind = "box"
            else:
                length = width = self.rng.uniform(*c.disc_diameter)
                kind = "disc"
            cy, cx = self.rng.normal(mid, c.pile_sigma, size=2)
            p = Part(
                kind=kind,
                cy=float(cy),
                cx=float(cx),
                length=float(length),
                width=float(width),
                angle=float(self.rng.uniform(0, np.pi)),
                thickness=float(self.rng.uniform(*c.thickness)),
            )
            mask = self._part_mask(p)
            if mask.sum() < 4 or (mask & self.wall_mask).any():
                continue
            self._drop(p)
            if p.top > c.max_stack_height:
                continue
            return p
        return None

    def _rebuild(self) -> None:
        self.heightmap = np.where(self.wall_mask, self.cfg.wall_height, 0.0)
        self.id_map = np.full(self.shape, -1, dtype=int)
        # Sonra bırakılan parça, çakıştığı yerde öncekilerin üstündedir.
        for i, p in enumerate(self.parts):
            if p.alive:
                self.heightmap[p.mask] = p.top
                self.id_map[p.mask] = i

    # ------------------------------------------------------------ gözlem

    def observe(self) -> np.ndarray:
        obs = self.heightmap.copy()
        if self.cfg.obs_noise > 0:
            noise = self.rng.normal(0.0, self.cfg.obs_noise, size=self.shape)
            obs[self.interior] += noise[self.interior]
        return obs

    @property
    def n_alive(self) -> int:
        return sum(p.alive for p in self.parts)

    def is_covered(self, idx: int) -> bool:
        """Sonradan bırakılıp bu parçayla çakışan canlı bir parça var mı."""
        mask = self.parts[idx].mask
        return any(q.alive and (q.mask & mask).any() for q in self.parts[idx + 1 :])

    # ------------------------------------------------------------ adım

    def step(self, action: Action):
        theta = float(self.angles[action.angle_idx])
        center = (action.row, action.col)
        anim = self._contact_heights(center, theta)
        if self.g.kind == "vacuum":
            outcome, code, pid = self._grasp_vacuum(center)
        else:
            outcome, code, pid = self._grasp_jaw(center, theta)
        if outcome == COLLISION:
            anim["stop_z"] = self.cfg.wall_height

        if outcome == SUCCESS:
            self.parts[pid].alive = False
            self._rebuild()
            reward = self.cfg.reward_success
            self.fail_map[:] = False          # kutu değişti: eski başarısızlıklar artık geçerli olmayabilir
        else:
            reward = self.cfg.reward_collision if outcome == COLLISION else self.cfg.reward_fail
            self.fail_map |= geometry.disk_mask(self.shape, center, self.cfg.memory_radius)

        self.steps += 1
        terminated = self.n_alive == 0
        truncated = not terminated and self.steps >= self.max_steps
        info = {
            "outcome": outcome,
            "reason": REASONS[code],
            "code": code,
            "anim": anim,
            "part": pid,
            "theta_deg": math.degrees(theta),
            "parts_left": self.n_alive,
            "step": self.steps,
        }
        return self.observe(), reward, terminated, truncated, info

    def _contact_heights(self, center, theta) -> dict:
        """Animasyon için: kavrama yüzeyinin yüksekliği ve tutucunun inebildiği en alt nokta."""
        top = float(self.heightmap[center])
        if self.g.kind == "vacuum":
            return {"top_z": top, "stop_z": top}
        fingers = geometry.jaw_finger_mask(self.shape, center, theta, self.g)
        under = float(np.where(self.wall_mask, 0.0, self.heightmap)[fingers].max()) if fingers.any() else 0.0
        return {"top_z": top, "stop_z": max(top - self.g.grasp_depth, under, 0.0)}

    def _wall_hit(self, center, theta) -> bool:
        return bool((geometry.body_mask(self.shape, center, theta, self.g) & self.wall_mask).any())

    def _offcenter(self, p: Part, center) -> float:
        """Kavrama noktasının parçanın merkezinden, uzun eksen boyunca normalize uzaklığı."""
        dy, dx = center[0] - p.cy, center[1] - p.cx
        along = abs(dx * math.cos(p.angle) + dy * math.sin(p.angle))
        return min(1.0, along / max(p.length / 2, 1e-6))

    def _grasp_jaw(self, center, theta):
        g = self.g
        if self._wall_hit(center, theta):
            return COLLISION, "wall", None
        pid = int(self.id_map[center])
        if pid < 0:
            return FAIL, "empty", None

        target = self.heightmap[center] - g.grasp_depth
        fingers = geometry.jaw_finger_mask(self.shape, center, theta, g)
        if fingers.any() and self.heightmap[fingers].max() > target + 1e-6:
            if (self.id_map[fingers] == pid).any():
                return FAIL, "too_wide", pid
            return FAIL, "finger_blocked", pid
        if self.is_covered(pid):
            return FAIL, "covered", pid

        p = self.parts[pid]
        # En iyi kapanma yönü parçanın kısa eksenidir (uzun eksene dik).
        diff = abs((theta - (p.angle + np.pi / 2)) % np.pi)
        misalign = min(diff, np.pi - diff) / (np.pi / 2)          # 0..1
        elongation = 1.0 - p.width / p.length                     # disk için 0
        p_slip = (
            g.slip_base
            + g.slip_misalign * elongation * misalign**2
            + g.slip_offcenter * self._offcenter(p, center) ** 2
        )
        if self.rng.random() < min(p_slip, 1.0):
            return FAIL, "slip", pid
        return SUCCESS, "ok", pid

    def _grasp_vacuum(self, center):
        g = self.g
        if self._wall_hit(center, 0.0):
            return COLLISION, "wall", None
        pid = int(self.id_map[center])
        if pid < 0:
            return FAIL, "empty", None

        cup = geometry.disk_mask(self.shape, center, g.cup_radius)
        same = (self.id_map[cup] == pid) & (np.abs(self.heightmap[cup] - self.heightmap[center]) < g.surface_tol)
        seal = float(same.mean())
        if seal < g.min_seal:
            return FAIL, "seal", pid
        if self.is_covered(pid):
            return FAIL, "covered", pid

        p = self.parts[pid]
        p_slip = (
            g.slip_base
            + g.slip_seal * (1.0 - seal) / max(1.0 - g.min_seal, 1e-6)
            + g.slip_offcenter * self._offcenter(p, center) ** 2
        )
        if self.rng.random() < min(p_slip, 1.0):
            return FAIL, "slip", pid
        return SUCCESS, "ok", pid
