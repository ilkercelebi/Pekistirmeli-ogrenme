"""Yükseklik haritasından ayrık (tablo) durum özellikleri.

Tüm yükseklik haritasını tek bir durum olarak almak tablo yöntemlerinde
imkânsızdır (durum sayısı astronomik). Bunun yerine her aday kavrama eylemi
a = (θ, satır, sütun) için, o noktanın çevresinden birkaç ayrık özellik çıkarılır.
Özellik birleşimi φ(s, a) tek bir tamsayıya kodlanır ve Q tablosunun indeksi olur:

    Q(s, a) ≈ Q_tablo[φ(s, a)]

Bu klasik "durum-eylem kümeleme" (state-action aggregation) yöntemidir; sinir ağı
yoktur, öğrenilen şey birkaç yüz hücrelik bir tablodur. Bütün aday eylemler için
tablo değerleri görüntünün üzerine yerleştirilince ödevdeki piksel bazlı
"Q-haritası" elde edilir.

Ajan sadece şunları kullanır: gürültülü yükseklik haritası, kutu duvarlarının yeri
(bilinen sabit geometri) ve tutucu geometrisi. Gerçek parça bilgisini görmez.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import geometry
from .config import EnvConfig, FeatureConfig, GripperConfig


# Her özelliğin kovaları için okunabilir adlar ve "iyi/kötü" tonu (arayüz için).
BIN_LABELS = {
    "yükseklik": [("zemin", "bad"), ("alçak", "mid"), ("orta", "mid"), ("yüksek", "good")],
    "düzlük": [("düz", "good"), ("hafif eğimli", "mid"), ("kenar/pürüzlü", "bad")],
    "yakında_yüksek": [("yok", "good"), ("var", "bad")],
    "duvar": [("çarpar", "bad"), ("yakın", "mid"), ("güvenli", "good")],
    "parmak_boşluğu": [("boş", "good"), ("sınırda", "mid"), ("dolu", "bad")],
    "genişlik": [("sığar", "good"), ("sınırda", "mid"), ("çok geniş", "bad")],
    "ortalı": [("kenarda", "mid"), ("ortada", "good")],
    "sızdırmazlık": [("iyi", "good"), ("orta", "mid"), ("kötü", "bad")],
}

FEATURE_INFO = {
    "yükseklik": "Kavrama noktasının zeminden yüksekliği. Yüksek parçalar genelde yığının üstündedir.",
    "düzlük": "3×3 komşulukta yükseklik değişimi. Değişim büyükse nokta bir parça kenarındadır.",
    "yakında_yüksek": "Yakında belirgin derecede daha yüksek bir nokta var mı? Varsa parçanın üstü kapalı olabilir.",
    "duvar": "Tutucunun ayak izi kutu duvarına değiyor mu ya da çok mu yakın?",
    "parmak_boşluğu": "Parmakların ineceği bölgede engel var mı?",
    "genişlik": "θ yönünde ölçülen yüzey genişliği, tutucu açıklığına sığıyor mu?",
    "ortalı": "Nokta, yüzeyin θ yönündeki ortasına yakın mı?",
    "sızdırmazlık": "Vantuz altındaki alanın ne kadarı aynı düz yüzey?",
}


@dataclass
class FeatureMaps:
    state_ids: np.ndarray   # (A, H, W) her aday eylemin tablo indeksi
    valid: np.ndarray       # (A, H, W) aday eylem maskesi
    raw: dict               # görselleştirme ve sezgisel ajan için ham haritalar


def _shift(a: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """out[r, c] = a[r + dr, c + dc]; ızgara dışı `fill`."""
    out = np.full_like(a, fill)
    rows, cols = a.shape
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(cols, cols - dc)
    if r0 < r1 and c0 < c1:
        out[r0:r1, c0:c1] = a[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
    return out


class FeatureExtractor:
    def __init__(self, env_cfg: EnvConfig, gripper_cfg: GripperConfig, feat_cfg: FeatureConfig, wall_mask: np.ndarray):
        self.ecfg, self.g, self.f = env_cfg, gripper_cfg, feat_cfg
        self.wall = wall_mask.astype(bool)
        self.wall_f = self.wall.astype(float)
        self.interior = ~self.wall
        self.angles = geometry.angle_values(self.g.angle_count)

        g = self.g
        if g.kind == "vacuum":
            self.names = ["yükseklik", "düzlük", "yakında_yüksek", "duvar", "sızdırmazlık"]
            self.dims = [4, 3, 2, 3, 3]
            self.cup_offsets = geometry.cup_offsets(g)
        else:
            self.names = ["yükseklik", "düzlük", "yakında_yüksek", "duvar", "parmak_boşluğu", "genişlik", "ortalı"]
            self.dims = [4, 3, 2, 3, 3, 3, 2]

        reach = (geometry.jaw_body_length(g) / 2 if g.kind != "vacuum" else g.cup_radius) + self.f.wall_margin + 1
        self.body_k, self.near_k, self.finger_k = [], [], []
        for th in self.angles:
            self.body_k.append(geometry.kernel(lambda s, c, th=th: geometry.body_mask(s, c, th, g), reach).astype(float))
            self.near_k.append(
                geometry.kernel(lambda s, c, th=th: geometry.body_mask(s, c, th, g, self.f.wall_margin), reach).astype(float)
            )
            if g.kind != "vacuum":
                self.finger_k.append(geometry.kernel(lambda s, c, th=th: geometry.jaw_finger_mask(s, c, th, g), reach))

        stride = max(1, self.f.candidate_stride)
        grid = np.zeros_like(self.interior)
        t = self.ecfg.wall_thickness
        grid[t::stride, t::stride] = True
        self.candidates = self.interior & grid

    @property
    def n_states(self) -> int:
        return int(np.prod(self.dims))

    def describe(self, state_id: int) -> dict:
        idx = np.unravel_index(int(state_id), self.dims)
        return {n: int(i) for n, i in zip(self.names, idx)}

    # ----------------------------------------------------------------

    def compute(self, obs: np.ndarray) -> FeatureMaps:
        f, g = self.f, self.g
        h = np.where(self.interior, obs, 0.0).astype(float)

        level = np.digitize(h, f.height_bins)
        mean = ndimage.uniform_filter(h, 3, mode="nearest")
        sq = ndimage.uniform_filter(h * h, 3, mode="nearest")
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        flat = np.digitize(std, f.flat_bins)
        nbr = ndimage.maximum_filter(h, size=2 * f.neighbor_radius + 1, mode="nearest")
        higher = (nbr - h > f.neighbor_thr).astype(int)

        A = len(self.angles)
        shape = (A,) + h.shape
        ids = np.zeros(shape, dtype=np.int64)
        raw = {"height": h, "level": level, "std": std, "higher": higher, "wall": [], "extent": [], "clear": [], "seal": None}
        h_obs = np.where(self.wall, self.ecfg.wall_height, h)

        seal_bin = None
        if g.kind == "vacuum":
            same = np.zeros_like(h)
            for dr, dc in self.cup_offsets:
                same += np.abs(_shift(h, int(dr), int(dc), -1e9) - h) < g.surface_tol
            seal = same / len(self.cup_offsets)
            raw["seal"] = seal
            seal_bin = np.where(seal >= f.seal_bins[0], 0, np.where(seal >= f.seal_bins[1], 1, 2))

        for k, th in enumerate(self.angles):
            overlap = ndimage.correlate(self.wall_f, self.body_k[k], mode="constant", cval=1.0) > 0.5
            near = ndimage.correlate(self.wall_f, self.near_k[k], mode="constant", cval=1.0) > 0.5
            wall = np.where(overlap, 0, np.where(near, 1, 2))
            raw["wall"].append(wall)

            if g.kind == "vacuum":
                feats = [level, flat, higher, wall, seal_bin]
            else:
                max_f = ndimage.maximum_filter(h_obs, footprint=self.finger_k[k], mode="constant", cval=self.ecfg.wall_height)
                clear = max_f - (h - g.grasp_depth)
                clear_bin = np.digitize(clear, f.clear_bins)
                raw["clear"].append(clear)

                runs = []
                for sign in (1, -1):
                    run = np.zeros(h.shape, dtype=int)
                    alive = np.ones(h.shape, dtype=bool)
                    for d in range(1, int(math.ceil(g.opening)) + 3):
                        dr = int(round(sign * d * math.sin(th)))
                        dc = int(round(sign * d * math.cos(th)))
                        alive &= np.abs(_shift(h, dr, dc, -1e9) - h) < g.surface_tol
                        run += alive
                    runs.append(run)
                extent = 1 + runs[0] + runs[1]
                raw["extent"].append(extent)
                width = np.where(extent <= g.opening - 2, 0, np.where(extent <= g.opening, 1, 2))
                centered = (np.abs(runs[0] - runs[1]) <= 1).astype(int)
                feats = [level, flat, higher, wall, clear_bin, width, centered]

            ids[k] = np.ravel_multi_index([np.clip(a, 0, d - 1) for a, d in zip(feats, self.dims)], self.dims)

        valid = np.broadcast_to(self.candidates, shape).copy()
        return FeatureMaps(state_ids=ids, valid=valid, raw=raw)
