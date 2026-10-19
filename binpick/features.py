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
    "hizalama": [("kısa kenardan", "good"), ("kısmen", "mid"), ("kare / uzun kenardan", "bad")],
    "denendi": [("hayır", "good"), ("evet", "bad")],
    "seri_başarısız": [("0", "good"), ("1", "mid"), ("2", "mid"), ("3+", "bad")],
    "eksen_ortası": [("tam ortada", "good"), ("ortada", "good"), ("kaymış", "mid"), ("uca yakın", "bad"), ("uçta", "bad")],
    "yüzey_kapalı": [("yok", "good"), ("var", "bad")],
}

FEATURE_INFO = {
    "yükseklik": "Kavrama noktasının zeminden yüksekliği. Yüksek parçalar genelde yığının üstündedir.",
    "düzlük": "3×3 komşulukta yükseklik değişimi. Değişim büyükse nokta bir parça kenarındadır.",
    "yakında_yüksek": "Yakında belirgin derecede daha yüksek bir nokta var mı? Varsa parçanın üstü kapalı olabilir.",
    "duvar": "Tutucunun ayak izi kutu duvarına değiyor mu ya da çok mu yakın?",
    "parmak_boşluğu": "Parmakların ineceği bölgede engel var mı?",
    "genişlik": "θ yönünde ölçülen yüzey genişliği, tutucu açıklığına sığıyor mu?",
    "ortalı": "Nokta, yüzeyin θ yönündeki ortasına yakın mı?",
    "seri_başarısız": "Robot bu kutuda üst üste kaç kez başarısız oldu? (kutunun tamamı hakkında: alınabilir parça kalmadığının işareti)",
    "denendi": "Robot bu bölümde bu bölgede daha önce başarısız oldu mu? (kendi hafızası; başarılı alımda silinir)",
    "hizalama": "Parmaklar parçanın kısa kenarından mı sıkıyor (θ yönündeki yüzey, dik yöndekinden kısa mı)?",
    "eksen_ortası": "Nokta, parçanın uzun ekseni boyunca ortada mı, yoksa bir ucuna mı yakın?",
    "sızdırmazlık": "Vantuz altındaki alanın ne kadarı aynı düz yüzey?",
    "yüzey_kapalı": "Noktanın üzerinde durduğu düz yüzey (parçanın görünen üst yüzü) kenarında daha yüksek bir "
    "yüzeye değiyor mu? Değiyorsa başka bir parça bu parçanın üstüne binmiş olabilir.",
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


def _surface_runs(h: np.ndarray, theta: float, max_d: int, tol: float):
    """Her pikselden θ ve -θ yönünde, yükseklik `tol` içinde kaldıkça kaç hücre gidilebildiği."""
    runs = []
    for sign in (1, -1):
        run = np.zeros(h.shape, dtype=int)
        alive = np.ones(h.shape, dtype=bool)
        for d in range(1, max_d + 1):
            dr = int(round(sign * d * math.sin(theta)))
            dc = int(round(sign * d * math.cos(theta)))
            alive &= np.abs(_shift(h, dr, dc, -1e9) - h) < tol
            run += alive
        runs.append(run)
    return runs


def face_touches_higher(h: np.ndarray, interior: np.ndarray, tol: float, thr: float, ring: int) -> np.ndarray:
    """Her parça pikseli için: içinde bulunduğu düz yüzey, kenarında belirgin daha yüksek bir yüzeye değiyor mu?

    `yakında_yüksek` yalnızca noktanın birkaç hücre çevresine bakar; üstü kapalı bir parçanın, üstteki
    parçadan uzak düz bölgeleri ise yerel olarak açık bir parçadan ayırt edilemez. Burada görünen
    yüzeyler bölütlenir (komşu yükseklik farkı `tol` altındaki bağlı bölgeler) ve özellik bölge
    düzeyinde hesaplanır: bölgenin `ring` hücrelik çevresinde bölge yüksekliğinden `thr` kadar yüksek
    bir nokta varsa bölgedeki bütün pikseller işaretlenir. Bir parçanın üstüne binen parça her zaman
    daha yüksektir ve o parçanın görünen yüzüne bitişiktir; bu yüzden bu bilgi ortamın "üstü kapalı"
    kuralının gözlemlenebilir karşılığıdır (öğrenme yok, sabit görüntü işleme).
    """
    on_part = interior & (h > 0.5)
    span = ndimage.maximum_filter(h, 3, mode="nearest") - ndimage.minimum_filter(h, 3, mode="nearest")
    lab, n = ndimage.label(on_part & (span < tol))
    if n == 0:
        return np.zeros(h.shape, dtype=int)
    idx = np.arange(1, n + 1)
    level = np.concatenate([[0.0], ndimage.median(h, lab, idx)])
    around = ndimage.maximum_filter(np.where(interior, h, 0.0), size=2 * ring + 1, mode="nearest")
    flag = np.concatenate([[False], np.asarray(ndimage.maximum(around, lab, idx)) > level[1:] + thr])
    # Yüzey kenarındaki (3×3 içinde basamak olan) pikseller en yakın yüzeye, yükseklikleri uyuyorsa katılır.
    _, (ri, ci) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    nearest = lab[ri, ci]
    lab = np.where((lab == 0) & on_part & (np.abs(h - level[nearest]) < tol), nearest, lab)
    return (flag[lab] & (lab > 0)).astype(int)


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
            if self.f.face_feature:
                self.names.append("yüzey_kapalı")
                self.dims.append(2)
            if self.f.vacuum_center_feature:
                self.names.append("eksen_ortası")
                self.dims.append(3)
            self.cup_offsets = geometry.cup_offsets(g)
        else:
            base = ["yükseklik", "düzlük", "yakında_yüksek", "duvar", "parmak_boşluğu", "genişlik"]
            if self.f.alignment_features:
                self.names = base + ["hizalama", "eksen_ortası"]
                self.dims = [4, 3, 2, 3, 3, 3, len(self.f.align_bins) + 1, len(self.f.axis_center_bins) + 1]
            else:
                self.names, self.dims = base + ["ortalı"], [4, 3, 2, 3, 3, 3, 2]
            if self.f.jaw_face_feature:
                self.names, self.dims = self.names + ["yüzey_kapalı"], self.dims + [2]
        if self.f.memory_feature:
            self.names = self.names + ["denendi"]
            self.dims = self.dims + [2]
        self.use_streak = g.kind in self.f.streak_grippers
        if self.use_streak:
            self.names = self.names + ["seri_başarısız"]
            self.dims = self.dims + [len(self.f.streak_bins) + 1]

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

    def compute(self, obs: np.ndarray, memory: np.ndarray | None = None, streak: int = 0) -> FeatureMaps:
        """`memory`: robotun bu bölümde başarısız olduğu noktalar (BinPickingEnv.fail_map); yoksa boş."""
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
        raw = {"height": h, "level": level, "std": std, "higher": higher, "wall": [], "extent": [], "clear": [], "seal": None,
               "face_higher": None}
        h_obs = np.where(self.wall, self.ecfg.wall_height, h)

        seal_bin = None
        jaw_face = None
        if g.kind != "vacuum" and f.jaw_face_feature:
            raw["face_higher"] = jaw_face = face_touches_higher(h, self.interior, g.surface_tol, f.neighbor_thr, f.face_ring)
        if g.kind == "vacuum":
            same = np.zeros_like(h)
            for dr, dc in self.cup_offsets:
                same += np.abs(_shift(h, int(dr), int(dc), -1e9) - h) < g.surface_tol
            seal = same / len(self.cup_offsets)
            raw["seal"] = seal
            seal_bin = np.where(seal >= f.seal_bins[0], 0, np.where(seal >= f.seal_bins[1], 1, 2))
            raw["face_higher"] = face_higher = face_touches_higher(h, self.interior, g.surface_tol, f.neighbor_thr, f.face_ring)
            vac_center = None
            if f.vacuum_center_feature:
                # Vantuzda kayma, noktanın parçanın uzun ekseninde ortadan uzaklığıyla artar. Uzun eksen,
                # yüzeyin en uzun olduğu yön olarak dört doğrultudan seçilir; dengesizlik o yönde ölçülür.
                best_ext = np.zeros(h.shape)
                vac_off = np.zeros(h.shape)
                for th in geometry.angle_values(4):
                    p, n = _surface_runs(h, th, f.axis_reach, g.surface_tol)
                    ext = 1 + p + n
                    longer = ext > best_ext
                    best_ext = np.where(longer, ext, best_ext)
                    vac_off = np.where(longer, np.abs(p - n) / (p + n + 1.0), vac_off)
                vac_center = np.digitize(vac_off, f.axis_center_bins)

        for k, th in enumerate(self.angles):
            overlap = ndimage.correlate(self.wall_f, self.body_k[k], mode="constant", cval=1.0) > 0.5
            near = ndimage.correlate(self.wall_f, self.near_k[k], mode="constant", cval=1.0) > 0.5
            wall = np.where(overlap, 0, np.where(near, 1, 2))
            raw["wall"].append(wall)

            if g.kind == "vacuum":
                feats = [level, flat, higher, wall, seal_bin] + ([face_higher] if f.face_feature else [])
                if f.vacuum_center_feature:
                    feats.append(vac_center)
            else:
                max_f = ndimage.maximum_filter(h_obs, footprint=self.finger_k[k], mode="constant", cval=self.ecfg.wall_height)
                clear = max_f - (h - g.grasp_depth)
                clear_bin = np.digitize(clear, f.clear_bins)
                raw["clear"].append(clear)

                runs = _surface_runs(h, th, int(math.ceil(g.opening)) + 2, g.surface_tol)
                extent = 1 + runs[0] + runs[1]
                raw["extent"].append(extent)
                width = np.where(extent <= g.opening - 2, 0, np.where(extent <= g.opening, 1, 2))
                if f.alignment_features:
                    # Kayma olasılığı iki şeye bağlı: parmakların parçayı kısa kenarından sıkması ve noktanın
                    # parçanın uzun ekseninde ortada olması. İkisi de kapanma yönüne dik doğrultudaki yüzey
                    # uzunluğundan ölçülür.
                    perp = _surface_runs(h, th + math.pi / 2, f.axis_reach, g.surface_tol)
                    extent_perp = 1 + perp[0] + perp[1]
                    # kapanma yönündeki yüzey / dik yöndeki yüzey: küçükse parça kısa kenarından sıkılıyor
                    align = np.digitize(extent / np.maximum(extent_perp, 1), f.align_bins, right=True)
                    off = np.abs(perp[0] - perp[1]) / (perp[0] + perp[1] + 1.0)
                    axis_center = np.digitize(off, f.axis_center_bins)                # 0 tam ortada … uçta
                    feats = [level, flat, higher, wall, clear_bin, width, align, axis_center]
                else:
                    centered = (np.abs(runs[0] - runs[1]) <= 1).astype(int)
                    feats = [level, flat, higher, wall, clear_bin, width, centered]
                if f.jaw_face_feature:
                    feats.append(jaw_face)

            if f.memory_feature:
                feats = feats + [np.zeros(h.shape, dtype=int) if memory is None else memory.astype(int)]
            if self.use_streak:
                # bütün aday eylemler için ortak: robot bu kutuda üst üste kaç kez başarısız oldu
                feats = feats + [np.full(h.shape, int(np.digitize(streak, f.streak_bins)), dtype=int)]
            ids[k] = np.ravel_multi_index([np.clip(a, 0, d - 1) for a, d in zip(feats, self.dims)], self.dims)

        valid = np.broadcast_to(self.candidates, shape).copy()
        return FeatureMaps(state_ids=ids, valid=valid, raw=raw)
