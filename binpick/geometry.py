"""Izgara üzerinde döndürülmüş şekil maskeleri ve tutucu ayak izleri.

Koordinat kuralı: ızgara indeksi (satır, sütun) = (y, x). Açı θ, x ekseninden
(sütun yönü) ölçülür. Parmaklı tutucuda θ, parmakların kapanma yönüdür.
Ortam (gerçek sonuç) ve özellik çıkarıcı (ajanın gördüğü) aynı fonksiyonları
kullanır; böylece tutucu geometrisi tek bir yerde tanımlanır.
"""

from __future__ import annotations

import math

import numpy as np

from .config import GripperConfig


def angle_values(n: int) -> np.ndarray:
    """[0, π) aralığında n eşit aralıklı bilek açısı."""
    return np.arange(n) * np.pi / n


def _local_coords(shape, center, angle):
    rows, cols = shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    cy, cx = center
    dx, dy = xx - cx, yy - cy
    c, s = math.cos(angle), math.sin(angle)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    return u, v


def rect_mask(shape, center, length, width, angle) -> np.ndarray:
    """Merkezi `center`, uzun kenarı `angle` yönünde olan dikdörtgen."""
    u, v = _local_coords(shape, center, angle)
    return (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)


def ellipse_mask(shape, center, length, width, angle) -> np.ndarray:
    u, v = _local_coords(shape, center, angle)
    return (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1.0


def disk_mask(shape, center, radius) -> np.ndarray:
    return ellipse_mask(shape, center, 2 * radius, 2 * radius, 0.0)


def rect_corners(center, length, width, angle) -> np.ndarray:
    """Çizim için dikdörtgen köşeleri, (x, y) sırasında."""
    cy, cx = center
    c, s = math.cos(angle), math.sin(angle)
    pts = []
    for su, sv in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        u, v = su * length / 2, sv * width / 2
        pts.append((cx + u * c - v * s, cy + u * s + v * c))
    return np.array(pts)


# ---------------------------------------------------------------- tutucu


def finger_centers(center, angle, g: GripperConfig):
    """Açık konumdaki iki parmağın merkezleri."""
    cy, cx = center
    off = g.opening / 2 + g.finger_thickness / 2
    c, s = math.cos(angle), math.sin(angle)
    return (cy - off * s, cx - off * c), (cy + off * s, cx + off * c)


def jaw_finger_mask(shape, center, angle, g: GripperConfig) -> np.ndarray:
    a, b = finger_centers(center, angle, g)
    return rect_mask(shape, a, g.finger_thickness, g.finger_width, angle) | rect_mask(
        shape, b, g.finger_thickness, g.finger_width, angle
    )


def jaw_body_length(g: GripperConfig) -> float:
    return g.opening + 2 * g.finger_thickness


def body_mask(shape, center, angle, g: GripperConfig, margin: float = 0.0) -> np.ndarray:
    """Tutucunun kutu içine inen kısmının tüm ayak izi (duvar çarpışması için)."""
    if g.kind == "vacuum":
        return disk_mask(shape, center, g.cup_radius + margin)
    return rect_mask(shape, center, jaw_body_length(g) + 2 * margin, g.finger_width + 2 * margin, angle)


def kernel(fn, extent: float) -> np.ndarray:
    """`fn(shape, center)` ile üretilen maskeyi, merkezi tam ortada olan tek boyutlu
    bir çekirdeğe (kernel) çevirir. Özellik çıkarıcı bunu filtre olarak kullanır."""
    half = int(math.ceil(extent)) + 1
    size = 2 * half + 1
    k = fn((size, size), (half, half))
    return k.astype(bool)


def cup_offsets(g: GripperConfig):
    k = kernel(lambda shape, c: disk_mask(shape, c, g.cup_radius), g.cup_radius)
    half = k.shape[0] // 2
    rr, cc = np.nonzero(k)
    return list(zip(rr - half, cc - half))
