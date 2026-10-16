"""Yükseklik haritası, Q-haritası ve tutucu çizimi."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from . import geometry  # noqa: E402
from .env import COLLISION, SUCCESS  # noqa: E402

OUTCOME_COLORS = {SUCCESS: "#2ca02c", COLLISION: "#d62728", "fail": "#ff9f1c", None: "#ffffff"}
OUTCOME_TR = {SUCCESS: "BAŞARILI (+1)", COLLISION: "ÇARPIŞMA (-1)", "fail": "BAŞARISIZ (-0.2)"}


def draw_gripper(ax, action, env, color="#ffffff"):
    g = env.g
    center = (action.row, action.col)
    theta = float(env.angles[action.angle_idx])
    if g.kind == "vacuum":
        ax.add_patch(Circle((action.col, action.row), g.cup_radius, fill=False, ec=color, lw=2))
    else:
        for fc in geometry.finger_centers(center, theta, g):
            pts = geometry.rect_corners(fc, g.finger_thickness, g.finger_width, theta)
            ax.add_patch(Polygon(pts, closed=True, fc=color, ec="black", lw=1, alpha=0.9))
        a, b = geometry.finger_centers(center, theta, g)
        ax.plot([a[1], b[1]], [a[0], b[0]], color=color, lw=1, ls="--")
    ax.plot(action.col, action.row, marker="+", color=color, ms=10, mew=2)


def render(env, *, agent=None, feats=None, action=None, info=None, title=None, show_q=True, dpi=90) -> np.ndarray:
    """Tek bir kare çizer ve RGB dizi olarak döndürür."""
    q = agent.q_map(feats) if (show_q and agent is not None and feats is not None) else None
    ncols = 2 if q is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.2), dpi=dpi)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    im = ax.imshow(env.heightmap, cmap="viridis", vmin=0, vmax=env.cfg.max_stack_height)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="yükseklik (cm)")
    ax.set_title(title or "Kutu (yükseklik haritası)")

    if q is not None:
        qmax, _ = q
        ax2 = axes[1]
        im2 = ax2.imshow(qmax, cmap="RdYlGn", vmin=-1, vmax=1)
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03, label="max_θ Q")
        ax2.set_title("Q-haritası (tablo değerleri)")

    if action is not None:
        color = OUTCOME_COLORS.get(info["outcome"] if info else None, "#ffffff")
        for a in axes:
            draw_gripper(a, action, env, color)
        if info:
            txt = f"{OUTCOME_TR[info['outcome']]} · θ={info['theta_deg']:.0f}° · {info['reason']}"
            fig.suptitle(txt, color=color, fontsize=11, fontweight="bold")

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_gif(frames, path, fps: float = 2.0) -> None:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def moving_average(x, window: int):
    x = np.asarray(x, dtype=float)
    if len(x) < window or window <= 1:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    ma = (c[window:] - c[:-window]) / window
    return np.concatenate([np.full(window - 1, np.nan), ma])


def angle_label(env, k: int) -> str:
    return f"{math.degrees(env.angles[k]):.0f}°"
