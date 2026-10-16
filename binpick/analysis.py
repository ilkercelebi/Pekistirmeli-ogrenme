"""Öğrenilen Q tablosunu yorumlamak için yardımcılar (arayüz ve rapor ortak kullanır)."""

from __future__ import annotations

import numpy as np


def marginal_q(agent, fe) -> dict[str, list[float | None]]:
    """Her özelliğin her kovası için ziyaret ağırlıklı ortalama Q.

    Örn. "duvar = çarpar" kovasına düşen tüm tablo hücrelerinin Q değerlerinin,
    ziyaret sayısıyla ağırlıklı ortalaması. Hiç ziyaret edilmemiş kova None olur.
    """
    visited = np.flatnonzero(agent.N > 0)
    idx = np.array(np.unravel_index(np.arange(fe.n_states), fe.dims))
    out = {}
    for i, name in enumerate(fe.names):
        vals = []
        for b in range(fe.dims[i]):
            sel = visited[idx[i, visited] == b]
            w = agent.N[sel]
            vals.append(float((agent.Q[sel] * w).sum() / w.sum()) if w.sum() > 0 else None)
        out[name] = vals
    return out


def extreme_states(agent, fe, k: int = 8, min_visits: int = 10):
    """En yüksek ve en düşük Q değerli (yeterince ziyaret edilmiş) tablo hücreleri."""
    visited = np.flatnonzero(agent.N >= min_visits)
    order = visited[np.argsort(-agent.Q[visited])]
    return order[:k], order[::-1][:k]
