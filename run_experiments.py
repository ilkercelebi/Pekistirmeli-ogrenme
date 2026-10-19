"""Tüm deneyi tek komutla çalıştırır: eğitim → değerlendirme → grafikler → GIF'ler → README.

    python run_experiments.py                 # tam deney (2 algoritma × 2 tutucu × 3 tohum × 3000 bölüm)
    python run_experiments.py --ablation      # + ablasyon (Q-learning): parmaklıda γ/α/açı, vakumda yüzey_kapalı
    python run_experiments.py --quick         # hızlı deneme: 1 tohum, 300 bölüm, results/quick/ altına
    python run_experiments.py --skip-train    # eğitim yapmadan mevcut sonuçlardan raporu yeniden üret

Tamamlanmış eğitimler atlanır (yarıda kalırsa komut tekrar çalıştırılınca kaldığı yerden devam eder);
baştan eğitmek için --force. Eğitimler çekirdek sayısına göre paralel koşar.

Çıktılar:
    <out>/<algo>_<tutucu>_s<tohum>/   her eğitim (config.json, metrics.csv, eval.csv, agent.npz)
    <out>/ablation/...                 ablasyon eğitimleri
    <out>/report/                      tablolar (CSV, Markdown), grafikler, GIF'ler
    docs/img/                          README'nin kullandığı görseller (--quick hariç)
    README.md                          <!-- SONUCLAR --> ve <!-- OZET --> blokları (--quick hariç)
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from binpick import ExperimentConfig, build, load_run, make_agent, run_episode  # noqa: E402
from binpick.analysis import marginal_q  # noqa: E402
from binpick.oracle import run_oracle_episode  # noqa: E402
from binpick.features import BIN_LABELS  # noqa: E402
from binpick.training import train  # noqa: E402
from binpick.viz import moving_average, render, save_gif  # noqa: E402

ROOT = Path(__file__).resolve().parent
TEST_SEED0 = 2_000_000           # eğitimde hiç görülmeyen test kutuları (evaluate.py ile aynı)
GIF_SEED = 7
METRICS = ["success_rate", "collision_rate", "fail_rate", "cleared", "return", "steps"]
LABEL = {"random": "Rastgele", "heuristic": "Sezgisel", "heuristic_face": "Sezgisel + yüzey bilgisi",
         "q_learning": "Q-learning", "sarsa": "SARSA", "double_q": "Double Q-learning", "oracle": "Kâhin (üst sınır)"}
GRIPPER = {"parallel_jaw": "Parmaklı tutucu", "vacuum": "Vakumlu tutucu"}
COLOR = {"random": "#8b949e", "heuristic": "#58a6ff", "heuristic_face": "#3fb950", "q_learning": "#ff7a1a",
         "sarsa": "#bc8cff", "double_q": "#39c5cf", "oracle": "#d4a72c"}
LEARNERS = ["q_learning", "sarsa", "double_q"]
METHODS = ["random", "heuristic", "heuristic_face", *LEARNERS, "oracle"]
# Kâhin: gizli bilgiyi (parça kimliği, gerçek kayma olasılığı) bilen politika; ulaşılabilecek üst sınır.
BASELINES = {"parallel_jaw": ["random", "heuristic", "oracle"],
             # vakumda sezgisele Q-learning'in gördüğü yüzey_kapalı bilgisi de verilir (adil karşılaştırma)
             "vacuum": ["random", "heuristic", "heuristic_face", "oracle"]}

# Ablasyon: Q-learning. Her satır (isim, tutucu, açıklama, konfigürasyon değişikliği, tüm tohumlar mı).
# Tek tohumlu satırlar yalnız tohum 0 ile eğitilir (ablation/<isim>); tüm tohumlu satırlar ana deneyle
# aynı tohumlarla (ablation/<isim>_s<tohum>), böylece ana deneydeki karşılığıyla ± değerleri karşılaştırılabilir.
ABLATIONS = [
    ("gamma0.0", "parallel_jaw", "γ = 0 (tek adımlık, bandit)", {"agent.gamma": 0.0}, False),
    ("gamma0.5", "parallel_jaw", "γ = 0.5 (varsayılan)", {"agent.gamma": 0.5}, False),
    ("gamma0.9", "parallel_jaw", "γ = 0.9", {"agent.gamma": 0.9}, False),
    ("alpha_const", "parallel_jaw", "sabit α = 0.1", {"agent.alpha_mode": "constant", "agent.alpha": 0.1}, False),
    ("angles1", "parallel_jaw", "tek bilek açısı (θ öğrenilmiyor)", {"gripper.n_angles": 1}, False),
    ("noalign", "parallel_jaw", "hizalama + eksen ortası yok (eski `ortalı`)", {"features.alignment_features": False}, False),
    ("noface", "parallel_jaw", "yüzeyi kapatan parça yok", {"features.jaw_face_feature": False}, False),
    ("nomem", "parallel_jaw", "hafıza (önceden denendi) yok", {"features.memory_feature": False}, False),
    ("vacuum_noface", "vacuum", "yüzey_kapalı yok", {"features.face_feature": False}, True),
    ("vacuum_nomem", "vacuum", "hafıza (önceden denendi) yok", {"features.memory_feature": False}, False),
    ("coarse_bins", "parallel_jaw", "kaba kovalar (hizalama 2, eksen ortası 3 kova)",
     {"features.align_bins": (1.0,), "features.axis_center_bins": (0.25, 0.55)}, False),
    ("nostop", "parallel_jaw", "dur eylemi yok", {"env.allow_stop": False}, False),
    ("vacuum_nostop", "vacuum", "dur eylemi yok", {"env.allow_stop": False}, False),
    ("vacuum_nostreak", "vacuum", "seri başarısızlık özelliği yok", {"features.streak_grippers": ()}, True),
    ("dq_gamma0.9", "parallel_jaw", "Double Q-learning, γ = 0.9", {"agent.algo": "double_q", "agent.gamma": 0.9}, False),
]


# ------------------------------------------------------------------ eğitim


def make_cfg(algo: str, gripper: str, episodes: int, overrides: dict | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.gripper.kind = gripper
    cfg.agent.algo = algo
    cfg.agent.eps_decay_episodes = max(1, episodes // 2)   # ε eğitimin ilk yarısında azalır
    for path, value in (overrides or {}).items():
        section, attr = path.split(".")
        setattr(getattr(cfg, section), attr, value)
    return cfg


def is_complete(run_dir: Path, episodes: int) -> bool:
    if not (run_dir / "agent.npz").exists() or not (run_dir / "metrics.csv").exists():
        return False
    with open(run_dir / "metrics.csv") as f:
        return sum(1 for _ in f) - 1 >= episodes


def train_job(job: dict) -> tuple[str, str]:
    """Alt süreçte çalışır: tek bir eğitim."""
    run_dir = Path(job["dir"])
    if not job["force"] and is_complete(run_dir, job["episodes"]):
        return job["name"], "zaten tamam"
    cfg = make_cfg(job["algo"], job["gripper"], job["episodes"], job.get("overrides"))
    t0 = time.time()
    train(cfg, job["episodes"], seed=job["seed"], run_dir=run_dir,
          eval_every=max(25, job["episodes"] // 12), eval_episodes=job["eval_episodes"])
    return job["name"], f"{time.time() - t0:.0f} sn"


def run_pool(fn, jobs: list[dict], workers: int, what: str):
    if not jobs:
        return []
    print(f"\n== {what}: {len(jobs)} iş, {workers} paralel ==")
    out, t0 = [], time.time()
    if workers <= 1:
        for j in jobs:
            r = fn(j)
            out.append(r)
            print(f"  ✓ {r[0]:40s} {_brief(r[1])}")
        return out
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            out.append(r)
            print(f"  ✓ [{i}/{len(jobs)}] {r[0]:40s} {_brief(r[1])}   (toplam {time.time() - t0:.0f} sn)")
    return out


def _brief(result) -> str:
    if isinstance(result, dict):
        return (f"başarı %{result['success_rate'] * 100:.1f} · çarpışma %{result['collision_rate'] * 100:.1f} · "
                f"boşaltma %{result['cleared'] * 100:.1f}")
    return str(result)


# ------------------------------------------------------------------ değerlendirme


def eval_job(job: dict) -> tuple[str, dict]:
    """Alt süreçte çalışır: bir yöntemi test kutularında değerlendirir (keşif kapalı)."""
    if job["method"] == "oracle":
        env, _ = build(make_cfg("q_learning", job["gripper"], 1), seed=0)
        rows = [run_oracle_episode(env, seed=TEST_SEED0 + i) for i in range(job["n"])]
    else:
        if job["method"] in ("random", "heuristic", "heuristic_face"):
            cfg = make_cfg("q_learning", job["gripper"], 1)
            agent = make_agent(job["method"], 0, seed=0)
        else:
            cfg, agent = load_run(job["dir"], seed=0)
        env, fe = build(cfg, seed=0)
        rows = [run_episode(env, agent, fe, learn=False, seed=TEST_SEED0 + i) for i in range(job["n"])]
    res = {k: float(np.mean([r[k] for r in rows])) for k in METRICS}
    res.update({k + "_box_std": float(np.std([r[k] for r in rows])) for k in METRICS})
    return job["key"], res


def aggregate(per_run: dict[str, dict], groups: dict[str, list[str]]) -> dict[str, dict]:
    """Aynı yöntemin farklı tohumlarını birleştirir: ortalama ve tohumlar arası std."""
    agg = {}
    for method, keys in groups.items():
        vals = [per_run[k] for k in keys if k in per_run]
        if not vals:
            continue
        a = {k: float(np.mean([v[k] for v in vals])) for k in METRICS}
        a.update({k + "_seed_std": float(np.std([v[k] for v in vals])) if len(vals) > 1 else None for k in METRICS})
        a.update({k + "_box_std": float(np.mean([v[k + "_box_std"] for v in vals])) for k in METRICS})
        a["n_seeds"] = len(vals)
        agg[method] = a
    return agg


# ------------------------------------------------------------------ grafikler

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in (rows[0] if rows else {})}


def plot_learning(run_dirs: dict[str, list[Path]], baselines: dict, gripper: str, path: Path, window: int):
    panels = [("success_rate", "Başarı oranı"), ("collision_rate", "Çarpışma oranı"),
              ("cleared", "Boşaltılan parça oranı"), ("return", "Bölüm getirisi")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for algo, dirs in run_dirs.items():
        curves = [read_csv(d / "metrics.csv") for d in dirs]
        n = min(len(c["episode"]) for c in curves)
        for ax, (k, _) in zip(axes.flat, panels):
            ys = np.array([moving_average(c[k][:n], window) for c in curves])
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # ilk pencere NaN
                m, s = np.nanmean(ys, axis=0), np.nanstd(ys, axis=0)
            x = np.arange(1, n + 1)
            ax.plot(x, m, color=COLOR[algo], lw=2, label=f"{LABEL[algo]} ({len(dirs)} tohum)")
            if len(dirs) > 1:
                ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.18, lw=0)
    for ax, (k, title) in zip(axes.flat, panels):
        for b, style in (("random", ":"), ("heuristic", "--"), ("heuristic_face", "-."), ("oracle", (0, (6, 2, 1, 2)))):
            if b in baselines:
                ax.axhline(baselines[b][k], color=COLOR[b], ls=style, lw=1.5, label=f"{LABEL[b]} (test)")
        ax.set_title(title)
        ax.set_xlabel("eğitim bölümü")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"{GRIPPER[gripper]}: eğitim eğrileri (keşifli politika, {window} bölüm hareketli ortalama; gölge = tohumlar arası ±1 std)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_greedy(run_dirs: dict[str, list[Path]], baselines: dict, gripper: str, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for algo, dirs in run_dirs.items():
        evs = [read_csv(d / "eval.csv") for d in dirs]
        n = min(len(e["episode"]) for e in evs)
        x = evs[0]["episode"][:n]
        for ax, k in zip(axes, ("success_rate", "collision_rate", "cleared")):
            ys = np.array([e[k][:n] for e in evs])
            ax.errorbar(x, ys.mean(0), yerr=ys.std(0) if len(dirs) > 1 else None, color=COLOR[algo],
                        marker="o", ms=3.5, lw=1.8, capsize=2, label=LABEL[algo])
    for ax, (k, t) in zip(axes, (("success_rate", "Başarı"), ("collision_rate", "Çarpışma"), ("cleared", "Boşaltma"))):
        for b, style in (("heuristic", "--"), ("heuristic_face", "-."), ("oracle", (0, (6, 2, 1, 2)))):
            if b in baselines:
                ax.axhline(baselines[b][k], color=COLOR[b], ls=style, lw=1.5, label=LABEL[b])
        ax.set_title(f"Açgözlü politika (ε = 0): {t}")
        ax.set_xlabel("eğitim bölümü")
        ax.set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    fig.suptitle(GRIPPER[gripper], fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(agg: dict, gripper: str, path: Path, n_boxes: int):
    keys = [("success_rate", "Başarı oranı"), ("collision_rate", "Çarpışma oranı"),
            ("fail_rate", "Başarısız kavrama"), ("cleared", "Boşaltılan parça")]
    methods = [m for m in METHODS if m in agg]
    x = np.arange(len(keys))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for i, m in enumerate(methods):
        vals = [agg[m][k] for k, _ in keys]
        err = [agg[m][k + "_seed_std"] or 0 for k, _ in keys]
        bars = ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=err if any(err) else None, capsize=3,
                      color=COLOR[m], label=LABEL[m])
        ax.bar_label(bars, labels=[f"{v * 100:.0f}" for v in vals], padding=2, fontsize=8)
    ax.set_xticks(x, [t for _, t in keys])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("oran")
    ax.legend(ncol=len(methods), fontsize=9, loc="upper center")
    ax.set_title(f"{GRIPPER[gripper]}: {n_boxes} test kutusu (hata çubuğu = tohumlar arası std)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_qtable(run_dir: Path, gripper: str, path: Path):
    cfg, agent = load_run(run_dir)
    _, fe = build(cfg)
    marg = marginal_q(agent, fe)
    n = len(marg)
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 3.1), sharey=True)
    for ax, (name, vals) in zip(np.atleast_1d(axes), marg.items()):
        labels = [lab for lab, _ in BIN_LABELS[name]]
        v = [0.0 if x is None else x for x in vals]
        ax.bar(labels, v, color=["#3fb950" if x >= 0 else "#f85149" for x in v])
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("ortalama Q")
    fig.suptitle(f"{GRIPPER[gripper]}: ajan ne öğrendi? (özellik kovası başına ziyaret ağırlıklı ortalama Q, {run_dir.name})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation(rows: list[tuple[str, dict]], path: Path, title: str):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = [r[0] for r in rows]
    succ = [r[1]["success_rate"] for r in rows]
    clr = [r[1]["cleared"] for r in rows]
    err = [[r[1].get(k + "_seed_std") or 0 for r in rows] for k in ("success_rate", "cleared")]
    x = np.arange(len(rows))
    b1 = ax.bar(x - 0.2, succ, 0.4, yerr=err[0] if any(err[0]) else None, capsize=3, color="#ff7a1a",
                label="Başarı oranı")
    b2 = ax.bar(x + 0.2, clr, 0.4, yerr=err[1] if any(err[1]) else None, capsize=3, color="#39c5cf",
                label="Boşaltılan parça")
    ax.bar_label(b1, labels=[f"{v * 100:.0f}" for v in succ], fontsize=8, padding=2)
    ax.bar_label(b2, labels=[f"{v * 100:.0f}" for v in clr], fontsize=8, padding=2)
    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_gif_for(method: str, gripper: str, run_dir: Path | None, path: Path, label: str):
    if run_dir is None:
        cfg = make_cfg("q_learning", gripper, 1)
        agent = make_agent(method, 0, seed=GIF_SEED)
    else:
        cfg, agent = load_run(run_dir, seed=GIF_SEED)
    env, fe = build(cfg, seed=GIF_SEED)
    frames = [render(env, agent=agent, feats=fe.compute(env.reset(seed=GIF_SEED)), title=f"{label} · başlangıç", dpi=70)]

    def on_step(env, obs, feats, action, reward, info):
        frames.append(render(env, agent=agent, feats=feats, action=action, info=info, dpi=70,
                             title=f"{label} · adım {info['step']} · kalan {info['parts_left']}"))

    stats = run_episode(env, agent, fe, learn=False, seed=GIF_SEED, on_step=on_step)
    save_gif(frames, path, fps=2.0)
    return stats


# ------------------------------------------------------------------ rapor


def fmt_cell(a: dict, k: str, pct: bool) -> str:
    v = a[k]
    s = a.get(k + "_seed_std")
    if pct:
        txt = f"%{v * 100:.1f}"
        return txt + (f" ± {s * 100:.1f}" if s is not None else "")
    return f"{v:.2f}" + (f" ± {s:.2f}" if s is not None else "")


def results_table(agg: dict) -> str:
    head = "| Yöntem | Başarı oranı | Çarpışma oranı | Başarısız kavrama | Boşaltılan parça | Getiri | Adım |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    best = max((m for m in agg if m != "oracle"), key=lambda m: agg[m]["success_rate"])
    for m in METHODS:
        if m not in agg:
            continue
        a = agg[m]
        name = f"**{LABEL[m]}**" if m == best else LABEL[m]
        tag = f" ({a['n_seeds']} tohum)" if m in LEARNERS else ""
        cells = [fmt_cell(a, k, True) for k in ("success_rate", "collision_rate", "fail_rate", "cleared")]
        cells += [fmt_cell(a, "return", False), fmt_cell(a, "steps", False)]
        lines.append(f"| {name}{tag} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(agg_by_gripper: dict) -> str:
    """Özet için tek cümlelik ana bulgu (parmaklı tutucu üzerinden)."""
    g = "parallel_jaw" if "parallel_jaw" in agg_by_gripper else next(iter(agg_by_gripper))
    agg = agg_by_gripper[g]
    learners = [m for m in LEARNERS if m in agg]
    if not learners or "heuristic" not in agg:
        return "Sonuç için eğitim gerekli."
    best = max(learners, key=lambda m: agg[m]["success_rate"])
    b, h, r = agg[best], agg["heuristic"], agg.get("random")
    diff = (b["success_rate"] - h["success_rate"]) * 100
    verb = "geçti" if diff > 0 else "geride kaldı"
    txt = (f"{GRIPPER[g]} ile {LABEL[best]}, eğitimde görülmemiş test kutularında **%{b['success_rate'] * 100:.1f}** "
           f"başarı oranına ulaştı ve en yüksek nokta sezgiselini (%{h['success_rate'] * 100:.1f}) "
           f"{abs(diff):.1f} puan farkla {verb}; çarpışma oranı %{b['collision_rate'] * 100:.1f}, "
           f"kutunun %{b['cleared'] * 100:.1f}'i boşaltıldı.")
    if r:
        txt += f" Rastgele seçimde başarı %{r['success_rate'] * 100:.1f}, çarpışma %{r['collision_rate'] * 100:.1f}."
    o = agg.get("oracle")
    if o:
        txt += (f" Gizli bilgiyi bilen kâhin politika %{o['success_rate'] * 100:.1f} başarıya ulaşıyor; öğrenen ajan bu "
                f"üst sınırın **%{100 * b['success_rate'] / max(o['success_rate'], 1e-9):.0f}**'ine erişti.")
    return txt


def replace_block(text: str, tag: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {tag}:BASLA[^>]*-->)(.*?)(<!-- {tag}:BITIR -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README'de <!-- {tag}:BASLA --> / <!-- {tag}:BITIR --> işaretleri bulunamadı.")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def n_states(gripper: str, overrides: dict | None = None) -> int:
    _, fe = build(make_cfg("q_learning", gripper, 1, overrides))
    return fe.n_states


def vacuum_comparison(agg: dict, abl_rows: list) -> list[tuple[str, str, str, str, dict]]:
    """Vakum: kazanç öğrenmeden mi, yalnızca ek bilgiden mi? (yöntem, kısa ad, yüzey bilgisi, tablo, sonuç).

    Sezgisel ve sezgisel + yüzey bilgisi aynı bilgiye sahip elle yazılmış kurallardır; Q-learning'in
    yüzey_kapalı olmadan ve onunla eğitilmiş tabloları bunların öğrenen karşılıklarıdır.
    """
    rows = [(LABEL[m], LABEL[m].replace(" + ", "\n+ "), info, "—", agg[m])
            for m, info in (("heuristic", "yok"), ("heuristic_face", "var")) if m in agg]
    learned = [("yok", {"features.face_feature": False}, a) for n, g, _, a in abl_rows if n == "vacuum_noface"]
    if "q_learning" in agg:
        learned.append(("var", {}, agg["q_learning"]))
    for info, ov, a in learned:
        cells = n_states("vacuum", ov)
        rows.append((f"Q-learning ({a['n_seeds']} tohum)", f"Q-learning\n{cells} hücre", info, f"{cells} hücre", a))
    return rows


def vacuum_verdict(agg: dict, abl_rows: list) -> str:
    """Bilgi etkisi ve öğrenme etkisi (başarı oranı, puan)."""
    noface = next((a for n, g, _, a in abl_rows if n == "vacuum_noface"), None)
    if noface is None or not all(m in agg for m in ("heuristic", "heuristic_face", "q_learning")):
        return ""
    h, hf = agg["heuristic"]["success_rate"], agg["heuristic_face"]["success_rate"]
    q0, q1 = noface["success_rate"], agg["q_learning"]["success_rate"]
    pt = lambda v: f"{v * 100:+.1f}"  # noqa: E731
    c0, c1 = n_states("vacuum", {"features.face_feature": False}), n_states("vacuum", {})
    return (f"Yüzey bilgisinin etkisi (başarı oranı, puan): sezgiselde **{pt(hf - h)}**, Q-learning'de **{pt(q1 - q0)}**. "
            f"Öğrenmenin aynı bilgiye sahip kurala göre etkisi: yüzey bilgisi olmadan **{pt(q0 - h)}** "
            f"(Q-learning {c0} hücre − sezgisel), yüzey bilgisiyle **{pt(q1 - hf)}** "
            f"(Q-learning {c1} hücre − sezgisel + yüzey bilgisi).")


def ablation_table(rows: list[tuple[str, dict]]) -> list[str]:
    out = ["| Değişiklik | Başarı oranı | Çarpışma oranı | Boşaltılan parça | Getiri |", "|---|---|---|---|---|"]
    for desc, a in rows:
        cells = [fmt_cell(a, k, True) for k in ("success_rate", "collision_rate", "cleared")] + [fmt_cell(a, "return", False)]
        out.append(f"| {desc} | " + " | ".join(cells) + " |")
    return out


def results_markdown(agg_by_gripper, ablation_rows, args, img: str, gifs: dict, qtables: dict) -> str:
    today = dt.date.today().isoformat()
    out = [f"_Bu bölüm `run_experiments.py` tarafından üretildi ({today}). Ayar: {len(args.seeds)} tohum × "
           f"{args.episodes} eğitim bölümü; test: eğitimde görülmemiş {args.test_boxes} kutu, keşif kapalı (ε = 0). "
           "± değerleri tohumlar arası standart sapmadır._", ""]
    sec = 1
    for g, agg in agg_by_gripper.items():
        out += [f"### 6.{sec} {GRIPPER[g]}", "", results_table(agg), "",
                f"![Eğitim eğrileri]({img}/learning_curves_{g}.png)", "",
                f"![Açgözlü politika]({img}/greedy_{g}.png)", "",
                f"![Karşılaştırma]({img}/bars_{g}.png)", ""]
        if g in qtables:
            out += [f"![Q tablosu]({img}/{qtables[g]})", ""]
        sec += 1
    if gifs:
        out += [f"### 6.{sec} Örnek bölümler (aynı kutu, tohum {GIF_SEED})", ""]
        for g, items in gifs.items():
            out += [f"**{GRIPPER[g]}**", "", "| " + " | ".join(lbl for lbl, _ in items) + " |",
                    "|" + "---|" * len(items), "| " + " | ".join(f"![]({img}/{f})" for _, f in items) + " |", ""]
        sec += 1
    if ablation_rows:
        out += [f"### 6.{sec} Ablasyon", ""]
        pj = [(desc, a) for _, g, desc, a in ablation_rows if g == "parallel_jaw"]
        if pj:
            out += ["**Parmaklı tutucu** (Q-learning, tohum 0)", ""] + ablation_table(pj)
            out += ["", f"![Ablasyon]({img}/ablation.png)", ""]
        vac = vacuum_comparison(agg_by_gripper.get("vacuum", {}), ablation_rows)
        if any(g == "vacuum" for _, g, _, _ in ablation_rows):
            out += ["**Vakumlu tutucu: kazanç öğrenmeden mi, yalnızca ek bilgiden mi?** Sezgisele Q-learning'in "
                    "gördüğü `yüzey_kapalı` bilgisi verilir (aday noktalar önce üzerinde durdukları düz yüzey daha "
                    "yüksek bir yüzeye değmeyenlerle sınırlanır); Q-learning de bu özellik olmadan eğitilir.", "",
                    "| Yöntem | Yüzey bilgisi | Q tablosu | Başarı oranı | Çarpışma oranı | Boşaltılan parça | Getiri |",
                    "|---|---|---|---|---|---|---|"]
            for label, _, info, table, a in vac:
                cells = [fmt_cell(a, k, True) for k in ("success_rate", "collision_rate", "cleared")]
                out.append(f"| {label} | {info} | {table} | " + " | ".join(cells + [fmt_cell(a, "return", False)]) + " |")
            verdict = vacuum_verdict(agg_by_gripper.get("vacuum", {}), ablation_rows)
            out += [""] + ([verdict, ""] if verdict else []) + [f"![Vakum ablasyonu]({img}/ablation_vacuum.png)", ""]
    return "\n".join(out).rstrip()


# ------------------------------------------------------------------ ana akış


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="hızlı deneme (1 tohum, 300 bölüm, results/quick)")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--algos", nargs="+", default=LEARNERS, choices=LEARNERS)
    p.add_argument("--grippers", nargs="+", default=["parallel_jaw", "vacuum"], choices=["parallel_jaw", "vacuum"])
    p.add_argument("--test-boxes", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=20, help="eğitim sırasındaki ara değerlendirme kutusu")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default=None)
    p.add_argument("--force", action="store_true", help="tamamlanmış eğitimleri de baştan çalıştır")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--no-gifs", action="store_true")
    p.add_argument("--readme", default=None, help="güncellenecek README (varsayılan: tam deneyde README.md)")
    p.add_argument("--img-dir", default=None, help="görsellerin kopyalanacağı klasör (varsayılan: docs/img)")
    a = p.parse_args(argv)
    a.episodes = a.episodes or (300 if a.quick else 3000)
    a.seeds = a.seeds if a.seeds is not None else ([0] if a.quick else [0, 1, 2])
    a.test_boxes = a.test_boxes or (30 if a.quick else 200)
    a.out = Path(a.out or (ROOT / "results" / "quick" if a.quick else ROOT / "results"))
    if a.readme is None and not a.quick:
        a.readme = str(ROOT / "README.md")
    if a.img_dir is None and not a.quick:
        a.img_dir = str(ROOT / "docs" / "img")
    return a


def main(argv=None):
    a = parse_args(argv)
    out, report = a.out, a.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"Deney: {a.algos} × {a.grippers} × tohum {a.seeds} × {a.episodes} bölüm → {out}")

    # 1) eğitim
    jobs = [{"name": f"{algo}_{g}_s{s}", "dir": str(out / f"{algo}_{g}_s{s}"), "algo": algo, "gripper": g,
             "seed": s, "episodes": a.episodes, "eval_episodes": a.eval_episodes, "force": a.force}
            for g in a.grippers for algo in a.algos for s in a.seeds]
    ablations = [r for r in ABLATIONS if r[1] in a.grippers] if a.ablation else []
    abl_jobs = []
    for name, g, _, ov, all_seeds in ablations:
        for s in (a.seeds if all_seeds else [0]):
            d = f"{name}_s{s}" if all_seeds else name
            abl_jobs.append({"name": f"ablation/{d}", "row": name, "dir": str(out / "ablation" / d), "algo": "q_learning",
                             "gripper": g, "seed": s, "episodes": a.episodes, "eval_episodes": a.eval_episodes,
                             "force": a.force, "overrides": ov})
    if not a.skip_train:
        run_pool(train_job, jobs + abl_jobs, a.workers, "Eğitim")

    # 2) değerlendirme
    ejobs = []
    for g in a.grippers:
        for m in BASELINES[g]:
            ejobs.append({"key": f"{m}|{g}", "method": m, "gripper": g, "n": a.test_boxes})
        for j in jobs:
            if j["gripper"] == g and is_complete(Path(j["dir"]), 1):
                ejobs.append({"key": f"{j['name']}|{g}", "method": j["algo"], "dir": j["dir"], "gripper": g, "n": a.test_boxes})
    for j in abl_jobs:
        if is_complete(Path(j["dir"]), 1):
            ejobs.append({"key": f"{j['name']}|abl", "method": "q_learning", "dir": j["dir"], "gripper": j["gripper"], "n": a.test_boxes})
    per_run = dict(run_pool(eval_job, ejobs, a.workers, "Değerlendirme"))

    agg_by_gripper, run_dirs_by_gripper = {}, {}
    with open(report / "per_run.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key"] + METRICS)
        for k, v in sorted(per_run.items()):
            w.writerow([k] + [round(v[m], 4) for m in METRICS])
    for g in a.grippers:
        groups = {m: [f"{m}|{g}"] for m in BASELINES[g]}
        dirs = {}
        for algo in a.algos:
            names = [j["name"] for j in jobs if j["gripper"] == g and j["algo"] == algo and f"{j['name']}|{g}" in per_run]
            if names:
                groups[algo] = [f"{n}|{g}" for n in names]
                dirs[algo] = [out / n for n in names]
        agg = aggregate(per_run, groups)
        agg_by_gripper[g] = agg
        run_dirs_by_gripper[g] = dirs
        (report / f"summary_{g}.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
        (report / f"summary_{g}.md").write_text(results_table(agg) + "\n")
        print(f"\n{GRIPPER[g]}\n{results_table(agg)}")

    # 3) grafikler
    print("\n== Grafikler ==")
    window = max(10, min(100, a.episodes // 20))
    qtables = {}
    for g in a.grippers:
        dirs = run_dirs_by_gripper[g]
        if dirs:
            plot_learning(dirs, agg_by_gripper[g], g, report / f"learning_curves_{g}.png", window)
            plot_greedy(dirs, agg_by_gripper[g], g, report / f"greedy_{g}.png")
            if "q_learning" in dirs:
                best = max(dirs["q_learning"], key=lambda d: per_run[f"{d.name}|{g}"]["success_rate"])
                qtables[g] = f"qtable_{g}.png"
                plot_qtable(best, g, report / qtables[g])
        plot_bars(agg_by_gripper[g], g, report / f"bars_{g}.png", a.test_boxes)
    abl_rows = []   # (isim, tutucu, açıklama, tohumlar üzerinden birleştirilmiş sonuç)
    for name, g, desc, _, _ in ablations:
        keys = [f"{j['name']}|abl" for j in abl_jobs if j["row"] == name]
        row = aggregate(per_run, {name: keys}).get(name)
        if row:
            abl_rows.append((name, g, desc, row))
    if abl_rows:
        pj = [(n, r) for n, g, _, r in abl_rows if g == "parallel_jaw"]
        if pj:
            plot_ablation(pj, report / "ablation.png", "Ablasyon: Q-learning, parmaklı tutucu (test kutuları)")
        vac = vacuum_comparison(agg_by_gripper.get("vacuum", {}), abl_rows)
        if any(g == "vacuum" for _, g, _, _ in abl_rows):
            plot_ablation([(short, r) for _, short, _, _, r in vac],
                          report / "ablation_vacuum.png",
                          "Vakumlu tutucu: sezgisel ve Q-learning, yüzey_kapalı bilgisiyle ve bilgisi olmadan")
        with open(report / "ablation.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "gripper", "description", "n_seeds"] + METRICS + [m + "_seed_std" for m in METRICS])
            for n, g, d, r in abl_rows:
                w.writerow([n, g, d, r["n_seeds"]] + [round(r[m], 4) for m in METRICS]
                           + ["" if r[m + "_seed_std"] is None else round(r[m + "_seed_std"], 4) for m in METRICS])

    # 4) GIF'ler: aynı kutuda rastgele, sezgisel ve en iyi öğrenen ajan
    gifs = {}
    if not a.no_gifs:
        print("== GIF'ler ==")
        for g in a.grippers:
            items = []
            for m in ("random", "heuristic"):
                fname = f"{m}_{g}.gif"
                s = make_gif_for(m, g, None, report / fname, LABEL[m])
                items.append((LABEL[m], fname))
                print(f"  {fname}: {s['success']}/{s['parts']} parça, {s['steps']} adım")
            dirs = run_dirs_by_gripper[g]
            learners = [(algo, d) for algo, ds in dirs.items() for d in ds]
            if learners:
                algo, best = max(learners, key=lambda x: per_run[f"{x[1].name}|{g}"]["success_rate"])
                fname = f"{algo}_{g}.gif"
                s = make_gif_for(algo, g, best, report / fname, LABEL[algo])
                items.append((LABEL[algo], fname))
                print(f"  {fname}: {s['success']}/{s['parts']} parça, {s['steps']} adım")
            gifs[g] = items

    # 5) README ve docs/img
    if a.img_dir:
        img_dir = Path(a.img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        for f in report.iterdir():
            if f.suffix in (".png", ".gif"):
                shutil.copy2(f, img_dir / f.name)
        print(f"Görseller kopyalandı → {img_dir}")
    if a.readme:
        readme = Path(a.readme)
        img_rel = os.path.relpath(a.img_dir or report, readme.parent).replace(os.sep, "/")
        text = readme.read_text()
        text = replace_block(text, "SONUCLAR", results_markdown(agg_by_gripper, abl_rows, a, img_rel, gifs, qtables))
        text = replace_block(text, "OZET", f"**Ana bulgu:** {headline(agg_by_gripper)}")
        readme.write_text(text)
        print(f"README güncellendi → {readme}")
    (report / "headline.txt").write_text(headline(agg_by_gripper) + "\n")
    print(f"\nBitti ({time.time() - t_start:.0f} sn). Rapor: {report}")


if __name__ == "__main__":
    main()
