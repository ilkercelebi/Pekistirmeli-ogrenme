"""Etkileşimli demo arayüzü (sınıf sunumu ve video için).

Çalıştırma:
    streamlit run app.py

Sekmeler:
  1. Canlı demo        — kutu üret, ajanı adım adım ya da sonuna kadar çalıştır
  2. Yan yana          — aynı kutuda iki yöntemi karşılaştır (ör. rastgele vs Q-learning)
  3. Eğitim sonuçları  — results/ altındaki eğitim eğrileri ve test tablosu
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from binpick import ExperimentConfig, build, load_run, make_agent, run_episode
from binpick.viz import OUTCOME_TR, render

RESULTS = Path("results")

st.set_page_config(page_title="Bin Picking · Q-learning", layout="wide")


def list_runs():
    if not RESULTS.exists():
        return []
    return sorted(p for p in RESULTS.iterdir() if (p / "agent.npz").exists() and (p / "config.json").exists())


def make(method: str, gripper: str, seed: int):
    """(cfg, agent) döndürür. method: 'random', 'heuristic' ya da bir eğitim klasörü."""
    if method in ("random", "heuristic"):
        cfg = ExperimentConfig()
        cfg.gripper.kind = gripper
        return cfg, make_agent(method, 0, seed=seed)
    cfg, agent = load_run(RESULTS / method, seed=seed)
    return cfg, agent


def method_options(gripper: str):
    opts = ["random", "heuristic"]
    for run in list_runs():
        kind = json.loads((run / "config.json").read_text())["gripper"]["kind"]
        if kind == gripper:
            opts.append(run.name)
    return opts


METHOD_TR = {"random": "Rastgele", "heuristic": "Sezgisel (en yüksek nokta)"}

# ------------------------------------------------------------------ kenar çubuğu

st.sidebar.title("Ayarlar")
gripper = st.sidebar.radio("Tutucu", ["parallel_jaw", "vacuum"],
                           format_func=lambda k: "Parmaklı" if k == "parallel_jaw" else "Vakumlu")
options = method_options(gripper)
if len(options) == 2:
    st.sidebar.info("Henüz eğitilmiş ajan yok. `python train.py` ile eğitince burada görünür.")
seed = st.sidebar.number_input("Kutu tohumu (seed)", min_value=0, value=7, step=1)
show_q = st.sidebar.checkbox("Q-haritasını göster", value=True)

st.title("Düzensiz Kutudan Parça Alma: Tablo Tabanlı Pekiştirmeli Öğrenme")
st.caption("Durum: yükseklik haritası · Eylem: (x, y, θ) · Ödül: +1 başarı, −1 çarpışma, −0.2 başarısız kavrama")

tab_live, tab_cmp, tab_res = st.tabs(["Canlı demo", "Yan yana", "Eğitim sonuçları"])

# ------------------------------------------------------------------ 1. canlı demo

with tab_live:
    method = st.selectbox("Yöntem", options, format_func=lambda m: METHOD_TR.get(m, m), key="live_method")
    key = (gripper, method, int(seed))
    ss = st.session_state
    if ss.get("live_key") != key:
        cfg, agent = make(method, gripper, int(seed))
        env, fe = build(cfg, seed=int(seed))
        obs = env.reset(seed=int(seed))
        ss.update(live_key=key, env=env, fe=fe, agent=agent, obs=obs, history=[], last=None, done=False)

    c1, c2, c3 = st.columns(3)
    if c1.button("Yeni kutu (aynı tohum)"):
        ss.obs = ss.env.reset(seed=int(seed))
        ss.history, ss.last, ss.done = [], None, False
    step_clicked = c2.button("1 adım", disabled=ss.done)
    run_clicked = c3.button("Kutuyu boşalt", disabled=ss.done)

    def do_step():
        feats = ss.fe.compute(ss.obs)
        action, _ = ss.agent.act(feats, explore=False)
        obs, reward, term, trunc, info = ss.env.step(action)
        ss.last = (feats, action, info)
        ss.obs = obs
        ss.done = term or trunc
        ss.history.append({"adım": info["step"], "θ": f"{info['theta_deg']:.0f}°",
                           "nokta (satır, sütun)": f"({action.row}, {action.col})",
                           "sonuç": OUTCOME_TR[info["outcome"]], "ödül": reward,
                           "açıklama": info["reason"], "kalan parça": info["parts_left"]})

    if step_clicked:
        do_step()
    if run_clicked:
        while not ss.done:
            do_step()

    if ss.last is not None:
        feats, action, info = ss.last
        img = render(ss.env, agent=ss.agent, feats=feats, action=action, info=info, show_q=show_q)
    else:
        img = render(ss.env, agent=ss.agent, feats=ss.fe.compute(ss.obs), show_q=show_q, title="Başlangıç")
    st.image(img, width="stretch")

    if ss.history:
        df = pd.DataFrame(ss.history)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam ödül", f"{df['ödül'].sum():.1f}")
        m2.metric("Başarılı", int((df["ödül"] > 0).sum()))
        m3.metric("Çarpışma", int((df["ödül"] <= -1).sum()))
        m4.metric("Kalan parça", int(df["kalan parça"].iloc[-1]))
        st.dataframe(df, hide_index=True, width="stretch")
    if ss.done:
        st.success("Bölüm bitti.")

# ------------------------------------------------------------------ 2. yan yana

with tab_cmp:
    st.write("Aynı kutuyu iki yöntem ayrı ayrı boşaltmaya çalışır.")
    a_col, b_col = st.columns(2)
    m_a = a_col.selectbox("Sol", options, index=0, format_func=lambda m: METHOD_TR.get(m, m), key="cmp_a")
    m_b = b_col.selectbox("Sağ", options, index=len(options) - 1, format_func=lambda m: METHOD_TR.get(m, m), key="cmp_b")
    if st.button("Karşılaştır"):
        for col, m in ((a_col, m_a), (b_col, m_b)):
            cfg, agent = make(m, gripper, int(seed))
            env, fe = build(cfg, seed=int(seed))
            stats = run_episode(env, agent, fe, learn=False, seed=int(seed))
            col.image(render(env, title=f"{METHOD_TR.get(m, m)} · son durum", show_q=False), width="stretch")
            col.metric("Boşaltılan parça", f"{stats['success']} / {stats['parts']}")
            col.write(f"Adım: {stats['steps']} · Çarpışma: {stats['collision']} · "
                      f"Başarısız: {stats['fail']} · Getiri: {stats['return']:.1f}")

# ------------------------------------------------------------------ 3. sonuçlar

with tab_res:
    runs = [r for r in list_runs() if r.name in options]
    if not runs:
        st.info("Bu tutucu için eğitim sonucu yok.")
    else:
        window = st.slider("Hareketli ortalama penceresi", 10, 500, 100, step=10)
        metric = st.selectbox("Ölçüt", ["return", "success_rate", "collision_rate", "cleared"],
                              format_func=lambda k: {"return": "Getiri", "success_rate": "Başarı oranı",
                                                     "collision_rate": "Çarpışma oranı",
                                                     "cleared": "Boşaltılan parça oranı"}[k])
        curves = {}
        for r in runs:
            df = pd.read_csv(r / "metrics.csv")
            curves[r.name] = df[metric].rolling(window, min_periods=1).mean()
        st.line_chart(pd.DataFrame(curves))

    summary = RESULTS / f"eval_{gripper}" / "summary.csv"
    if summary.exists():
        st.subheader("Test kutularında karşılaştırma")
        st.dataframe(pd.read_csv(summary), hide_index=True, width="stretch")
        bars = summary.parent / "bars.png"
        if bars.exists():
            st.image(str(bars))
    for name in ("learning_curves.png", "greedy_eval.png"):
        path = RESULTS / "plots" / name
        if path.exists():
            st.image(str(path))

    if runs:
        st.subheader("Öğrenilen Q tablosu: en iyi ve en kötü durumlar")
        run = st.selectbox("Koşu", [r.name for r in runs], key="qtab_run")
        cfg, agent = load_run(RESULTS / run)
        env, fe = build(cfg)
        visited = np.flatnonzero(agent.N > 0)
        df = pd.DataFrame([{**fe.describe(s), "Q": agent.Q[s], "ziyaret": int(agent.N[s])} for s in visited])
        if not df.empty:
            df = df[df["ziyaret"] >= 5].sort_values("Q", ascending=False)
            st.write("En yüksek Q değerli durumlar")
            st.dataframe(df.head(10), hide_index=True, width="stretch")
            st.write("En düşük Q değerli durumlar")
            st.dataframe(df.tail(10), hide_index=True, width="stretch")
