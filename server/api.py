"""Arayüz için HTTP API (FastAPI).

Simülasyon ve ajanlar Python'da çalışır; React arayüzü bu uç noktalar üzerinden
kutu durumunu, ajan kararlarını, Q-haritasını ve eğitim sonuçlarını alır.

    uvicorn server.api:app --port 8000      (ya da: python serve.py)
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from binpick import Action, ExperimentConfig, build, load_run, make_agent, run_episode
from binpick.agents import TabularAgent
from binpick.analysis import extreme_states, marginal_q
from binpick.features import BIN_LABELS, FEATURE_INFO
from binpick.scenarios import SCENARIOS, run_scenario
from binpick.training import train

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="Bin Picking RL")

BASELINES = {"random": "Rastgele", "heuristic": "Sezgisel"}
ALGO_LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA", "double_q": "Double Q-learning"}


# ------------------------------------------------------------------ yardımcılar


def r2(a) -> list:
    """NaN'ları null yapan, 2 basamağa yuvarlanmış iç içe liste."""
    a = np.round(np.asarray(a, dtype=float), 2)
    return [[None if np.isnan(v) else float(v) for v in row] for row in a]


def gripper_kind_of_run(run: Path) -> str:
    return json.loads((run / "config.json").read_text())["gripper"]["kind"]


def list_runs() -> list[Path]:
    """Kayıtlı eğitimler: önce algoritma sırası (Q-learning, SARSA, Double Q), sonra ad."""
    if not RESULTS.exists():
        return []
    runs = [p for p in RESULTS.iterdir() if (p / "agent.npz").exists() and (p / "config.json").exists()]
    order = list(ALGO_LABELS)

    def key(p: Path):
        algo = json.loads((p / "config.json").read_text())["agent"]["algo"]
        return (order.index(algo) if algo in order else len(order), p.name)

    return sorted(runs, key=key)


def run_label(name: str, algo: str) -> str:
    """'q_learning_parallel_jaw_s1' → 'Q-learning · tohum 1'; diğer adlar algoritma etiketiyle birlikte."""
    label = ALGO_LABELS.get(algo, algo)
    m = re.fullmatch(rf"{algo}_(parallel_jaw|vacuum)_s(\d+)", name)
    return f"{label} · tohum {m.group(2)}" if m else f"{label} · {name}"


def make_method(method: str, gripper: str, seed: int = 0):
    if method in BASELINES:
        cfg = ExperimentConfig()
        cfg.gripper.kind = gripper
        return cfg, make_agent(method, 0, seed=seed)
    run = RESULTS / method
    if not (run / "agent.npz").exists():
        raise HTTPException(404, f"Eğitim bulunamadı: {method}")
    return load_run(run, seed=seed)


def method_label(method: str) -> str:
    if method in BASELINES:
        return BASELINES[method]
    cfg = RESULTS / method / "config.json"
    if not cfg.exists():
        return method
    return run_label(method, json.loads(cfg.read_text())["agent"]["algo"])


def parts_json(env) -> list[dict]:
    return [{"id": i, "kind": p.kind, "cx": p.cx, "cy": p.cy, "length": p.length, "width": p.width,
             "angle": p.angle, "base": p.base, "thickness": p.thickness, "alive": p.alive}
            for i, p in enumerate(env.parts)]


def action_json(env, a: Action) -> dict:
    if a.stop:
        return {"angle_idx": -1, "row": -1, "col": -1, "theta_deg": 0.0, "stop": True}
    return {"angle_idx": a.angle_idx, "row": a.row, "col": a.col,
            "theta_deg": float(np.degrees(env.angles[a.angle_idx]))}


def feature_vector(fe, feats, a: Action) -> dict:
    sid = int(feats.state_ids[a.angle_idx, a.row, a.col])
    bins = fe.describe(sid)
    return {"state_id": sid, "valid": bool(feats.valid[a.angle_idx, a.row, a.col]),
            "features": [{"name": n, "bin": b, "label": BIN_LABELS[n][b][0], "tone": BIN_LABELS[n][b][1]}
                         for n, b in bins.items()]}


def meta_for(cfg) -> dict:
    env, fe = build(cfg)
    g, e = cfg.gripper, cfg.env
    return {
        "gripper": {"kind": g.kind, "opening": g.opening, "finger_thickness": g.finger_thickness,
                    "finger_width": g.finger_width, "grasp_depth": g.grasp_depth, "cup_radius": g.cup_radius},
        "grid": {"size": e.size, "wall_thickness": e.wall_thickness, "wall_height": e.wall_height,
                 "max_stack_height": e.max_stack_height},
        "angles_deg": [float(v) for v in np.degrees(env.angles)],
        "rewards": {"success": e.reward_success, "collision": e.reward_collision, "fail": e.reward_fail},
        "features": [{"name": n, "bins": [{"label": lab, "tone": t} for lab, t in BIN_LABELS[n]],
                      "info": FEATURE_INFO[n]} for n in fe.names],
        "n_states": fe.n_states,
        "dims": fe.dims,
        "n_parts": [e.n_parts_min, e.n_parts_max],
        "candidates": int(fe.candidates.sum()) * len(env.angles),
    }


# ------------------------------------------------------------------ oturumlar


@dataclass
class Session:
    cfg: ExperimentConfig
    env: object
    fe: object
    agent: object
    method: str
    seed: int
    obs: np.ndarray = None
    feats: object = None
    history: list = field(default_factory=list)
    done: bool = False

    def refresh(self):
        self.feats = self.fe.compute(self.obs, self.env.fail_map, self.env.fail_streak)

    def state(self) -> dict:
        env = self.env
        out = {
            "id": None, "method": self.method, "method_label": method_label(self.method), "seed": self.seed,
            "gripper": self.cfg.gripper.kind, "parts": parts_json(env), "depth": r2(self.obs),
            "parts_left": env.n_alive, "n_initial": env.n_initial, "steps": env.steps, "max_steps": env.max_steps,
            "done": self.done, "history": self.history, "learns": isinstance(self.agent, TabularAgent),
            "qmap": None,
        }
        if isinstance(self.agent, TabularAgent) and self.feats is not None:
            qmax, best = self.agent.q_map(self.feats)
            out["qmap"] = {"q": r2(qmax), "best_angle": best.tolist()}
        return out


SESSIONS: dict[str, Session] = {}
LOCK = threading.Lock()


def get_session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Oturum bulunamadı; sayfayı yenileyin.")
    return s


def session_state(sid: str, s: Session) -> dict:
    st = s.state()
    st["id"] = sid
    return st


class SessionIn(BaseModel):
    gripper: str = "parallel_jaw"
    method: str = "heuristic"
    seed: int = 7


class ResetIn(BaseModel):
    seed: int | None = None


class ActionIn(BaseModel):
    angle_idx: int
    row: int
    col: int


class StepIn(BaseModel):
    action: ActionIn | None = None


@app.get("/api/meta")
def meta(gripper: str = "parallel_jaw"):
    cfg = ExperimentConfig()
    cfg.gripper.kind = gripper
    return meta_for(cfg)


@app.get("/api/methods")
def methods(gripper: str = "parallel_jaw"):
    out = [{"id": k, "label": v, "learns": False} for k, v in BASELINES.items()]
    for run in list_runs():
        if gripper_kind_of_run(run) == gripper:
            out.append({"id": run.name, "label": method_label(run.name), "learns": True})
    return out


@app.post("/api/session")
def create_session(body: SessionIn):
    cfg, agent = make_method(body.method, body.gripper, seed=body.seed)
    env, fe = build(cfg, seed=body.seed)
    s = Session(cfg=cfg, env=env, fe=fe, agent=agent, method=body.method, seed=body.seed)
    s.obs = env.reset(seed=body.seed)
    agent.allow_stop = cfg.env.allow_stop
    agent.begin_episode()
    s.refresh()
    sid = uuid.uuid4().hex[:12]
    with LOCK:
        if len(SESSIONS) > 200:
            SESSIONS.pop(next(iter(SESSIONS)))
        SESSIONS[sid] = s
    return session_state(sid, s)


@app.post("/api/session/{sid}/reset")
def reset_session(sid: str, body: ResetIn):
    s = get_session(sid)
    if body.seed is not None:
        s.seed = body.seed
    s.obs = s.env.reset(seed=s.seed)
    s.agent.begin_episode()
    s.history, s.done = [], False
    s.refresh()
    return session_state(sid, s)


@app.post("/api/session/{sid}/step")
def step_session(sid: str, body: StepIn):
    s = get_session(sid)
    if s.done:
        raise HTTPException(400, "Bölüm bitti; yeni kutu oluşturun.")
    if body.action is None:
        action, _ = s.agent.act(s.feats, explore=False)
        chosen_by = "agent"
    else:
        a = body.action
        n = s.env.shape[0]
        if not (0 <= a.row < n and 0 <= a.col < n and 0 <= a.angle_idx < len(s.env.angles)):
            raise HTTPException(400, "Geçersiz eylem")
        action = Action(a.angle_idx, a.row, a.col)
        chosen_by = "user"

    if action.stop:
        fv, q = {"features": []}, 0.0 if isinstance(s.agent, TabularAgent) else None
    else:
        fv = feature_vector(s.fe, s.feats, action)
        q = float(s.agent.Q[fv["state_id"]]) if isinstance(s.agent, TabularAgent) else None
    parts_before = parts_json(s.env)
    s.obs, reward, term, trunc, info = s.env.step(action)
    s.agent.observe(info)
    s.done = term or trunc
    record = {"step": info["step"], "action": action_json(s.env, action), "reward": reward,
              "outcome": info["outcome"], "reason": info["reason"], "code": info["code"],
              "part": info["part"], "anim": info["anim"], "chosen_by": chosen_by, "q": q,
              "features": fv["features"]}
    s.history.append(record)
    s.refresh()
    return {"event": record, "parts_before": parts_before, "state": session_state(sid, s)}


@app.post("/api/session/{sid}/preview")
def preview(sid: str, body: ActionIn):
    s = get_session(sid)
    a = Action(body.angle_idx, body.row, body.col)
    fv = feature_vector(s.fe, s.feats, a)
    fv["q"] = float(s.agent.Q[fv["state_id"]]) if isinstance(s.agent, TabularAgent) else None
    fv["wall_hit"] = bool(s.env._wall_hit((a.row, a.col), float(s.env.angles[a.angle_idx])))
    return fv


@app.get("/api/session/{sid}/features")
def feature_maps(sid: str, angle_idx: int = 0):
    s = get_session(sid)
    f, fe, g = s.feats, s.fe, s.cfg.gripper
    k = max(0, min(angle_idx, len(s.env.angles) - 1))
    maps = {
        "yükseklik": f.raw["level"],
        "düzlük": np.digitize(f.raw["std"], fe.f.flat_bins),
        "yakında_yüksek": f.raw["higher"],
        "duvar": f.raw["wall"][k],
    }
    if g.kind == "vacuum":
        seal = f.raw["seal"]
        maps["sızdırmazlık"] = np.where(seal >= fe.f.seal_bins[0], 0, np.where(seal >= fe.f.seal_bins[1], 1, 2))
        maps["yüzey_kapalı"] = f.raw["face_higher"]
    else:
        ext = f.raw["extent"][k]
        maps["parmak_boşluğu"] = np.digitize(f.raw["clear"][k], fe.f.clear_bins)
        maps["genişlik"] = np.where(ext <= g.opening - 2, 0, np.where(ext <= g.opening, 1, 2))
    wall = s.env.wall_mask
    return {name: np.where(wall, -1, m).astype(int).tolist() for name, m in maps.items()}


# ------------------------------------------------------------------ senaryolar


@app.get("/api/scenarios")
def scenarios():
    out = []
    for sc in SCENARIOS:
        before, action, reward, info = run_scenario(sc)
        out.append({"key": sc.key, "title": sc.title, "explanation": sc.explanation, "gripper": sc.kind,
                    "parts": parts_json(before), "depth": r2(before.heightmap),
                    "event": {"action": action_json(before, action), "reward": reward, "outcome": info["outcome"],
                              "reason": info["reason"], "code": info["code"], "part": info["part"],
                              "anim": info["anim"]}})
    return out


# ------------------------------------------------------------------ değerlendirme

EVAL_SEED0 = 2_000_000


class EvalIn(BaseModel):
    gripper: str = "parallel_jaw"
    methods: list[str] = ["random", "heuristic"]
    episodes: int = 30


@app.post("/api/evaluate")
def evaluate(body: EvalIn):
    n = max(1, min(body.episodes, 300))
    out = []
    for m in body.methods:
        cfg, agent = make_method(m, body.gripper, seed=0)
        env, fe = build(cfg, seed=0)
        rows = [run_episode(env, agent, fe, learn=False, seed=EVAL_SEED0 + i) for i in range(n)]
        res = {"method": m, "label": method_label(m)}
        for k in ("return", "success_rate", "collision_rate", "fail_rate", "cleared", "steps"):
            v = np.array([r[k] for r in rows], dtype=float)
            res[k] = float(v.mean())
            res[k + "_std"] = float(v.std())
        out.append(res)
    return {"episodes": n, "results": out}


# ------------------------------------------------------------------ eğitim işleri


@dataclass
class Job:
    id: str
    name: str
    algo: str
    gripper: str
    episodes: int
    status: str = "running"
    episode: int = 0
    started: float = field(default_factory=time.time)
    curve: list = field(default_factory=list)
    evals: list = field(default_factory=list)
    error: str | None = None
    cancel: bool = False


JOBS: dict[str, Job] = {}


class TrainIn(BaseModel):
    algo: str = "q_learning"
    gripper: str = "parallel_jaw"
    episodes: int = 1500
    name: str | None = None


def _run_job(job: Job, cfg):
    window = []

    def cb(ep, row, agent, eval_row):
        job.episode = ep + 1
        window.append(row)
        if len(window) >= 20:
            job.curve.append({"episode": ep + 1, "epsilon": row["epsilon"],
                              **{k: float(np.mean([w[k] for w in window]))
                                 for k in ("return", "success_rate", "collision_rate", "cleared")}})
            window.clear()
        if eval_row:
            job.evals.append(eval_row)
        return not job.cancel

    try:
        train(cfg, job.episodes, seed=0, run_dir=RESULTS / job.name, eval_every=max(50, job.episodes // 10),
              eval_episodes=15, callback=cb)
        job.status = "cancelled" if job.cancel else "done"
    except Exception as exc:  # noqa: BLE001
        job.status, job.error = "error", str(exc)


@app.post("/api/train")
def start_train(body: TrainIn):
    if any(j.status == "running" for j in JOBS.values()):
        raise HTTPException(409, "Zaten çalışan bir eğitim var.")
    if body.algo not in ALGO_LABELS or body.gripper not in ("parallel_jaw", "vacuum"):
        raise HTTPException(400, "Geçersiz parametre")
    cfg = ExperimentConfig()
    cfg.gripper.kind = body.gripper
    cfg.agent.algo = body.algo
    cfg.agent.eps_decay_episodes = max(1, int(body.episodes * 0.5))
    name = body.name or f"{body.algo}_{body.gripper}_{body.episodes}"
    name = "".join(ch for ch in name if ch.isalnum() or ch in "_-")[:60] or "run"
    job = Job(id=uuid.uuid4().hex[:10], name=name, algo=body.algo, gripper=body.gripper,
              episodes=max(10, min(body.episodes, 20000)))
    JOBS[job.id] = job
    threading.Thread(target=_run_job, args=(job, cfg), daemon=True).start()
    return job.__dict__


@app.get("/api/train/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı")
    return {**job.__dict__, "elapsed": time.time() - job.started}


@app.post("/api/train/{job_id}/cancel")
def cancel_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "İş bulunamadı")
    job.cancel = True
    return {"ok": True}


@app.get("/api/train")
def current_job():
    running = [j for j in JOBS.values() if j.status == "running"]
    return {**running[0].__dict__, "elapsed": time.time() - running[0].started} if running else None


# ------------------------------------------------------------------ eğitim sonuçları


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


@app.get("/api/runs")
def runs(gripper: str | None = None):
    out = []
    for run in list_runs():
        cfg = json.loads((run / "config.json").read_text())
        if gripper and cfg["gripper"]["kind"] != gripper:
            continue
        ev = _read_csv(run / "eval.csv")
        n = sum(1 for _ in open(run / "metrics.csv")) - 1 if (run / "metrics.csv").exists() else 0
        out.append({"name": run.name, "label": run_label(run.name, cfg["agent"]["algo"]),
                    "algo": cfg["agent"]["algo"], "gripper": cfg["gripper"]["kind"],
                    "episodes": n, "last_eval": ev[-1] if ev else None})
    return out


@app.get("/api/runs/{name}")
def run_detail(name: str, window: int = 50):
    run = RESULTS / name
    if not (run / "agent.npz").exists():
        raise HTTPException(404, "Eğitim bulunamadı")
    metrics = _read_csv(run / "metrics.csv")
    curve = []
    if metrics:
        keys = ("return", "success_rate", "collision_rate", "cleared")
        arr = {k: np.array([m[k] for m in metrics]) for k in keys}
        w = max(1, min(window, len(metrics)))
        stride = max(1, len(metrics) // 300)
        for i in range(w - 1, len(metrics), stride):
            curve.append({"episode": i + 1, "epsilon": metrics[i]["epsilon"],
                          **{k: float(arr[k][i - w + 1 : i + 1].mean()) for k in keys}})

    cfg, agent = load_run(run)
    _, fe = build(cfg)
    marginal = marginal_q(agent, fe)
    visited = np.flatnonzero(agent.N > 0)

    def describe(s):
        return {"state_id": int(s), "q": float(agent.Q[s]), "visits": int(agent.N[s]),
                "features": [{"name": n, "label": BIN_LABELS[n][b][0], "tone": BIN_LABELS[n][b][1]}
                             for n, b in fe.describe(s).items()]}

    best, worst = extreme_states(agent, fe)
    return {
        "name": name, "config": cfg.to_dict(), "curve": curve, "evals": _read_csv(run / "eval.csv"),
        "marginal": marginal, "visited": int(len(visited)), "n_states": fe.n_states,
        "best": [describe(s) for s in best], "worst": [describe(s) for s in worst],
    }


# ------------------------------------------------------------------ derlenmiş arayüz

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = WEB_DIST / path
        if path and target.is_file():
            return FileResponse(target)
        # index.html önbelleğe alınmasın: yeniden derlemeden sonra eski parça dosyaları istenmesin
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
