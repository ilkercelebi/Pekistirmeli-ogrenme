"""Ajanlar: tablo tabanlı Q-learning ve SARSA, karşılaştırma için rastgele ve sezgisel.

Hepsi aynı arayüzü kullanır:
    act(feats, explore) -> (Action, state_id)
    learn(sid, reward, next_feats, next_sid, terminated)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import AgentConfig, ExperimentConfig
from .env import STOP_ACTION, Action
from .features import FeatureMaps


def _to_action(flat_index: int, shape) -> Action:
    k, r, c = np.unravel_index(int(flat_index), shape)
    return Action(int(k), int(r), int(c))


class BaseAgent:
    name = "base"
    learns = False
    allow_stop = False       # ortam "dur" eylemine izin veriyor mu (run_episode ayarlar)

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def set_episode(self, episode: int) -> None:
        pass

    def begin_episode(self) -> None:
        """Yeni kutu: bölüm içi sayaçları sıfırlar."""

    def observe(self, info: dict) -> None:
        """Her adımın sonucunu görür (öğrenmeden bağımsız; ör. üst üste başarısızlık sayacı)."""

    def act(self, feats: FeatureMaps, explore: bool = False):
        raise NotImplementedError

    def learn(self, sid, reward, next_feats, next_sid, terminated) -> None:
        pass

    def q_map(self, feats: FeatureMaps):
        return None


class RandomAgent(BaseAgent):
    """Aday eylemler arasından tamamen rastgele seçim (alt sınır)."""

    name = "random"

    def act(self, feats, explore=False):
        idx = np.flatnonzero(feats.valid)
        j = idx[self.rng.integers(len(idx))]
        return _to_action(j, feats.valid.shape), int(feats.state_ids.flat[j])


class HeuristicAgent(BaseAgent):
    """Elle yazılmış kural: en yüksek noktayı tut.

    Duvara çarpacağı bilinen açılar elenir; parmaklıda kalan açılardan, o noktada
    ölçülen yüzey genişliği en küçük olan yön seçilir. Öğrenme yoktur; Q-learning'in
    bu kuralı geçip geçemediği deneyin ana sorusudur.
    """

    name = "heuristic"
    use_face = False
    tabu_radius = 2.0        # başarısız olunan noktanın bu yarıçapı bir daha denenmez (hafızanın karşılığı)

    def begin_episode(self) -> None:
        self.tabu: list[tuple[int, int]] = []

    def observe(self, info: dict) -> None:
        if info["outcome"] == "success":
            self.tabu = []          # kutu değişti
        elif info.get("row", -1) >= 0:
            self.tabu.append((info["row"], info["col"]))

    def _not_tabu(self, shape) -> np.ndarray:
        ok = np.ones(shape, dtype=bool)
        if getattr(self, "tabu", None):
            yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
            for r, c in self.tabu:
                ok &= (yy - r) ** 2 + (xx - c) ** 2 > self.tabu_radius ** 2
        return ok

    def act(self, feats, explore=False):
        free = self._not_tabu(feats.valid.shape[1:])
        if self.allow_stop and not (free & (feats.raw["level"] > 0) & feats.valid[0]).any():
            return STOP_ACTION, -1          # denenmemiş parça noktası kalmadı
        safe = np.stack([w > 0 for w in feats.raw["wall"]]) & feats.valid   # (A, H, W)
        if not safe.any():
            safe = feats.valid
        safe = safe & free
        if not safe.any():
            safe = feats.valid & free if (feats.valid & free).any() else feats.valid
        h = np.where(safe.any(axis=0), feats.raw["height"], -np.inf)
        seal = feats.raw.get("seal")
        if seal is not None:
            # Vakum: vantuzun tam oturduğu (ölçülen sızdırmazlık ≥ %85) en yüksek parça noktası;
            # yoksa parça üstündeki en iyi sızdırmazlık. Aksi hâlde dar parçaların kenarı seçilir
            # ve kutu değişmediği için aynı başarısız kavrama tekrar tekrar denenir.
            on_part = np.isfinite(h) & (feats.raw["level"] > 0)
            fits = on_part & (seal >= 0.85)
            face = feats.raw.get("face_higher")
            if self.use_face and face is not None and (fits & (face == 0)).any():
                fits &= face == 0
            if fits.any():
                h = np.where(fits, h, -np.inf)
            elif on_part.any():
                h = np.where(on_part, seal, -np.inf)
        best = np.flatnonzero(h >= h.max() - 1e-9)
        r, c = np.unravel_index(best[self.rng.integers(len(best))], h.shape)
        angles = np.flatnonzero(safe[:, r, c])
        if feats.raw["extent"]:
            ext = np.array([feats.raw["extent"][k][r, c] for k in angles])
            angles = angles[ext == ext.min()]
        k = int(self.rng.choice(angles))
        return Action(k, int(r), int(c)), int(feats.state_ids[k, r, c])


class HeuristicFaceAgent(HeuristicAgent):
    """Sezgisel + yüzey bilgisi: vakumda, Q-learning'e verilen `yüzey_kapalı` bilgisini de kullanır.

    Aynı kural, yalnız aday noktalar önce üzerinde durdukları düz yüzey daha yüksek bir yüzeye
    değmeyenlerle (`face_higher == 0`) sınırlanır; böyle nokta yoksa sezgiselin kuralına dönülür.
    Öğrenen ajanla aynı bilgiye sahip elle yazılmış bir kuraldır: vakumdaki kazancın öğrenmeden mi
    yoksa yalnızca ek bilgiden mi geldiğini ayırmak için. Parmaklı tutucuda sezgiselle aynıdır.
    """

    name = "heuristic_face"
    use_face = True


class TabularAgent(BaseAgent):
    """Q(s, a) ≈ Q[φ(s, a)] tablosu üzerinde ε-açgözlü TD öğrenme."""

    learns = True

    def __init__(self, n_states: int, cfg: AgentConfig | None = None, seed=None):
        super().__init__(seed)
        self.cfg = cfg or AgentConfig()
        self.n_states = n_states
        self.Q = np.full(n_states, self.cfg.q_init, dtype=float)
        self.N = np.zeros(n_states, dtype=np.int64)
        self.epsilon = self.cfg.eps_start
        self.td_errors: list[float] = []

    @property
    def name(self):
        return self.cfg.algo

    def set_episode(self, episode: int) -> None:
        c = self.cfg
        frac = min(1.0, episode / max(1, c.eps_decay_episodes))
        self.epsilon = c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, feats, explore=False):
        idx = np.flatnonzero(feats.valid)
        ids = feats.state_ids.flat[idx]
        if explore and self.rng.random() < self.epsilon:
            j = int(self.rng.integers(len(idx)))
        else:
            q = self.Q[ids]
            if not explore and self.cfg.avoid_unvisited:
                q = self._known(q, ids)
            # Durmanın değeri tam olarak bilinir: ödül 0 ve bölüm biter. En iyi kavramanın beklenen
            # değeri bunun altına düştüğünde (her deneme zarar bekleniyorsa) ajan durur.
            if self.allow_stop and q.max() < self.cfg.stop_margin:
                return STOP_ACTION, -1
            best = np.flatnonzero(q >= q.max() - 1e-9)
            j = int(best[self.rng.integers(len(best))])
        return _to_action(idx[j], feats.valid.shape), int(ids[j])

    def _next_value(self, q_max: float) -> float:
        """Bir sonraki durumun değeri: dur eylemi açıksa en az 0 (durmak her zaman mümkün)."""
        return max(q_max, 0.0) if self.allow_stop else q_max

    def _known(self, q, ids):
        """Ziyaret edilmiş hücre varsa ziyaret edilmemişleri dışarıda bırakır."""
        seen = self.N[ids] > 0
        return np.where(seen, q, -np.inf) if seen.any() else q

    def _alpha(self, sid: int) -> float:
        c = self.cfg
        if c.alpha_mode == "constant":
            return c.alpha
        return max(c.alpha_min, 1.0 / (self.N[sid] ** c.alpha_omega))

    def _bootstrap(self, next_feats, next_sid) -> float:
        raise NotImplementedError

    def learn(self, sid, reward, next_feats, next_sid, terminated):
        if sid < 0:          # "dur" eylemi tabloda değildir; değeri sabit 0
            return
        target = reward
        if not terminated and next_feats is not None:
            target += self.cfg.gamma * self._bootstrap(next_feats, next_sid)
        self.N[sid] += 1
        delta = target - self.Q[sid]
        self.Q[sid] += self._alpha(sid) * delta
        self.td_errors.append(float(delta))

    def q_map(self, feats):
        """Her piksel için açılar üzerinden en iyi Q değeri (görselleştirme)."""
        q = np.where(feats.valid, self.Q[feats.state_ids], -np.inf)
        best_angle = q.argmax(axis=0)
        qmax = q.max(axis=0)
        qmax[~feats.valid.any(axis=0)] = np.nan
        return qmax, best_angle

    # --------------------------------------------------- kaydet / yükle

    def save(self, path: Path) -> None:
        np.savez(path, Q=self.Q, N=self.N)

    def load_arrays(self, path: Path) -> None:
        data = np.load(path)
        if data["Q"].shape != self.Q.shape:
            raise ValueError("Kaydedilen tablo boyutu bu konfigürasyonla uyuşmuyor.")
        self.Q, self.N = data["Q"], data["N"]


class QLearningAgent(TabularAgent):
    """Off-policy: hedef = r + γ · max_a' Q(s', a')."""

    def _bootstrap(self, next_feats, next_sid):
        ids = np.unique(next_feats.state_ids[next_feats.valid])
        q = self.Q[ids]
        if self.cfg.avoid_unvisited:
            q = self._known(q, ids)
        return self._next_value(float(q.max()))


class SarsaAgent(TabularAgent):
    """On-policy: hedef = r + γ · Q(s', a'), a' gerçekten seçilen sonraki eylem."""

    def _bootstrap(self, next_feats, next_sid):
        return 0.0 if next_sid < 0 else float(self.Q[next_sid])


class DoubleQAgent(TabularAgent):
    """Double Q-learning (van Hasselt, 2010): iki bağımsız tablo, Q_A ve Q_B.

    Her güncellemede tablolardan biri rastgele seçilir; sonraki durumdaki en iyi eylem o tabloyla
    seçilir ama değeri öbür tablodan okunur. Böylece tek tablodaki max işleminin yarattığı
    aşırı iyimser tahmin (overestimation bias) azalır. Eylem seçimi iki tablonun ortalamasıyla yapılır
    (`self.Q`, arayüz ve analiz de bunu kullanır).
    """

    def __init__(self, n_states: int, cfg: AgentConfig | None = None, seed=None):
        super().__init__(n_states, cfg, seed)
        self.QA = self.Q.copy()
        self.QB = self.Q.copy()

    def learn(self, sid, reward, next_feats, next_sid, terminated):
        if sid < 0:
            return
        upd, other = (self.QA, self.QB) if self.rng.random() < 0.5 else (self.QB, self.QA)
        target = reward
        if not terminated and next_feats is not None:
            ids = np.unique(next_feats.state_ids[next_feats.valid])
            a_star = ids[int(np.argmax(upd[ids]))]
            value = float(other[a_star])
            if self.allow_stop and upd[a_star] < 0.0:      # seçen tablo durmayı tercih ediyor
                value = 0.0
            target += self.cfg.gamma * value
        self.N[sid] += 1
        delta = target - upd[sid]
        upd[sid] += self._alpha(sid) * delta
        self.Q[sid] = 0.5 * (self.QA[sid] + self.QB[sid])
        self.td_errors.append(float(delta))

    def save(self, path: Path) -> None:
        np.savez(path, Q=self.Q, N=self.N, QA=self.QA, QB=self.QB)

    def load_arrays(self, path: Path) -> None:
        super().load_arrays(path)
        data = np.load(path)
        self.QA = data["QA"] if "QA" in data else self.Q.copy()
        self.QB = data["QB"] if "QB" in data else self.Q.copy()


def make_agent(name: str, n_states: int, cfg: AgentConfig | None = None, seed=None) -> BaseAgent:
    if name == "random":
        return RandomAgent(seed)
    if name == "heuristic":
        return HeuristicAgent(seed)
    if name == "heuristic_face":
        return HeuristicFaceAgent(seed)
    cfg = cfg or AgentConfig()
    cfg.algo = name
    if name == "q_learning":
        return QLearningAgent(n_states, cfg, seed)
    if name == "sarsa":
        return SarsaAgent(n_states, cfg, seed)
    if name == "double_q":
        return DoubleQAgent(n_states, cfg, seed)
    raise ValueError(f"Bilinmeyen ajan: {name}")


def load_run(run_dir: str | Path, seed=None):
    """Bir eğitim klasöründen (config.json + agent.npz) konfigürasyon ve ajanı yükler."""
    run_dir = Path(run_dir)
    cfg = ExperimentConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    from .features import FeatureExtractor
    from .env import BinPickingEnv

    env = BinPickingEnv(cfg.env, cfg.gripper)
    fe = FeatureExtractor(cfg.env, cfg.gripper, cfg.features, env.wall_mask)
    agent = make_agent(cfg.agent.algo, fe.n_states, cfg.agent, seed)
    agent.load_arrays(run_dir / "agent.npz")
    return cfg, agent
