"""Tüm deney parametreleri.

Birimler: ızgaranın bir hücresi 1 cm x 1 cm, yükseklikler cm cinsindendir.
Konfigürasyonlar JSON'a yazılıp okunabilir; böylece her eğitim koşusu
hangi parametrelerle yapıldığını kendi klasöründe saklar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class EnvConfig:
    """Kutu ve parça üretimi."""

    interior_size: int = 40          # kutunun iç ölçüsü (hücre)
    wall_thickness: int = 2          # duvar kalınlığı (hücre)
    wall_height: float = 20.0        # duvar yüksekliği (cm)

    n_parts_min: int = 6
    n_parts_max: int = 14
    box_prob: float = 0.7            # parça kutu (dikdörtgen) mu disk mi
    # Bazı parçalar tutucu açıklığına (9 cm) yakın genişlikte: yanlış açı/nokta seçimi cezalanır.
    box_length: tuple = (4.0, 14.0)
    box_width: tuple = (2.0, 8.0)
    disc_diameter: tuple = (3.0, 7.0)
    thickness: tuple = (2.0, 5.0)
    pile_sigma: float = 8.0          # parçalar kutu merkezinde yığın oluştursun diye
    max_stack_height: float = 15.0

    max_steps_per_part: float = 2.5  # bölüm uzunluğu = ceil(parça sayısı * bu değer)
    memory_radius: float = 2.0      # başarısız kavrama hafızasının yarıçapı (cm)
    obs_noise: float = 0.05          # derinlik kamerası gürültüsü (cm, std)

    # Ödev metnindeki ödül fonksiyonu
    reward_success: float = 1.0
    reward_collision: float = -1.0
    reward_fail: float = -0.2
    # "Dur" eylemi: robot kutuda alınabilir parça kalmadığına karar verip bölümü bitirir (ör. kutuyu
    # sallatmak ya da operatöre bildirmek için). Ödülü 0'dır; boşa denemelerin cezasından kaçınır.
    allow_stop: bool = True
    reward_stop: float = 0.0

    @property
    def size(self) -> int:
        return self.interior_size + 2 * self.wall_thickness


@dataclass
class GripperConfig:
    """Tutucu geometrisi ve kavrama kalitesi modeli."""

    kind: str = "parallel_jaw"       # "parallel_jaw" (parmaklı) veya "vacuum" (vakumlu)
    n_angles: int = 4                # bilek açısı θ için kova sayısı, [0, 180) aralığında

    # Parmaklı tutucu
    opening: float = 9.0             # parmaklar arası en büyük açıklık (cm)
    finger_thickness: float = 1.0
    finger_width: float = 3.0
    grasp_depth: float = 1.5         # parmakların parça üst yüzeyinden ne kadar aşağı ineceği

    # Vakumlu tutucu
    cup_radius: float = 1.5          # küçük parçalar için küçük vantuz (2 cm dar parçalara oturmuyor)
    min_seal: float = 0.85           # vakumun tutması için gereken en az sızdırmazlık oranı

    surface_tol: float = 0.6         # "aynı yüzey" sayılan yükseklik farkı (cm)

    # Kayıp düşme (slip) olasılığı modeli
    slip_base: float = 0.03
    slip_misalign: float = 0.5       # parmak yönü ile parçanın kısa ekseni arasındaki açı hatası
    slip_offcenter: float = 0.25     # parçanın ağırlık merkezinden uzak tutma
    slip_seal: float = 0.3           # vakumda zayıf sızdırmazlık

    @property
    def angle_count(self) -> int:
        # Dairesel vantuzda bilek açısının etkisi yoktur.
        return 1 if self.kind == "vacuum" else self.n_angles


@dataclass
class FeatureConfig:
    """Yükseklik haritasından ayrık durum özelliklerinin çıkarılması."""

    height_bins: tuple = (0.5, 4.0, 8.0)   # zemin / alçak / orta / yüksek
    flat_bins: tuple = (0.3, 1.0)          # yerel standart sapma: düz / hafif / pürüzlü
    neighbor_radius: int = 3
    neighbor_thr: float = 1.5              # yakında daha yüksek bir şey var mı
    face_ring: int = 2                     # vakum: düz yüzeyin çevresinde daha yüksek yüzey aranan genişlik
    axis_reach: int = 16                   # dik doğrultuda yüzey ölçümünün menzili (uzun parçalar 14 cm)
    axis_center_bins: tuple = (0.1, 0.25, 0.45, 0.7)  # eksen_ortası: tam ortada / ortada / kaymış / uca yakın / uçta
    align_bins: tuple = (0.75, 1.0)        # hizalama: yüzey oranı ≤0.75 kısa kenardan, ≤1 kısmen, >1 uzun kenardan
    jaw_face_feature: bool = True          # parmaklı: `yüzey_kapalı` (vakumdaki yüzey bölütleme özelliği)
    # `seri_başarısız`: üst üste başarısız deneme sayısı (0, 1, 2, 3+). Vakumda durmayı kararlı öğrenmeyi sağlar;
    # parmaklıda tabloyu 4 kat büyütüp biraz zarar verdiği için yalnız vakumda açık.
    streak_grippers: tuple = ("vacuum",)
    streak_bins: tuple = (1, 2, 3)
    memory_feature: bool = True            # `denendi`: bu bölgede bu bölüm başarısız olundu mu (robotun hafızası)
    vacuum_center_feature: bool = False    # vakum: `eksen_ortası` (uzun eksende ortalanma)
    alignment_features: bool = True        # parmaklı: `hizalama` + `eksen_ortası` (False → eski `ortalı`, 1296 hücre)
    face_feature: bool = True              # vakum: `yüzey_kapalı` tabloya girsin mi (False → 216 hücre, ablasyon)
    wall_margin: float = 2.0               # duvara "yakın" sayılan mesafe
    clear_bins: tuple = (0.0, 1.0)         # parmak bölgesi boşluğu: boş / sınırda / dolu
    seal_bins: tuple = (0.95, 0.85)        # vakum: iyi / orta / kötü
    candidate_stride: int = 1              # aday kavrama noktalarının ızgara adımı


@dataclass
class AgentConfig:
    """Tablo tabanlı ajanın hiperparametreleri."""

    algo: str = "q_learning"         # "q_learning" veya "sarsa"
    alpha: float = 0.1
    alpha_mode: str = "visit"        # "constant" veya "visit" (1 / N^omega)
    alpha_omega: float = 0.6
    alpha_min: float = 0.01
    gamma: float = 0.5
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 1500
    q_init: float = 0.0
    # Açgözlü seçimde ve Q-learning hedefinde hiç ziyaret edilmemiş tablo hücreleri (Q = başlangıç değeri)
    # "en iyi" sayılmasın. Keşif sırasında iyimser başlangıç değeri yine keşfi teşvik eder.
    stop_margin: float = 0.0         # en iyi kavramanın Q değeri bunun altındaysa dur (dur eylemi açıksa)
    avoid_unvisited: bool = False    # denendi: eğitimde iyimser keşfi kesip başarıyı düşürdü (%85 → %79)


def _from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    kwargs = {}
    for key, value in (data or {}).items():
        if key in names:
            kwargs[key] = tuple(value) if isinstance(value, list) else value
    return cls(**kwargs)


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_dict(self) -> dict:
        return {
            "env": asdict(self.env),
            "gripper": asdict(self.gripper),
            "features": asdict(self.features),
            "agent": asdict(self.agent),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        feats = dict(data.get("features") or {})
        # Bu ayar eklenmeden önce kaydedilmiş eğitimler eski özellik setiyle eğitildi.
        feats.setdefault("alignment_features", False)
        feats.setdefault("align_bins", (1.0,))
        feats.setdefault("axis_center_bins", (0.25, 0.55))
        env_cfg = dict(data.get("env") or {})
        env_cfg.setdefault("allow_stop", False)
        data = {**data, "env": env_cfg}
        feats.setdefault("jaw_face_feature", False)
        feats.setdefault("memory_feature", False)
        feats.setdefault("streak_grippers", ())
        data = {**data, "features": feats}
        return cls(
            env=_from_dict(EnvConfig, data.get("env")),
            gripper=_from_dict(GripperConfig, data.get("gripper")),
            features=_from_dict(FeatureConfig, data.get("features")),
            agent=_from_dict(AgentConfig, data.get("agent")),
        )
