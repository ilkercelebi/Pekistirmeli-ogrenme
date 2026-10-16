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
    obs_noise: float = 0.05          # derinlik kamerası gürültüsü (cm, std)

    # Ödev metnindeki ödül fonksiyonu
    reward_success: float = 1.0
    reward_collision: float = -1.0
    reward_fail: float = -0.2

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
    cup_radius: float = 2.0
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
        return cls(
            env=_from_dict(EnvConfig, data.get("env")),
            gripper=_from_dict(GripperConfig, data.get("gripper")),
            features=_from_dict(FeatureConfig, data.get("features")),
            agent=_from_dict(AgentConfig, data.get("agent")),
        )
