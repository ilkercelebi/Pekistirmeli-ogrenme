# Ödev 8 — Düzensiz Kutulardan Parça Alma (Bin Picking) İçin Optimal Kavrama Seçimi

> Durum (2026-10-16): Tüm kod yazıldı, testler geçiyor. **Eğitimler henüz yapılmadı.**
> Teslim: YouTube videosu + GitHub linki + makale tarzı README + sınıfta sunum.

## Sıradaki adımlar
1. `python run_experiments.py --quick` ile hızlı dene, sonra `python run_experiments.py --ablation` (tam deney, ~15-30 dk).
   Eğitim, değerlendirme, grafikler, GIF'ler, `docs/img/` ve README bölüm 6 + özet otomatik dolar.
2. README'de kalan elle doldurulacak yerler: ad/numara, video linki, "Yorum" paragrafları, Sonuç bölümü
4. Arayüz: `cd web && npm install && npm run build`, sonra `python serve.py` → http://localhost:8000
5. Video: `docs/video_senaryosu.md` · Sunum: `docs/sunum_plani.md`
6. İsteğe bağlı: `--gripper vacuum` ile aynı deneyler

Not: Kısa bir kontrolde (400 bölüm) Q-learning, sezgisel yöntemi geçti (başarı oranı 0.85'e karşı 0.69). Bu bir kontroldü, asıl sonuç değil.

## Kararlar
- **Robot kol kullanılmayacak** → tamamen simülasyon.
- **Sadece klasik pekiştirmeli öğrenme** (tablo tabanlı Q-learning / SARSA vb.).
  **Derin pekiştirmeli öğrenme (DQN, sinir ağı) kullanılmayacak.**

## Ödevde istenenler (görselden)
1. **Endüstriyel problem:** Kutuya rastgele yığılmış parçalar, vakumlu veya parmaklı tutucu ile alınırken
   tutucu kutu kenarına çarpabilir ya da parça tutucudan kayıp düşebilir.
2. **MDP tasarımı:**
   - **Durum (State):** RGB-D kameradan alınan yükseklik haritası (heightmap) + tutucu ucunun geometrisi.
   - **Eylem (Action):** Görüntü koordinatlarında kavrama noktası `(x, y)` ve bilek döndürme açısı `θ`.
   - **Ödül (Reward):**
     - `+1` → parça kutuya çarpmadan ve düşmeden kaldırıldı
     - `-1` → tutucu kutuya çarptı
     - `-0.2` → kavrama başarısız

## Klasik RL'e uyarlama fikri (taslak, sonra netleşecek)
DQN olmadan büyük görüntü durumunu tabloya sığdırmak için **ayrıklaştırma** gerekecek:
- Yükseklik haritasını kaba bir ızgaraya böl (ör. 5×5 / 8×8 hücre).
- **Durum:** Her hücre için basit özellikler (ör. yükseklik seviyesi düşük/orta/yüksek, kenara yakın mı)
  → küçük, ayrık bir durum uzayı.
- **Eylem:** `(hücre, açı)` çifti; açı birkaç kovaya bölünür (ör. 0°, 45°, 90°, 135°).
- **Algoritma:** Tablo tabanlı Q-learning (ε-greedy keşif). Alternatif: SARSA, ya da durum basitse
  çok kollu bandit / contextual bandit yaklaşımı.
- **Ortam:** Basit 2D/ızgara tabanlı Python simülasyonu (ağır fizik motoru şart değil);
  istenirse PyBullet ile basit sahne.

## Değerlendirme metrikleri
- Başarı oranı (başarılı kavrama / toplam deneme)
- Bölüm başına toplam ödül eğrisi
- Çarpışma oranı
- (Ek) Başarısız kavrama oranı, öğrenme hızı

## Teknolojiler
| Bileşen | Teknoloji |
|---|---|
| Programlama | Python |
| RL | Tablo tabanlı Q-learning (NumPy) |
| Simülasyon | Kendi yazdığımız ızgara ortamı (opsiyonel: PyBullet) |
| Grafik / analiz | Matplotlib |

## Açık sorular
- Vakumlu mu parmaklı tutucu mu modellenecek?
- Ortam ne kadar gerçekçi olmalı (2D ızgara yeterli mi)?
