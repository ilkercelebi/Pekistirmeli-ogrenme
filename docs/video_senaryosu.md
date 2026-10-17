# YouTube Video Senaryosu (4–5 dk)

Kayıt: arayüzün ekran kaydı (`python serve.py` → http://localhost:8000) ve üstüne anlatım sesi.
Kayıttan önce en az bir Q-learning eğitimi bitmiş olmalı (komut satırından ya da "Eğitim ve sonuçlar" sayfasından).

| Süre | Ekranda | Anlatım |
|---|---|---|
| 0:00–0:20 | **Problem** sayfasının başlığı | Ödev 8: düzensiz kutudan parça alma için optimal kavrama seçimi. Klasik, tablo tabanlı pekiştirmeli öğrenme. |
| 0:20–1:00 | **Problem** · senaryo oynatıcı (1→4) | Robot hücresi: kutu, gantry robot, derinlik kamerası. Dört senaryo: doğru kavrama, duvara çarpma, yanlış açı, üstü kapalı parça. |
| 1:00–1:40 | **MDP tasarımı** | Gerçek sahne ve kameranın gördüğü harita yan yana; dört bilek açısı; ödül kartları (+1 / −1 / −0.2). |
| 1:40–2:30 | **Yöntem** | Neden derin RL değil; 6.400 aday eylem → 15.552 tablo hücresi (10 özellik); özellik haritaları; bir piksele tıklayıp φ(s,a) hesabını göster; Q-learning formülü. |
| 2:30–3:00 | **Eğitim ve sonuçlar** | Öğrenme eğrisi: ε azaldıkça başarı artıyor, çarpışma sıfıra iniyor. "Ajan ne öğrendi?": duvar = çarpar kırmızı, genişlik = sığar yeşil. |
| 3:00–3:50 | **Simülasyon** · Q-learning ajanı, 3D katman "Q-haritası", "Kutuyu boşalt" | Q-haritası parçaların üstünde: yeşil iyi noktalar. Ajan kutuyu boşaltıyor, parçalar tepsiye diziliyor. |
| 3:50–4:15 | **Simülasyon** · manuel mod | Duvar dibine tıkla: çarpışma ve kırmızı ışık. Doğru noktaya tıkla: başarı. |
| 4:15–4:40 | **Karşılaştırma** | Rastgele, sezgisel ve Q-learning aynı test kutularında: grafik ve tablo. |
| 4:40–5:00 | Problem sayfası + GitHub linki | Sınırlamalar (2.5B simülasyon, durum kümeleme) ve gelecek çalışma. |

## Kayıt ipuçları
- Tarayıcıyı tam ekran yap. 3D sahnede sürükleyerek kamerayı çevirebilir, "Hücre / Yakın / Üstten" görünümlerine geçebilirsin.
- Simülasyon sayfasında animasyon hızını 1× ya da 2× tut; 4× video için hızlı kalır.
- Sezgisel yöntem ile Q-learning'in farkının açıkça görüldüğü bir kutu tohumu seç (birkaç tohum dene).
- Videoyu "Liste dışı" (unlisted) yükleyip linki README'nin en üstüne ekle.
