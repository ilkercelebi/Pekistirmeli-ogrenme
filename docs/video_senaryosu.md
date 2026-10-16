# YouTube Video Senaryosu (4–5 dk)

Kayıt: ekran kaydı (Streamlit arayüzü + grafikler) ve üstüne anlatım sesi.
Kayıttan önce: eğitim bitmiş, `evaluate.py`, `plots.py` ve `make_gif.py` çalıştırılmış olmalı.

| Süre | Ekranda | Anlatım |
|---|---|---|
| 0:00–0:20 | Başlık slaydı | Ödev 8: düzensiz kutudan parça alma için optimal kavrama seçimi. Klasik, tablo tabanlı pekiştirmeli öğrenme. |
| 0:20–0:50 | Gerçek bin picking fotoğrafı/çizimi | Problem: rastgele yığılmış parçalar; tutucu duvara çarpabilir, parça kayabilir. Karar: hangi nokta, hangi açı? |
| 0:50–1:30 | Arayüz, "Canlı demo", Q-haritası kapalı | Simülasyon: yükseklik haritası, renkler yükseklik. Parmaklı tutucu, 4 açı. MDP: durum / eylem / ödül (+1, −1, −0.2). |
| 1:30–2:00 | Yöntem: "Rastgele", "1 adım" birkaç kez | Rastgele ajan: duvara çarpıyor (kırmızı), boşa kavrıyor (turuncu). |
| 2:00–2:45 | Yöntem slaydı (özellik tablosu) | Neden derin RL değil; görüntüyü 7 ayrık özelliğe indirgeme; 1296 hücrelik Q tablosu; Q-learning güncelleme kuralı. |
| 2:45–3:15 | "Eğitim sonuçları" sekmesi, öğrenme eğrisi | ε azaldıkça başarı artıyor, çarpışma sıfıra iniyor; sezgisel çizgiyi geçtiği nokta. |
| 3:15–3:50 | "Canlı demo", Q-learning ajanı, Q-haritası açık, "Kutuyu boşalt" | Yeşil/kırmızı Q-haritası: duvar kenarları kırmızı, parça ortaları yeşil. Ajan kutuyu boşaltıyor. |
| 3:50–4:15 | "Yan yana": Sezgisel vs Q-learning, aynı tohum | Aynı kutuda karşılaştırma; adım sayısı ve başarısız deneme farkı. |
| 4:15–4:40 | Test tablosu ve "en iyi/en kötü durumlar" tablosu | Sayısal sonuçlar; tablonun yorumlanabilirliği (ajanın ne öğrendiği okunabiliyor). |
| 4:40–5:00 | Sonuç slaydı + GitHub linki | Sınırlamalar (2.5B simülasyon, kümeleme) ve gelecek çalışma. |

## Kayıt ipuçları
- Kutu tohumunu önceden seç: Q-learning ile sezgisel yöntemin farkının açıkça görüldüğü bir tohum bul (arayüzde "Yan yana" sekmesinde birkaç tohum dene).
- Tarayıcıyı tam ekran yap, kenar çubuğunu daralt.
- Videoyu "Liste dışı" (unlisted) yükleyip linki README'nin en üstüne ekle.
