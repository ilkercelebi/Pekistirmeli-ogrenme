# Sınıf Sunumu Planı (~10 dk, 10–12 slayt)

1. **Başlık:** Ödev 8, isim, GitHub ve video linki (QR kod).
2. **Endüstriyel problem:** bin picking fotoğrafı; üç hata türü (çarpışma, başarısız kavrama, kayıp düşme).
3. **MDP tasarımı:** durum = yükseklik haritası + tutucu geometrisi; eylem = $(x, y, \theta)$; ödül tablosu.
4. **Neden klasik RL?** Ödevde önerilen yöntem FCN-DQN. Biz tablo tabanlı yöntemi seçtik: yorumlanabilir, GPU gerektirmiyor, tasarım kararları görünür.
5. **Simülasyon:** kutu, parça tipleri, yığın oluşumu, parmaklı ve vakumlu tutucu, sonuç kuralları (akış şeması).
6. **Durum kümeleme:** 7 özelliğin tablosu; "6400 aday eylem → 1296 tablo hücresi"; Q-haritası nasıl oluşur.
7. **Algoritma:** Q-learning ve SARSA güncelleme denklemleri, ε-açgözlü keşif, hiperparametreler.
8. **Canlı demo (2–3 dk):** `python serve.py` → http://localhost:8000
   - Problem sayfası: senaryo oynatıcı
   - Simülasyon: önce Sezgisel, sonra Q-learning ajanı, 3D katman "Q-haritası", "Kutuyu boşalt"
   - Manuel mod: sınıftan birine kavrama noktası seçtir
   - Karşılaştırma sayfası
9. **Öğrenme eğrileri:** `learning_curves.png`
10. **Test sonuçları:** karşılaştırma tablosu ve çubuk grafik.
11. **Ajan ne öğrendi?** En yüksek ve en düşük Q değerli durumlar; insan sezgisiyle karşılaştırma.
12. **Sınırlamalar ve sonuç.**

## Olası sorular ve cevaplar
- *"Bu gerçekten RL mi, yoksa sınıflandırma mı?"* — Bölümler ardışıktır: başarılı alım haritayı değiştirir, alttaki parçaları açığa çıkarır. Hedefte $\gamma > 0$ ile sonraki durumun değeri kullanılır (TD öğrenme). Ödül de etiket olarak değil, deneme-yanılma ile gelir.
- *"Özellikleri siz seçtiniz, öğrenme nerede?"* — Özellikler yalnızca ölçüm yapar ve hangi durumun iyi olduğunu söylemez. Hangi özellik birleşiminin iyi ya da kötü olduğunu ajan ödülden öğrenir. Sezgisel kural, aynı ölçümleri elle yazılmış bir mantıkla kullanır; Q-learning ise bu kuralı geçer.
- *"Neden DQN değil?"* — Ödev kapsamında klasik RL ile sınırlı kaldık. Tablo yöntemi yorumlanabilir ve hızlıdır. DQN'in avantajı özellik tasarımı gerektirmemesidir; bu bir gelecek çalışma olabilir.
- *"Gerçek robota nasıl aktarılır?"* — Derinlik kamerasından yükseklik haritası üretilir; aynı özellik çıkarıcı ve Q tablosu kullanılır. Simülasyon ile gerçek dünya arasındaki fark ince ayarla kapatılmalıdır.
