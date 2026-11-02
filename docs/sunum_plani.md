# Sınıf Sunumu Planı (~12 dk, 17 slayt)

Slaytlar hazır (konuşmacı notlarıyla). Akış:

1. **Kapak:** Ödev 8, ad, öğrenci no, ders.
2. **Endüstriyel problem:** üç hata türü (çarpışma −1, başarısız kavrama −0.2, kayıp düşme −0.2).
3. **MDP tasarımı:** durum = 44 × 44 yükseklik haritası + robot hafızası; eylem = $(x, y, \theta)$ + dur; ödül tablosu.
4. **Simülasyon:** 2.5B kutu, parmaklı (9 cm) ve vakumlu (1.5 cm) tutucu, sonuç kuralları.
5. **Neden tablo tabanlı?** Okunabilir, hafif, ölçülebilir.
6. **Durum–eylem kümeleme:** 6.400 aday → 38.880 (parmaklı) / 3.456 (vakum) hücre.
7. **Algoritmalar:** Q-learning, SARSA, Double Q-learning; ε, α, γ; dur kuralı.
8. **Eğitim düzeni:** 3.000 bölüm, 3 algoritma × 2 tutucu × 3 tohum = 18 eğitim, 200 görülmemiş test kutusu, eğitim başına ~8 dk (Apple M4, GPU yok).
9. **Ne kullandık?** Python · NumPy · SciPy (simülasyon ve öğrenme), Matplotlib · pytest (deney), FastAPI · React · Three.js (arayüz).
10. **Geliştirme süreci:** %85.3 → kâhin %96.6 → hizalama/yüzey/hafıza → %91.5.
11. **Sonuç · parmaklı:** %91.5 vs sezgisel %70.6 vs kâhin %96.6, sıfır çarpışma.
12. **Öğrenme eğrileri:** ~1.200. bölümde sezgiseli geçiyor; çarpışma %31 → %1.5.
13. **Sonuç · vakum:** SARSA %85.1 vs sezgisel + yüzey %70.2 vs kâhin %94.4.
14. **Ajan ne öğrendi?** Duvar tek negatif değer; seri başarısızlık; hizalama tuhaflığı.
15. **Ablasyon:** tek açı −25 puan, γ = 0.9 çöküşü ve Double Q.
16. **Canlı demo (2–3 dk):** `python serve.py` → http://localhost:8000
   - Problem sayfası: senaryo oynatıcı
   - Simülasyon: yeni kutu, Q-learning ajanı, "Q-haritası", "Kutuyu boşalt"
   - Manuel mod: sınıftan birine kavrama noktası seçtir
   - Karşılaştırma sayfası
17. **Sonuç, sınırlamalar, sorular.**

Demo yedeği: README'deki GIF'ler ve YouTube videosu.

## Olası sorular ve cevaplar
- *"Bu gerçekten RL mi, yoksa sınıflandırma mı?"* — Ajan doğru kavramayı etiketli örneklerden değil, kendi denemelerinin ödülünden öğreniyor (deneme-yanılma, keşif-sömürü dengesi, TD güncellemesi). Ama dürüst olmak gerekirse ablasyon, hafıza eklendikten sonra γ = 0'ın (tek adımlık, bandit benzeri) γ = 0.5 kadar iyi olduğunu gösteriyor. Bu problemde kararlar arasındaki bağ zayıf; iyi kavrama seçimi çoğunlukla tek adımlık bir karar. Bunu sınırlama olarak raporluyoruz.
- *"Özellikleri siz seçtiniz, öğrenme nerede?"* — Özellikler yalnızca ölçüm yapar ve hangi durumun iyi olduğunu söylemez. Hangi özellik birleşiminin iyi ya da kötü olduğunu ajan ödülden öğrenir. Sezgisel kural, aynı ölçümleri elle yazılmış bir mantıkla kullanır; Q-learning ise bu kuralı geçer.
- *"Neden DQN değil?"* — Ödev klasik pekiştirmeli öğrenme kapsamında; derin RL kullanmadık. Tablo yöntemi yorumlanabilir ve hızlıdır. DQN'in avantajı özellik tasarımı gerektirmemesidir; bu bir gelecek çalışma olabilir.
- *"Gerçek robota nasıl aktarılır?"* — Derinlik kamerasından yükseklik haritası üretilir; aynı özellik çıkarıcı ve Q tablosu kullanılır. Simülasyon ile gerçek dünya arasındaki fark ince ayarla kapatılmalıdır.
