# 🎬 Video Senaryosu

**Süre:** yaklaşık 6 dakika
**Kayıt:** OBS ile yan ekran (MAG 272) · Safari tam ekran · `http://localhost:8000`

Her bölümde önce **🖱️ Ekranda** yapacaklarını, sonra **🎙️ Söyle** kısmında okuyacaklarını bulursun.
Metni kelimesi kelimesine okumak zorunda değilsin; kendi cümlelerinle anlatman daha doğal olur.

---

## ✅ Kayıttan önce

- [ ] Sunucu açık: `python serve.py`
- [ ] Safari tam ekran (Ctrl+Cmd+F), sık kullanılanlar çubuğu gizli (Cmd+Shift+B)
- [ ] Rahatsız Etme modu açık
- [ ] OBS'de mikrofon (Mic/Aux) açık, bilgisayar sesi kapalı
- [ ] 10 saniyelik deneme kaydı alındı, ses ve görüntü kontrol edildi
- [ ] Simülasyon ayarları biliniyor: **Q-learning · tohum 0** ve **kutu tohumu 99**

---

## 🗺️ Akış

| # | Süre | Sayfa | Konu |
|:-:|:-:|---|---|
| 1 | 0:00 | Açılış → **Problem** | Giriş |
| 2 | 0:20 | **Problem** · senaryolar | Ne ters gidebilir? |
| 3 | 1:00 | **MDP tasarımı** | Durum, eylem, ödül |
| 4 | 1:40 | **Yöntem** | Tablo ve algoritmalar |
| 5 | 2:30 | **Eğitim ve sonuçlar** (üst kısım) | Nasıl eğittik, ne kullandık |
| 6 | 3:20 | **Eğitim ve sonuçlar** (grafikler) | Ajan ne öğrendi |
| 7 | 4:00 | **Simülasyon** | Ajan kutuyu boşaltıyor |
| 8 | 4:50 | **Simülasyon** · manuel | Çarpışma ve başarı |
| 9 | 5:10 | **Karşılaştırma** | Sayılar |
| 10 | 5:35 | README / GitHub | Kapanış |

> 💡 Her bölümü ayrı kaydedebilirsin. Hata yaparsan sadece o parçayı tekrar çekersin, sonra iMovie'de birleştirirsin.

---

## 1 · Giriş  `0:00 – 0:20`

**🖱️ Ekranda**
- Sayfayı **Cmd+R** ile yenile.
- Robot kollu açılış oynar, kapaklar açılır ve **Problem** sayfası gelir.

**🎙️ Söyle**

> Merhaba.
> Bu videoda **“Düzensiz Kutulardan Parça Alma (Bin Picking) İçin Optimal Kavrama Seçimi”** projemizi anlatıyorum.
>
> Bin picking, bir robotun kutuya rastgele dökülmüş parçaları tek tek almasıdır.
> Robot her seferinde şuna karar vermeli: **parçayı hangi noktadan ve hangi açıyla tutacak?**
>
> Bu problemi derin öğrenme kullanmadan, **klasik, tablo tabanlı pekiştirmeli öğrenmeyle** çözdük.

---

## 2 · Problem senaryoları  `0:20 – 1:00`

**🖱️ Ekranda**
- Senaryo oynatıcıda sırayla **1 → 2 → 3 → 4**'e tıkla.
- Her birinde animasyonun bitmesini bekle.

**🎙️ Söyle**

> Robot hücresinde bir kutu, üstte bir robot ve bir derinlik kamerası var.
>
> **1 — Doğru kavrama.** Parmaklar parçayı kısa kenarından sıkıyor, parça tepsiye gidiyor.
>
> **2 — Duvara çarpma.** Parça duvarın dibinde; bu açıyla parmak duvarın üstüne iniyor.
> Bu en pahalı hata.
>
> **3 — Yanlış açı.** Parça tutuluyor ama yanlış açıyla tutulduğu için kalkarken kayıyor.
>
> **4 — Üstü kapalı.** Parçanın üstünde başka bir parça var; robot onu çekip çıkaramıyor.

---

## 3 · MDP tasarımı  `1:00 – 1:40`

**🖱️ Ekranda**
- **MDP tasarımı** sayfasına geç.
- Gerçek sahne ↔ kameranın gördüğü harita, dört bilek açısı, ödül kartları ve “dur” eylemi.

**🎙️ Söyle**

> Bunu bir **Markov karar süreci** olarak kurduk.
>
> **Durum:** kameranın gördüğü 44 × 44'lük yükseklik haritası ve robotun kendi başarısızlık hafızası.
>
> **Eylem:** haritadaki bir nokta ve dört bilek açısından biri. Adım başına **6.400 aday**.
>
> **Ödül:**
> başarılı alım **+1**,
> çarpışma **−1**,
> diğer başarısızlıklar **−0.2**.
>
> Bir de **“dur” eylemi** ekledik: alınabilir parça kalmadıysa robot boşa denemek yerine durabiliyor.

---

## 4 · Yöntem  `1:40 – 2:30`

**🖱️ Ekranda**
- **Yöntem** sayfasına geç.
- Özellik haritalarını göster.
- Haritada **bir piksele tıkla**, düştüğü tablo hücresini göster.
- Aşağı kaydırıp **Q-learning / SARSA / Double Q** formüllerini göster.

**🎙️ Söyle**

> Ham görüntü bir tabloya sığmaz.
> Bu yüzden her aday kavramayı birkaç basit soruya indiriyoruz:
>
> - Nokta yüksek mi?
> - Yüzey düz mü?
> - Yanında daha yüksek parça var mı?
> - Duvara yakın mı?
> - Parmakların ineceği yer boş mu?
> - Parça açıklığa sığıyor mu?
> - Parmaklar parçanın hangi kenarıyla hizalı?
> - Parçanın üstü kapalı mı?
> - Burası daha önce denendi mi?
>
> Bu cevapların her kombinasyonu bir tablo hücresi:
> parmaklı tutucu için **38.880**, vakum için **3.456** hücre.
>
> Bir piksele tıklayınca hangi hücreye düştüğünü görüyoruz.
>
> Tabloyu üç klasik algoritmayla öğreniyoruz: **Q-learning, SARSA ve Double Q-learning.**

---

## 5 · Nasıl eğittik, ne kullandık?  `2:30 – 3:20`

**🖱️ Ekranda**
- **Eğitim ve sonuçlar** sayfasının üstündeki eğitim formunu göster.
- İstersen README'deki “Deney Düzeni” bölümüne kısaca geç.

**🎙️ Söyle**

> Her şeyi **Python**'da kendimiz yazdık.
> Simülasyon, özellik çıkarıcı ve algoritmalar **NumPy** ve **SciPy** ile çalışıyor.
> PyTorch ya da herhangi bir sinir ağı kütüphanesi yok.
>
> **Eğitim:**
> Ajan her bölümde rastgele üretilmiş yeni bir kutuyu boşaltmaya çalışıyor.
> Bir eğitim **3.000 bölüm**, yani 3.000 farklı kutu.
> Başta tamamen rastgele deniyor; keşif oranı ε, eğitimin ilk yarısında **1'den 0.05'e** iniyor.
>
> **Güvenilirlik:**
> Her algoritmayı ve her tutucuyu **üç farklı tohumla** eğittik ki sonuç şansa bağlı olmasın.
>
> **Test:**
> Eğitimde hiç görülmemiş **200 sabit kutu** kullandık ve testte keşfi kapattık.
> Bütün yöntemler aynı kutularda ölçüldü.
>
> **Süre:**
> Bir eğitim Apple M4 dizüstünde yaklaşık **8 dakika**.
> 18 eğitim ve ablasyonların hepsi paralel çalışarak **23 dakikada** bitiyor.
> Eğitilmiş Q tablosu sadece **600 KB**.
>
> Tüm deney tek komutla yeniden üretilebiliyor: `python run_experiments.py --ablation`
>
> **Arayüz:**
> Arka uçta **FastAPI**, ön yüzde **React** ve **Three.js**.
> 3D sahnedeki her kavrama, Python simülasyonunun gerçek çıktısı.

---

## 6 · Eğitim sonuçları  `3:20 – 4:00`

**🖱️ Ekranda**
- Aynı sayfada kayıtlı bir Q-learning eğitimini seç.
- **Öğrenme eğrisini** göster.
- **“Ajan ne öğrendi?”** çubuklarına in.

**🎙️ Söyle**

> Başta ajan rastgele davranıyor; kavramaların **üçte biri duvara çarpıyor**.
>
> Keşif azaldıkça başarı artıyor.
> Yaklaşık **1.200. bölümde** elle yazdığımız sezgisel kuralı geçiyor.
>
> Tablonun güzel tarafı, ajanın öğrendiğini **okuyabilmek**.
> **“Duvar = çarpar”** tablodaki tek negatif değer.
> Bunu kimse söylemedi; ajan kendisi buldu.
>
> Diğer sıralamalar da sezgiyle uyumlu:
> yüksek nokta iyi, boş parmak bölgesi iyi, üstü kapalı yüzey kötü.

---

## 7 · Simülasyon  `4:00 – 4:50`

**🖱️ Ekranda**
1. **Simülasyon** sayfasına geç.
2. Yöntem: **Q-learning · tohum 0**
3. Kutu tohumu: **99**
4. **Q-haritası** katmanını aç.
5. Animasyon hızı: **1×** ya da **2×**
6. **“Kutuyu boşalt”**a bas.

**🎙️ Söyle**

> Şimdi ajana **hiç görmediği** bir kutu veriyorum.
>
> Q-haritasında yeşil noktalar, ajanın iyi bulduğu kavramalar.
>
> “Kutuyu boşalt” diyorum…
> Ajan parçaları tek tek alıp tepsiye diziyor.
>
> Alınabilir parça kalmayınca **kendisi duruyor**.

---

## 8 · Manuel mod  `4:50 – 5:10`

**🖱️ Ekranda**
- **Manuel** moda geç.
- Önce **duvarın dibindeki** bir noktaya tıkla → çarpışma, kırmızı ışık.
- Sonra Q-haritasında **yeşil** bir noktaya tıkla → başarılı alım.

**🎙️ Söyle**

> Elle duvarın dibine tıklarsam robot **duvara çarpıyor**, kırmızı ışık yanıyor.
>
> Ajanın seçtiği türden bir noktaya tıklarsam alım **başarılı**.

---

## 9 · Karşılaştırma  `5:10 – 5:35`

**🖱️ Ekranda**
- **Karşılaştırma** sayfasına geç.
- Çubukların üzerinde fareyle yavaşça gez.

**🎙️ Söyle**

> Aynı 200 test kutusunda sonuçlar:

| Yöntem | Başarı |
|---|:-:|
| Rastgele | %3 |
| Sezgisel kural | %70.6 |
| **Q-learning** | **%91.5** · çarpışma **0** |
| “Kâhin” (gizli bilgiyi gören tavan) | %96.6 |

> Yani ajan, ulaşılabilecek tavanın **yüzde 95'inde**.
>
> Daha zor olan **vakumlu** tutucuda: SARSA **%85.1**, sezgisel **%70.2**.

---

## 10 · Kapanış  `5:35 – 5:50`

**🖱️ Ekranda**
- GitHub'daki README sayfasını göster.

**🎙️ Söyle**

> Sınırlamalarımız:
> simülasyon basit ve kurallı,
> özellikleri elle tasarladık,
> ve kararlar büyük ölçüde tek adımlık.
>
> Kod, makale tarzı README ve tüm deneyler GitHub'da; link açıklamada.
>
> **İzlediğiniz için teşekkürler.**

---

## 🎥 Kayıt ipuçları

- Fareyi **yavaş** hareket ettir; tıklamadan önce bir saniye bekle.
- 3D sahnede sürükleyerek kamerayı çevirebilir, **Hücre / Yakın / Üstten** görünümlerine geçebilirsin.
- Simülasyonda **4×** hız video için fazla hızlı kalır.
- Fareyi ekranın en üstüne götürme; Safari'nin araç çubuğu açılır.
- Yükleme: YouTube'a **“Liste dışı”** yükle, linki README'nin başındaki `<YOUTUBE_LINKI>` yerine yaz.
