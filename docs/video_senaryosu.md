# YouTube Video Senaryosu (≈6 dk)

Kayıt: arayüzün ekran kaydı (`python serve.py` → http://localhost:8000) ve üstüne anlatım sesi.
Kayıttan önce `python run_experiments.py` (ya da en az bir Q-learning eğitimi) bitmiş olmalı; arayüz `results/` klasöründeki modelleri yükler.

Sayılar README'deki son deneyden (3 tohum × 3.000 bölüm, eğitimde görülmemiş 200 test kutusu, keşif kapalı).

## Akış

| Süre | Ekranda | Ne gösterilecek |
|---|---|---|
| 0:00–0:20 | **Problem** sayfasının başlığı | Başlık, ödev adı |
| 0:20–1:00 | **Problem** · senaryo oynatıcı (1 → 4) | Doğru kavrama, duvara çarpma, yanlış açı (kayma), üstü kapalı parça |
| 1:00–1:40 | **MDP tasarımı** | Gerçek sahne ↔ kameranın gördüğü harita; dört bilek açısı; ödül kartları; dur eylemi |
| 1:40–2:30 | **Yöntem** | 6.400 aday → 38.880 hücre; özellik haritaları; bir piksele tıklayıp φ(s,a); Q-learning / SARSA / Double Q formülleri |
| 2:30–3:20 | **Eğitim ve sonuçlar** sayfasının üstü (yeni eğitim formu) + README “Deney Düzeni” ve “Proje Yapısı” | Nasıl eğittik, hangi araçları kullandık |
| 3:20–4:00 | **Eğitim ve sonuçlar** · kayıtlı eğitim | Öğrenme eğrisi; “Ajan ne öğrendi?” çubukları |
| 4:00–4:50 | **Simülasyon** · Q-learning ajanı, kutu tohumu 99, “Q-haritası” katmanı, “Kutuyu boşalt” | Yeni (görülmemiş) kutu; ajan kutuyu boşaltıyor, parçalar tepsiye diziliyor |
| 4:50–5:10 | **Simülasyon** · manuel mod | Duvar dibine tıkla → çarpışma, kırmızı ışık; iyi noktaya tıkla → başarı |
| 5:10–5:35 | **Karşılaştırma** | Rastgele, sezgisel ve ajanlar aynı test kutularında |
| 5:35–5:50 | README / GitHub sayfası | Sonuç, sınırlamalar, link |

## Anlatım metni

**0:00 — Giriş.**
Merhaba. Bu videoda “Düzensiz Kutulardan Parça Alma (Bin Picking) İçin Optimal Kavrama Seçimi” projemizi anlatıyorum: bin picking, yani bir robotun kutuya rastgele dökülmüş parçaları tek tek alması. Robot her seferinde şuna karar vermeli: parçayı hangi noktadan ve hangi açıyla tutacak? Bu problemi derin öğrenme kullanmadan, klasik, tablo tabanlı pekiştirmeli öğrenmeyle çözdük.

**0:20 — Problem senaryoları.**
Robot hücresinde bir kutu, üstte bir gantry robot ve derinlik kamerası var. İlk senaryo doğru bir kavrama: parmaklar parçanın kısa kenarından sıkıyor ve parça tepsiye gidiyor. İkincisinde parça duvarın dibinde; bu açıyla parmak duvarın üstüne iniyor: çarpışma, en pahalı hata. Üçüncüsünde parça tutuluyor ama yanlış açıyla tutulduğu için kalkarken kayıyor. Dördüncüsünde parçanın üstünde başka bir parça var; robot onu çekip çıkaramıyor.

**1:00 — MDP.**
Bunu bir Markov karar süreci olarak kurduk. Durum, kameranın gördüğü 44'e 44'lük yükseklik haritası ve robotun kendi başarısızlık hafızası. Eylem, haritadaki bir nokta ve dört bilek açısından biri; adım başına 6.400 aday. Ödül ödevdeki gibi: başarılı alım +1, çarpışma −1, diğer başarısızlıklar −0.2. Bir de “dur” eylemi ekledik: kutuda alınabilir parça kalmadıysa robot boşa denemek yerine durabiliyor.

**1:40 — Yöntem.**
Ham görüntü tabloya sığmaz. O yüzden her aday kavramayı birkaç soruya indiriyoruz: nokta yüksek mi, yüzey düz mü, yanında daha yüksek parça var mı, duvara yakın mı, parmakların ineceği yer boş mu, parça açıklığa sığıyor mu, parmaklar parçanın hangi kenarıyla hizalı, parçanın üstü kapalı mı, burası daha önce denendi mi. Bu cevapların her kombinasyonu bir tablo hücresi: parmaklı tutucu için 38.880, vakum için 3.456 hücre. Bir piksele tıklayınca hangi hücreye düştüğünü görüyoruz. Tabloyu üç klasik algoritmayla öğreniyoruz: Q-learning, SARSA ve Double Q-learning.

**2:30 — Nasıl eğittik, ne kullandık?**
Her şeyi Python'da kendimiz yazdık. Simülasyon, özellik çıkarıcı ve öğrenme algoritmaları NumPy ve SciPy ile çalışıyor; PyTorch ya da herhangi bir sinir ağı kütüphanesi yok. Eğitimde ajan her bölümde rastgele üretilmiş yeni bir kutuyla karşılaşıyor ve kutuyu boşaltmaya çalışıyor. Bir eğitim 3.000 bölüm, yani 3.000 farklı kutu. Başta tamamen rastgele deniyor; keşif oranı ε eğitimin ilk yarısında 1'den 0.05'e iniyor. Her algoritmayı ve her tutucuyu üç farklı rastgele tohumla eğittik ki sonuç şansa bağlı olmasın. Test için eğitimde hiç görülmemiş 200 sabit kutu kullandık ve testte keşfi kapattık; bütün yöntemler aynı kutularda ölçüldü. Bir eğitim Apple M4 dizüstü bilgisayarda yaklaşık 8 dakika sürüyor; 18 eğitim ve ablasyonların tamamı paralel çalışarak 23 dakikada bitiyor. Eğitilmiş Q tablosu sadece 600 KB'lık bir dosya. Tüm deney tek komutla yeniden üretilebiliyor: `python run_experiments.py --ablation`. Gördüğünüz arayüz de bizim: arka uçta FastAPI, ön yüzde React ve Three.js; 3D sahnedeki her kavrama Python simülasyonunun gerçek çıktısı.

**3:20 — Eğitim sonuçları.**
Başta ajan rastgele davranıyor ve kavramaların üçte biri duvara çarpıyor. Keşif azaldıkça başarı artıyor; yaklaşık 1.200. bölümde elle yazdığımız sezgisel kuralı geçiyor. Tablonun en güzel tarafı, ajanın öğrendiğini okuyabilmek: “duvar = çarpar” tablodaki tek negatif değer. Bunu kimse söylemedi, ajan kendisi buldu. Diğer sıralamalar da sezgiyle uyumlu: yüksek nokta iyi, boş parmak bölgesi iyi, üstü kapalı yüzey kötü.

**4:00 — Simülasyon.**
Şimdi ajana hiç görmediği bir kutu veriyorum. Q-haritasını açınca yeşil noktalar ajanın iyi bulduğu kavramalar. “Kutuyu boşalt” diyorum: ajan parçaları tek tek alıp tepsiye diziyor. Alınabilir parça kalmayınca duruyor.

**4:50 — Manuel mod.**
Elle duvarın dibine tıklarsam robot duvara çarpıyor ve kırmızı ışık yanıyor. Ajanın seçtiği türden bir noktaya tıklarsam alım başarılı.

**5:10 — Karşılaştırma.**
Aynı test kutularında: rastgele politika yüzde 3, sezgisel kural yüzde 70.6, Q-learning yüzde 91.5 başarı. Çarpışma sıfır. Simülasyonun gizli bilgisini gören bir “kâhin” kural yüzde 96.6'ya çıkıyor; yani ajan ulaşılabilecek tavanın yüzde 95'inde. Daha zor olan vakumlu tutucuda SARSA yüzde 85.1, sezgisel yüzde 70.2.

**5:35 — Kapanış.**
Sınırlamalarımız: simülasyon basit ve kurallı, özellikleri elle tasarladık ve deneyler kararların büyük ölçüde tek adımlık olduğunu gösteriyor. Kod, makale tarzı README ve tüm deneyler GitHub'da; link açıklamada. İzlediğiniz için teşekkürler.

## Kayıt ipuçları
- Tarayıcıyı tam ekran yap. 3D sahnede sürükleyerek kamerayı çevirebilir, “Hücre / Yakın / Üstten” görünümlerine geçebilirsin.
- Simülasyon sayfasında animasyon hızını 1× ya da 2× tut; 4× video için hızlı kalır.
- Q-learning'in sezgiselden açıkça iyi olduğu bir kutu tohumu seç (birkaç tohum dene).
- Sesi ayrı kaydetmek daha kolay: önce ekranı kaydet, sonra anlatımı üstüne oku.
- Videoyu “Liste dışı” (unlisted) yükleyip linki README'nin en üstündeki `<YOUTUBE_LINKI>` yerine yaz.
