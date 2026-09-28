# Yüz Lekesi.5 — Morfolojik Kapama Tabanlı Arka Plan Tahmini

## Kullanıcının bildirdiği sorun (ekran görüntüsüyle)

Yoğun çilli/benli bir yüzde, Yüz Lekesi.4 gözle görülür pek çok lekeyi **kaçırıyordu** — "40 tespit" derken fotoğrafta bariz şekilde çok daha fazla leke vardı. Bu bir yanlış-pozitif değil, bir **recall** (kaçırma) sorunuydu.

## Kök neden

Yüz Lekesi.4, "yerel ortalama"yı **Gauss bulanıklaştırma** ile hesaplıyordu (komşu piksellerin ağırlıklı ortalaması). Ama yoğun çilli bir bölgede, bir pikselin yakın çevresi zaten **başka çillerle dolu** — yani "yerel ortalama"nın kendisi de bu çillerden dolayı aşağı çekiliyor. Sonuç: bir çil, çevresindeki diğer çillerle kıyaslandığında artık o kadar "farklı/koyu" görünmüyor — **benekler birbirini maskeliyor**. Bu, ortalama-tabanlı her yöntemin (Yüz Lekesi.1-4 dahil) paylaştığı bir zayıflık.

## Bu sürümde ne değişti: ortalama → morfolojik kapama (closing)

Ortalama yerine, Gözenek ailesindeki black-hat tekniğiyle aynı matematiksel aracı — ama **tersten, bir arka plan tahmin edici** olarak — kullanıyoruz. Morfolojik kapama (önce yerel maksimum alma, sonra yerel minimum alma) işlemi, **küçük koyu noktaları (çilleri) "görmezden gelerek"** onların altında/arasında olması beklenen düz ten yüzeyini yeniden inşa eder:

```
arka_plan(x,y) = closing(L)(x,y)    [çilleri "doldurarak" düz ten tahmini]
delta(x,y)     = arka_plan(x,y) - L(x,y)
```

**Neden bu işliyor:** Kapamanın ilk adımı (yerel maksimum), küçük koyu benekleri (kapama çekirdeğinden küçükse) tamamen yok sayar — o bölgede en yakın normal ten değerini "yayar". Bu, yoğun çil kümelerinde bile, her bir çilin kendi etrafındaki **gerçek (çilsiz) ten seviyesine** göre değerlendirilmesini sağlıyor — ortalama gibi komşu benekler tarafından aşağı çekilmiyor.

Göz/kaş/dudak gibi dışlanan alanların kapamayı yanlış yönlendirmemesi için, önce bu alanlar `cv2.inpaint` ile civar ten değerleriyle dolduruluyor, sonra kapama uygulanıyor.

## Kalibrasyon — dürüst bir düzeltme hikayesi

İlk denemede `C_OFFSET`'i düşürdüm (13→9), "artık arka plan çillerden etkilenmiyor, daha hassas olabilir" varsayımıyla — ama bu **yanlış çıktı**: gerçek delta dağılımını ölçünce (`p90=13, p95=21, p99=49`), 9'luk bir eşiğin piksellerin **%10'undan fazlasını** eşik-üstü saydığını gördüm (normal fotoğrafta bile 116 sahte tespit). Eşiği gerçek yüzdelik dilimlere göre **25**'e yükselterek düzelttim.

## Doğrulama

**Gerçek fotoğrafta, aynı ayarlarla v4 ile karşılaştırma:**

| | Yüz Lekesi.4 | Yüz Lekesi.5 |
|---|---|---|
| Aynı fotoğrafta tespit sayısı | ~18 | **28** (%55 artış) |

**Çözünürlük kararlılığı:**

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 700 px | %61.5 | 34 |
| 1200 px | %70.5 | 39 |
| 2000 px | %75.7 | 42 |

**Sıfır-bölge tutarlılığı** korundu: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

**Dürüstçe belirtilmesi gereken bir nokta:** Kullanıcının kendi fotoğrafı elimde olmadığı için, "yoğun çil kümesi" senaryosunu yapay olarak simüle etmeye çalıştım. İlk denemem (2000 nokta, iç içe geçmiş) çok agresifti — noktalar birleşip tek bir büyük koyu leke oluşturdu, gerçekçi değildi. İkinci, daha gerçekçi denemem (140 küçük, aralıklı, hafif koyulaştırılmış nokta) ise **her iki sürümde de** çoğunlukla algılanmadı — çünkü çok soluktu, ikisinin de eşiğinin altında kaldı; bu da anlamlı bir v4-v5 karşılaştırması sağlamadı. Bu yüzden asıl kanıtım **gerçek fotoğraftaki** tespit sayısı artışı (18→28) ve closing tekniğinin teorik/bilinen üstünlüğü (görüntü işlemede yoğun küçük yapılar için standart bir arka plan tahmin yöntemi olması). **Kendi fotoğrafınızla test edip sonucu paylaşırsanız, gerçek veriyle daha kesin bir değerlendirme yapabiliriz.**

## Önceki sürümle kıyaslama

| | Yüz Lekesi.4 | Yüz Lekesi.5 |
|---|---|---|
| Yerel referans yöntemi | Gauss ortalaması (bulanıklaştırma) | **Morfolojik kapama (closing)** |
| Yoğun kümede davranış | Benekler birbirini maskeler | **Her benek kendi civarına göre değerlendirilir** |
| Delik/kenar kontaminasyonu önleme | Normalize konvolüsyon + kenar marjı | **İnpainting + daha küçük kenar marjı** |
| Eşik (`C_OFFSET`) | 13.0 | 25.0 (farklı delta dağılımına göre kalibre) |
| Şekil filtresi | `MIN_CIRCULARITY=0.22` | **0.18 (biraz gevşetildi)** |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`leke.py` içinde)

- `CLOSING_KERNEL_RATIO = 0.035`, `MIN/MAX_CLOSING_KERNEL` — kapama çekirdeğinin göreli boyutu.
- `C_OFFSET = 25.0` — gerçek delta dağılımına göre kalibre edildi (bkz. yukarı).
- `DETECTION_MARGIN_RATIO = 0.02` — Yüz Lekesi.4'ün 0.05'inden küçültüldü (inpainting sayesinde daha az marja ihtiyaç var).
- `COUNT_CAP = 55.0` — yeni (daha yüksek) tespit sayılarına göre yeniden kalibre edildi.
- `MIN_CIRCULARITY = 0.18`, `MIN_AREA_RATIO` — biraz gevşetildi, daha fazla gerçek (ama küçük/hafif düzensiz) lekeyi yakalamak için.

## Bilinen sınırlama

- `cv2.inpaint`, büyük fotoğraflarda (yüksek çözünürlük) biraz daha yavaş çalışabilir — performans kritikse `INPAINT_NS` gibi daha hızlı bir alternatif denenebilir.
- Kapama çekirdeği tek bir çilden büyük olmalı ki onu "görmezden gelebilsin" — ama bu, ÇOK BÜYÜK tekil lekelerin (çekirdekten büyük) de yanlışlıkla "arka planın parçası" sayılıp kaçırılması riski taşır.
- Yapay test senaryom (yukarıda açıklandığı gibi) kesin bir doğrulama sağlamadı — gerçek kullanıcı fotoğrafıyla tekrar test edilmesi öneriliyor.

## Nasıl çalıştırılır

```bash
cd "yüz lekesi\5"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8505` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit sayısı | Yüz Lekesi.4 ile kıyas | Not |
|---|---|---|---|---|
| File.jpg (1024px) | — | 28 | 18 (%55 artış) | Aynı ayarlarla doğrudan karşılaştırma |
| _(kendi (özellikle çilli/benli) fotoğrafınızla tekrar deneyip sonucu paylaşabilirsiniz)_ | | | | |
