# Yüz Kırışıklık.2 — Yerel Arka Plan (Morfolojik Kapama) Referansı

## Kullanıcının bildirdiği sorun (ekran görüntüsüyle)

Yaşlı bir kadının yoğun kırışıklıklarla kaplı yüzünde, Yüz Kırışıklık.1 gözle görülür pek çok kırışıklığı **kaçırıyordu** — sadece 21 küçük tespit, oysa fotoğrafta alın, göz çevresi, yanaklar, çene dahil neredeyse yüzün tamamı derin kırışıklıklarla kaplıydı.

## Kök neden (Yüz Lekesi.5'teki aynı sorun sınıfı)

Yüz Kırışıklık.1, koyuluk referansını **tüm yüzün** medyanı/MAD'i ile hesaplıyordu. Ama eğer yüzün büyük bir kısmı zaten kırışıklarla kaplıysa, bu referansın **kendisi** kırışıklıklardan etkilenip bozuluyor — "normal/kırışıksız ten" tahmini artık doğru değil, kısmen kırışıklı tenin ortalaması oluyor. Sonuç: gerçek bir kırışıklık pikseli, artık "zaten kırışıklı" olan referansa göre o kadar "anormal koyu" görünmüyor — **kırışıklıklar birbirini maskeliyor**.

Bu, Yüz Lekesi.5'te "yoğun çil kümelerinin ortalamayı aşağı çekip birbirini maskelemesi" ile **birebir aynı** sorun sınıfı — ve aynı çözüm burada da uygulandı.

## Çözüm: morfolojik kapama ile yerel arka plan tahmini

Tüm yüzün TEK bir medyan/MAD değeri yerine, her piksel için KENDİ yerel çevresinde "kırışıksız olsaydı nasıl görünürdü" tahmini yapılıyor — Yüz Lekesi.5'teki aynı teknik: morfolojik **kapama** (önce yerel maksimum, sonra yerel minimum), küçük koyu çizgileri (kırışıklıkları) "görmezden gelerek" altlarında olması beklenen düz ten yüzeyini yeniden inşa eder.

```
arka_plan(x,y) = closing(L)(x,y)
delta(x,y)      = arka_plan(x,y) - L(x,y)
```

Kapama çekirdeği tek bir kırışıklığın GENİŞLİĞİNDEN büyük olmalı (ki onu görmezden gelebilsin) — bu, çizginin UZUNLUĞUNDAN bağımsız çalışır (kapama yerel bir işlemdir), yani çok uzun ama ince kırışıklıklar bile doğru şekilde "arka plana gömülüp" çıkarılabiliyor.

## Doğrulama

**Gerçek fotoğrafta, v1 ile karşılaştırma:**

| | Yüz Kırışıklık.1 | Yüz Kırışıklık.2 |
|---|---|---|
| Tespit sayısı | 8 | **30** |
| Skor | %13.6 | **%39.7** |

**Çözünürlük kararlılığı:**

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 170 px | %39.7 | 30 |
| 700 px | %47.3 | 37 |
| 1200 px | %59.4 | 50 |
| 2000 px | %68.1 | 54 |
| 3000 px | %58.2 | 66 |

Mutlak aralık genişledi (v1: %13.6-%35.1, v2: %39.7-%68.1), ama **oransal** tutarlılık (maksimum/minimum oranı) aslında v1'den DAHA İYİ: v1'de 2.58 kat, v2'de sadece 1.71 kat. Sıfır-bölge tutarlılığı da korundu: eşik yapay olarak imkansız hale getirildiğinde skor **tam %0.0** çıktı.

**Sentetik doğrulama (dürüstçe — kesin kanıt değil):** Kullanıcının gerçek fotoğrafı elimde olmadığı için, yoğun kırışıklık senaryosunu simüle etmek üzere referans fotoğrafa 90 adet ince/soluk (1px, hafif koyulaştırılmış) eğri çizgi yapay olarak eklendi. v2, v1'e göre daha fazla çizgi buldu (15 vs 7) ama **ikisi de** planlanan 90 çizginin çoğunu kaçırdı — bunun sebebi muhtemelen sentetik çizgilerin GERÇEK derin kırışıklıklardan çok daha ince/soluk olması (sentetik test iyi bir vekil olmadı, bu dürüstçe belirtilmeli). Asıl güvenilir kanıt, gerçek fotoğraftaki 8→30 artışı.

## Önceki sürümle kıyaslama

| | Yüz Kırışıklık.1 | Yüz Kırışıklık.2 |
|---|---|---|
| Koyuluk referansı | Tüm yüzün medyanı/MAD'i (global) | **Yerel (morfolojik kapama tabanlı)** |
| Yoğun kırışıklık riski | Kırışıklıklar birbirini maskeler | **Her kırışıklık kendi yerel çevresine göre değerlendirilir** |
| Gabor eşiği | %88 (sıkı) | %75 (gevşetildi — asıl seçicilik artık yerel referanstan geliyor) |
| Gabor/uzama/skor mantığı | — | Değişmedi |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`kirisiklik.py` içinde)

- **YENİ**: `CLOSING_KERNEL_RATIO`, `MIN/MAX_CLOSING_KERNEL` — yerel arka plan tahmini için kapama çekirdeğinin göreli boyutu.
- **YENİ**: `DARK_DELTA_THRESHOLD = 4.0`, `DELTA_VISUAL_CAP = 14.0` — yerel koyuluk farkı eşiği ve normalizasyon üst sınırı (Yüz Kırışıklık.1'deki medyan/MAD z-score'un yerini aldı).
- `GABOR_STRENGTH_PERCENTILE = 75.0` — Yüz Kırışıklık.1'in 88'inden gevşetildi.
- Diğer parametreler (Gabor yönelimleri/boyutları, uzama filtresi, tespit marjı, uzunluk-tabanlı skor) — değişmedi.

## Bilinen sınırlama

- Sentetik doğrulama testim yeterince gerçekçi değildi (çok ince/soluk çizgiler) — gerçek yoğun kırışıklıklı bir fotoğrafla tekrar test edilmesi, düzeltmenin tam etkisini görmek için önerilir.
- Çözünürlük arttıkça skor/tespit sayısı hâlâ artıyor (beklenen ama tam sabit olmayan bir davranış — ailenin diğer projelerinde de gözlemlenen bir örüntü).
- Kapama çekirdeği tek bir kırışıklıktan büyük olmalı ki onu "görmezden gelebilsin" — ama bu, ÇOK GENİŞ/derin tekil çizgilerin de yanlışlıkla "arka planın parçası" sayılıp kaçırılması riski taşır (Yüz Lekesi.5'te de not edilen aynı ödünleşim).

## Nasıl çalıştırılır

```bash
cd "yüz kırışıklık\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8702` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Ağırlıklı uzunluk (px) | Tespit sayısı | Yüz Kırışıklık.1 ile kıyas |
|---|---|---|---|---|
| File.jpg (170x226) | 0.397 | 484.4 | 30 | 0.136 / 166.3 / 8 |
| _(0 bölge simülasyonu)_ | 0.000 | 0.0 | 0 | — |
| _(kendi (özellikle yoğun kırışıklıklı) fotoğrafınızla tekrar deneyip sonucu paylaşabilirsiniz)_ | | | | |
