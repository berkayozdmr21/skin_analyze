# Göz Altı Morluk.5 — Yerel Aydınlatma Düzeltmeli Mahalanobis Mesafesi

## Önceki sürümün (M.4) kör noktası

Göz Altı Morluk.4, ten bölgesi için **tek bir global ortalama vektör** (`L, a, b`) ve **tek bir global kovaryans matrisi** hesaplıyordu — yüzün her noktası, aynı tek referansla kıyaslanıyordu.

Sorun: **yüzün aydınlatması her zaman homojen değildir.** Yandan gelen ışık, kafa açısı veya oda ışığının konumu yüzünden bir yanak diğerinden doğal olarak daha parlak/koyu görünebilir. Bu durumda tek bir global ortalama **yanlış** bir referans verir: doğal olarak daha gölgeli tarafta kalan göz altı, sırf o taraf global ortalamadan uzak olduğu için morlukla **hiç ilgisi olmadan** yapay olarak yüksek skor alabilir.

## Bu sürümde ne değişti

### 1. Global ortalama → yerel (piksel-bazlı) ortalama alanı

Tek bir skaler yerine, her piksel konumu için **ayrı** bir "burada ten normalde nasıl olmalı" değeri hesaplanıyor — **normalize edilmiş konvolüsyon** (normalized convolution) ile:

```
yerel_ortalama(x,y) = GaussianBlur(ten_maskesi × L/a/b)(x,y) / GaussianBlur(ten_maskesi)(x,y)
```

Bu, ten bölgesindeki komşu piksellerin Gauss-ağırlıklı ortalamasını göz altı dahil tüm görüntüye extrapole eder. Çekirdek genişliği (`sigma`), **yüz genişliğinin bir oranı** (`LOCAL_BASELINE_SIGMA_RATIO = 0.18`) — yeterince geniş ki gürültü değil aydınlatma *trendi* yakalansın, yeterince dar ki gerçekten "yerel" kalsın.

### 2. Ham kovaryans → rezidüel (detrend edilmiş) kovaryans

M.4'te kovaryans, ten piksellerinin **ham** değerlerinden hesaplanıyordu — bu da aydınlatma gradyanının kendisini varyansa dahil ediyordu (yüzün bir ucundan diğerine doğal parlaklık değişimi, gerçek doku gürültüsünden çok daha büyük olabilir). Sonuç: kovaryans olması gerekenden **büyük**, mesafe ölçüsü olması gerekenden **az hassas**.

Bu sürümde her ten pikselinin **kendi konumundaki yerel ortalamadan sapması** (`rezidüel = piksel − o_noktadaki_yerel_ortalama`) hesaplanıp, kovaryans **bu** rezidüellerden çıkarılıyor. Aydınlatma trendi çıkarıldığı için kalan varyans gerçekten doku/gürültü varyansı — daha küçük, daha **sıkı** bir kovaryans. Aynı büyüklükteki bir renk/koyuluk sapması artık daha **yüksek** bir Mahalanobis mesafesi veriyor, çünkü "normal" dalgalanma artık daha dar tanımlanmış.

### 3. Değişmeyenler

- Yön kısıtlaması aynı: sadece **"yerel referansa göre daha koyu (L) VE daha mavi (b)"** yönündeki sapmalar sayılıyor.
- Ölçek çerçevesi aynı: `MAHALANOBIS_CAP = 3.37` (χ²(3), %99 persentili), `DETECTION_THRESHOLD = 2.15` (χ²(3), %80 persentili) — M.4 ile skorlar kabaca karşılaştırılabilir kalsın diye.
- Ten/göz altı maskeleme mantığı değişmedi.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(yön_kısıtlı_D_yerel[göz_altı_bandı]) / MAHALANOBIS_CAP, 0, 1 )
```

Formül M.4 ile aynı görünüyor ama `D_yerel` artık **piksele özel** bir referansa göre hesaplanıyor — global bir sabit değil.

## Önceki sürümlerle kıyaslama

| | M.3 | M.4 | M.5 |
|---|---|---|---|
| Referans | Global ortalama (L+b, ayrı) | Global ortalama (L+a+b, birlikte) | **Yerel/piksel-bazlı ortalama alanı** |
| Kovaryans | — (sabit ağırlık) | Ham (global) | **Rezidüel (aydınlatma trendi çıkarılmış)** |
| Aydınlatma gradyanı riski | Yüksek | Orta (hâlâ global referans) | **Düşük (piksel-bazlı düzeltme)** |
| Hassasiyet | — | Orta | **Daha yüksek (daha sıkı kovaryans)** |

## Bilinen sınırlama

Yerel ortalama alanı, ten maskesi göz altına yeterince yakın/geniş değilse (örn. yüzün büyük kısmı kadraj dışıysa) güvenilmez extrapolasyon üretebilir — `MIN_SIGMA` bir alt sınır koyar ama garanti değildir. Ayrıca daha sıkı kovaryans, gürültüye karşı biraz daha hassas hale gelebilir; bu yüzden `MIN_CONTOUR_AREA` ve winsorize adımı (aynı M.4'teki gibi) önemini koruyor.

## Maskeleme

**Değişmedi** — `face_mesh.py` ve `preprocess.py`, Göz Altı Morluk.4 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`morluk.py` içinde)

- `MAHALANOBIS_CAP = 3.37`, `DETECTION_THRESHOLD = 2.15` — M.4 ile aynı (χ²(3) tabanlı).
- `LOCAL_BASELINE_SIGMA_RATIO = 0.18` — yerel ortalama alanının Gauss çekirdeği, yüz genişliğinin oranı olarak.
- `MIN_SIGMA = 15.0` — çekirdek genişliği için alt sınır.
- `WINSOR_PERCENTILES = (5, 95)`, `COV_REG_EPS = 1e-3` — M.4 ile aynı amaçla.
- `MIN_CONTOUR_AREA = 20` — gürültü temizliği.

## Nasıl çalıştırılır

```bash
cd "göz altı morluk\5"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8205` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Mahalanobis D | z_L | z_a | z_b | Tespit sayısı | M.4 skoru (kıyas) | Not |
|---|---|---|---|---|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | | | | | | |
