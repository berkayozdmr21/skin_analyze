# Yüz Kırışıklık.3 — Minimum Uzunluk Filtresi + Sıkılaştırılmış Eşikler

## Kullanıcının bildirdiği yeni sorun

> "bu sefer kırış olmayan yüzü de çok çıkardı"

Yüz Kırışıklık.2, yoğun kırışıklıklarla kaplı yüzlerde kaçırma sorununu çözdü — ama bu sefer **düz/kırışıksız** bir yüzde de çok sayıda yanlış tespit üretmeye başladı. Klasik bir **hassasiyet/özgüllük dengesi kayması**: recall'u iyileştirmek için (a) eşiği gevşetmiş (Gabor %88→%75), (b) yerel (kapama tabanlı) referansa geçmiştik — ikisi birlikte, düz cilteki ince doku/gözenek gibi normal detayları da "kırışıklık" gibi göstermeye başlamıştı.

## Somut ölçüm: gerçek tespitlerin çoğu çok kısaydı

Referans fotoğrafta (nispeten düz ten) Yüz Kırışıklık.2'nin bulduğu **30 bölge tek tek ölçüldü** (`face_scale = 304.9 px`):

| | Uzunluk aralığı | face_scale'e oranı |
|---|---|---|
| **20/30 tespit** | 5.7-20.1 px | **%1.9-6.6** |
| 7/30 tespit | 26.9-90.1 px | %8.8-29.6 |

Tespitlerin çoğu (20/30), yüzün ölçeğine kıyasla **çok kısaydı** — bu boyutta bir "kırışıklık" fiziksel olarak anlamsız, muhtemelen tek bir gözenek/ince doku noktası uzama (elongation) testini şans eseri geçmiş.

## Çözüm: üç hedefli, ölçüme dayalı sıkılaştırma

1. **Minimum uzunluk filtresi (YENİ)**: `MAX_LENGTH_RATIO_OF_FACE` gibi bir üst sınır zaten vardı ama alt sınır yoktu. Ölçülen verideki doğal boşluğa (20px altı çoğunlukla gürültü, 26px üstü çoğunlukla gerçek) dayanarak `MIN_LENGTH_RATIO_OF_FACE = 0.075` eklendi.
2. **Koyuluk eşiği yükseltildi**: `DARK_DELTA_THRESHOLD` 4.0'dan **7.0**'a çıkarıldı — ölçülen gerçek tespitlerin delta değerleri 7.7-18.2 arasındaydı, 4.0 eşiği bu aralığın çok altında kalıp fazla piksele izin veriyordu.
3. **Gabor eşiği kısmen geri çekildi**: %75'ten **%83**'e — Yüz Kırışıklık.1'in %88'ine tam dönmeden (o sürümün kaçırma sorununu geri getirmemek için), ama v2'nin aşırı gevşek eşiğinden sıkılaştırılarak.

## Doğrulama

**Düz yüzde yanlış-pozitif kontrolü:**

| | Yüz Kırışıklık.2 | Yüz Kırışıklık.3 |
|---|---|---|
| Tespit sayısı | 30 | **6** |
| Skor | %39.7 | **%20.2** |

Görsel kontrol: kalan 6 işaret, kaşlar arası çizgi, göz altı ince izler, burun kıvrımı gibi gerçekten anlamlı yerlerde — dağınık gözenek/doku gürültüsü kayboldu.

**Çözünürlük kararlılığı:**

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 170 px | %20.2 | 6 |
| 700 px | %20.3 | 6 |
| 1200 px | %24.3 | 7 |
| 2000 px | %32.7 | 10 |
| 3000 px | %13.9 | 3 |

**Sıfır-bölge tutarlılığı** korundu: eşik yapay olarak imkansız hale getirildiğinde skor **tam %0.0** çıktı.

## Dürüstçe: bu bir ödünleşim, bedelsiz değil

Aynı sentetik yoğun-kırışıklık testinde (v2'de kullanılan), v3'ün tespit sayısı **15'ten 5'e düştü** — yani düz yüzdeki yanlış-pozitifleri azaltırken, gerçekten kırışıklıklı bir yüzdeki hassasiyetten de biraz feda edildi. Bu bilinçli bir karar: kullanıcının **en son, en spesifik** şikayeti "düz yüzü de çok çıkardı" idi, bu yüzden önceliği hassasiyete (precision) verdim. Sentetik testin zaten gerçekçi olmadığı (v2'nin README'sinde de not edilmişti) göz önünde bulundurulduğunda, gerçek kullanıcı fotoğraflarıyla (hem düz hem kırışıklıklı) tekrar test edilmesi en güvenilir yol.

## Önceki sürümle kıyaslama

| | Yüz Kırışıklık.2 | Yüz Kırışıklık.3 |
|---|---|---|
| Minimum uzunluk filtresi | Yok | **Var (`MIN_LENGTH_RATIO_OF_FACE=0.075`)** |
| Koyuluk eşiği | 4.0 | **7.0** |
| Gabor eşiği | %75 | **%83** |
| Düz yüzde tespit sayısı | 30 (fazla) | **6 (makul)** |
| Yoğun kırışıklıklı sentetik testte tespit | 15 | 5 (bilinçli ödünleşim) |
| Yerel (kapama tabanlı) referans, Gabor filtre bankası, uzunluk-tabanlı skor | Aynı | Değişmedi |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`kirisiklik.py` içinde)

- **YENİ**: `MIN_LENGTH_RATIO_OF_FACE = 0.075` — bir bölgenin kırışıklık sayılması için gereken minimum uzunluk (yüzün karakteristik uzunluğuna oranlı).
- `DARK_DELTA_THRESHOLD = 7.0` — Yüz Kırışıklık.2'nin 4.0'ından yükseltildi.
- `GABOR_STRENGTH_PERCENTILE = 83.0` — Yüz Kırışıklık.2'nin 75'inden sıkılaştırıldı.
- Diğer parametreler (kapama tabanlı yerel referans, Gabor yönelimleri, uzama filtresi, tespit marjı) — değişmedi.

## Bilinen sınırlama

- `MIN_LENGTH_RATIO_OF_FACE` tek bir referans fotoğrafın ölçümüyle kalibre edildi; farklı yüz boyutlarında/mesafelerde çekilmiş fotoğraflarda ince ayar gerekebilir.
- Hassasiyet/özgüllük dengesi hâlâ mükemmel değil — gerçek hem düz hem yoğun kırışıklıklı fotoğraflarla test edilip geri bildirim verilirse daha ince ayar yapılabilir.

## Nasıl çalıştırılır

```bash
cd "yüz kırışıklık\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8703` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit sayısı | Yüz Kırışıklık.2 ile kıyas |
|---|---|---|---|
| File.jpg (düz ten, 170x226) | 0.202 | 6 | 0.397 / 30 (fazla tespit) |
| _(0 bölge simülasyonu)_ | 0.000 | 0 | — |
| _(kendi fotoğraflarınızla — hem düz hem kırışıklıklı — tekrar deneyip sonucu paylaşabilirsiniz)_ | | | |
