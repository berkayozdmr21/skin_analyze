# Akne.2 — Tüm Ten Referanslı Z-Score

## Akne.1'deki iki sorun

Akne.1'de test edilirken ortaya çıkan iki kalibrasyon sorunu — **Göz Altı Morluk.1 → .2** geçişindeki motivasyonla birebir aynı:

1. **Referans (yanak) kontamine.** `cheeks` maskesi, yüzü ortadan ikiye bölen bir convex hull olduğu için **dudak bölgesini de içine alıyordu**. Dudağın doğal kırmızı rengi, "tipik ten kırmızılığı" ortalamasını (`cheek_a_mean`) yapay olarak **yukarı çekiyordu** — bu da eşiğin gerekenden yüksek bir referansa göre ölçülmesine, yani akne tespitinin **gerekenden az hassas** olmasına yol açıyordu.
2. **Sabit ölçek adil değil.** Skor sabit bir ölçekle (`DELTA_VISUAL_CAP = 18` a\* birimi) hesaplanıyordu. Ama herkesin teninde aynı "doğal kırmızılık dalgalanması" yok — bu yüzden sabit ölçek, farklı ten tiplerini adil kıyaslamıyordu.

## Kullanıcının istediği düzeltme

> "aynı mantıkta değiştir, sistem biraz farklı sonuçlar versin doğrulanacak şekilde"

## Yöntem — ne değişti

### 1. Referans temizlendi ve genişletildi: "yanak" → "dışlama uygulanmış tüm ten"

Artık referans, `face_scan_mask`'in **kendisi** — yani alın+burun+çene+yanak birleşiminden göz/kaş/dudak bölgelerinin çıkarılmış hali (Akne.1'de zaten hedef bölge olarak vardı, bu sürümde **aynı zamanda referans** olarak da kullanılıyor). Bunun iki faydası var:

- **Dudak kontaminasyonu ortadan kalkıyor** — referans artık sadece gerçek ten piksellerinden hesaplanıyor.
- **Referans daha geniş/stabil** — Göz Altı Morluk.2'deki gerekçeyle aynı: tek bir bölgedeki lokal bir sapma (örn. yanaktaki hafif bir gölge) artık tüm referansı yanlış yönlendiremiyor.

### 2. Sabit ölçek → Z-SCORE (kişiye özel, istatistiksel ölçek)

```
ten_a_ortalama = mean(a[face_scan_mask])
ten_a_std      = std(a[face_scan_mask])

z = (a_piksel - ten_a_ortalama) / ten_a_std
```

**Neden bu önemli:** Sabit "18 a\* birimi" gibi bir sayı **herkes için aynı** işliyordu — ama insanların doğal ten kırmızılığı/dokusu farklı olabilir. Z-score, "bu kişinin teninde normalde ne kadar kırmızılık dalgalanması var, bu nokta bunun **kaç katı**" sorusunu soruyor:
- Pürüzsüz/düz tenli biri için: küçük bir kırmızılık farkı bile **yüksek** z-score verir.
- Doğal olarak kırmızımsı/dokulu tenli biri için: aynı **mutlak** fark daha **düşük** z-score verir.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(clip(z, 0, None)[face_scan_mask]) / 2.5, 0, 1 )
```

`Z_SCORE_CAP = 2.5` ve `DETECTION_Z_THRESHOLD = 1.2` — Göz Altı Morluk ailesiyle **aynı sabitler**, sistemler arası kabaca karşılaştırılabilirlik için. Arayüzde artık sadece % skor değil, **ortalama Z-score** değeri de gösteriliyor (örn. "1.4" gibi) — kaç standart sapmalık bir farkla karşı karşıya olduğumuzu doğrudan gösteriyor.

## Önceki sürümle kıyaslama

| | Akne.1 | Akne.2 |
|---|---|---|
| Referans bölge | Sadece yanak (dudak dahil, kontamine) | **Dışlama uygulanmış tüm ten (temiz)** |
| Ölçek | Sabit (18 a\* birimi) | **Z-score (kişinin kendi std sapması)** |
| Hedef bölge | Tüm yüz cildi | Tüm yüz cildi (**değişmedi**) |
| Dudak/göz/kaş dışlama | Sadece hedefte | **Hem referansta hem hedefte** |

## Doğrulama — sonuçlar gerçekten farklı mı?

Aynı 4 test fotoğrafında Akne.1 ile Akne.2, aynı script içinde art arda çalıştırılarak karşılaştırıldı (`göz altı morluk/1/testler` klasöründeki fotoğraflar, iki sürüme de ortak test seti olarak kullanıldı). Gerçek çalıştırma çıktısı:

| Fotoğraf | Akne.1 skoru | Akne.1 bölge sayısı | Akne.2 skoru | Akne.2 ortalama Z | Akne.2 bölge sayısı |
|---|---|---|---|---|---|
| Ekran görüntüsü 131457 | 0.066 | 12 | 0.122 | 0.30 | 12 |
| Ekran görüntüsü 131546 | 0.060 | 7 | 0.140 | 0.35 | 15 |
| Ekran görüntüsü 131606 | 0.085 | 5 | 0.154 | 0.38 | 11 |
| Ekran görüntüsü 131624 | 0.140 | 10 | 0.164 | 0.41 | 10 |

Skorlar tutarlı biçimde **yükseldi** (dudak kontaminasyonu kalktığı için referans ortalaması düştü, dolayısıyla aynı piksel artık referanstan daha uzak/anlamlı sayılıyor) ve bölge sayısı bazı fotoğraflarda değişti (ör. 131546: 7 → 15) — bu, referans temizliğinin ve kişiye özel z-score ölçeğinin gerçekten farklı, doğrulanabilir sonuçlar ürettiğini gösteriyor. Kendi fotoğraflarınla da "Nasıl çalıştırılır" bölümündeki adımlarla deneyebilirsin.

## Maskeleme

**Değişmedi** — `face_mesh.py`, Akne.1 ile birebir aynı kopya (zaten `make_skin_tone_mask`/`make_face_scan_mask` fonksiyonları oradaydı, bu sürüm onları referans için de kullanmaya başladı).

## Ayarlanabilir parametreler (`akne.py` içinde)

- `Z_SCORE_CAP = 2.5` — skorun 1.0'a (tam) ulaştığı standart sapma eşiği.
- `DETECTION_Z_THRESHOLD = 1.2` — bir pikselin "tespit edildi" sayılması için gereken minimum z-score.
- `MIN_CONTOUR_AREA = 6` — gürültü temizliği (Akne.1 ile aynı, sivilce büyüklüğüne uygun).

## Nasıl çalıştırılır

```bash
cd "akne\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8302` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Ortalama Z-score | Tespit edilen bölge sayısı | Akne.1 skoru (kıyas) | Not |
|---|---|---|---|---|---|
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
