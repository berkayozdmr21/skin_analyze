# Göz Altı Morluk.2 — Ten Rengi Referanslı Z-Score

## Çözülen sorun

Göz altı morluk.1'de test edilen bir fotoğrafta, görsel olarak "hafif bir morluk var" denecek bir durum, algoritma tarafından skor **1.0** (maksimum/tavan) olarak işaretlenmişti — açıkça yanlış kalibrasyon. Sebep: skor sabit bir ölçekle (`DELTA_VISUAL_CAP = 30`, yani 30 L\* birimi) hesaplanıyordu ve referans olarak sadece **yanak** kullanılıyordu.

## Kullanıcının istediği düzeltme

> Ten rengini referans alsın, ona göre kıyaslasın.

## Yöntem — ne değişti

### 1. Referans genişletildi: "yanak" → "tüm ten"

Artık referans, kişinin **tüm düz cilt bölgelerinin birleşimi** (`forehead + nose + chin + cheeks`) — yani gerçek anlamda **genel ten rengi**. `face_mesh.py`'ye yeni bir fonksiyon eklendi: `make_skin_tone_mask()`. Bunun sebebi: sadece yanağı referans almak, o bölgede lokal bir gölge/kızarıklık varsa (ışık açısı, allık vb.) referansı yanlış yönlendirebiliyordu. Daha geniş bir alan, daha **stabil/temsili** bir "bu kişinin teni normalde nasıl" ölçümü veriyor.

### 2. Sabit ölçek → Z-SCORE (kişiye özel, istatistiksel ölçek)

Bu, asıl kalibrasyon düzeltmesi. Artık:
```
ten_L_ortalama = mean(L[ten_bölgesi])
ten_L_std      = std(L[ten_bölgesi])

z = (ten_L_ortalama - L_piksel) / ten_L_std
```
**Neden bu önemli:** Sabit "30 L\* birimi" gibi bir sayı, **herkes için aynı** işliyordu — ama insanların ten dokusu/doğal ışık-gölge varyasyonu çok farklı olabilir. Z-score, "bu kişinin teninde normalde ne kadar dalgalanma var, göz altı bunun **kaç katı**" sorusunu soruyor:
- Pürüzsüz/düz tenli biri için: küçük bir fark bile **yüksek** z-score verir (çünkü o kişide normalde hiç dalgalanma yok, küçük fark bile göze batar).
- Doğal olarak gölgeli/dokulu tenli biri için: aynı **mutlak** fark daha **düşük** z-score verir (çünkü o kişide zaten biraz dalgalanma normal).

Bu, "ten rengine göre kıyaslama" isteğinin istatistiksel karşılığı — sabit sayı yerine, kişinin **kendi** ten dağılımına göre göreceli bir ölçüm.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(clip(z, 0, None)[göz_altı_bandı]) / 2.5, 0, 1 )
```
`2.5` (`Z_SCORE_CAP`), "2.5 standart sapmalık bir fark = maksimum/tam morluk" demek — istatistikte bu zaten oldukça büyük/belirgin bir sapma sayılır (normal dağılımda ~%99'luk dilimin dışında kalır), bu yüzden makul bir üst sınır.

**Tespit işaretleme** de artık z-score cinsinden: `z > 1.2` (1.2 standart sapma) olan pikseller "tespit edilen bölge" sayılıp sarı kontur ile işaretleniyor.

Arayüzde artık sadece % skor değil, **ortalama Z-score** değeri de gösteriliyor (örn. "1.8" gibi) — bu, "kaç standart sapmalık bir farkla karşı karşıyayız" sorusuna doğrudan cevap, ham sayıyı görmek kalibrasyon yaparken çok işine yarayacak.

## Önceki versiyonla kıyaslama

| | Göz Altı Morluk.1 | Göz Altı Morluk.2 |
|---|---|---|
| Referans bölge | Sadece yanak | **Tüm ten (alın+burun+çene+yanak)** |
| Ölçek | Sabit (30 L* birimi) | **Z-score (kişinin kendi std sapması)** |
| Riski | Lokal yanak gölgesi/kızarıklığı referansı bozabilir | Daha stabil, ama ten bölgesi çok küçük/net değilse std sapma güvenilmez olabilir |
| Aşırı yüksek skor riski | Yüksek (sabit ölçek herkese uymuyor) | **Düşük (istatistiksel olarak kişiye göre normalize)** |

## Maskeleme

`under_eye` bandı **değişmedi**. Yeni eklenen: `make_skin_tone_mask()` — mevcut 4 bölge maskesinin birleşimi, yeni bir maskeleme yöntemi değil, var olanların birleştirilmiş hali.

## Ayarlanabilir parametreler (`morluk.py` içinde)

- `Z_SCORE_CAP = 2.5` — skorun 1.0'a (tam) ulaştığı standart sapma eşiği.
- `DETECTION_Z_THRESHOLD = 1.2` — bir pikselin "tespit edildi" (işaretlenecek) sayılması için gereken minimum z-score.
- `MIN_CONTOUR_AREA = 20` — gürültü temizliği.

## Nasıl çalıştırılır

```bash
cd "göz altı morluk\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8202` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Ortalama Z-score | Tespit edilen bölge sayısı | K.1 skoru (kıyas) | Not |
|---|---|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | | | |
