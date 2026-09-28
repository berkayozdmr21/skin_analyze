# Akne.4 — Çözünürlükten Bağımsız Tespit

## Kullanıcının bildirdiği sorun

> "biraz kötü olmuş, 85 adet bulunmuş bir testte %18, diğerinde 22 tane bulunmuş %16 — bu mantığı da düzelt"

Akne.3, gerçek fotoğraflarla test edildiğinde beklenmedik şekilde **çok fazla** bölge tespit ediyordu (85, 22 gibi) — Akne.3'ü geliştirirken kullanılan tek referans fotoğrafta (15 bölge) görülmeyen bir davranış.

## Kök neden (aynı fotoğrafla, farklı çözünürlüklerde doğrulandı)

Akne.3'ün `MIN_CONTOUR_AREA` (4 piksel) ve gürültü temizleme çekirdeği (3×3, sabit boyut) parametreleri **mutlak piksel sayılarıydı** — fotoğrafın çözünürlüğünden bağımsız, hep aynı sabit değer. Aynı test fotoğrafı farklı kaynak genişliklerine ölçeklenip Akne.3 ile çalıştırıldığında:

| Kaynak fotoğraf genişliği | Akne.3 tespit sayısı |
|---|---|
| 170 px | 15 |
| 700 px | 13 |
| 1200 px | **32** |
| 2000 px | **28** |
| 3000 px | **48** |

Skor neredeyse hiç değişmiyordu (~0.143-0.144) ama tespit sayısı **çözünürlükle birlikte patlıyordu**. Sebep: yüksek çözünürlüklü bir fotoğrafta tenin doğal dokusu (gözenekler, ince kıl gölgeleri, hafif ışık dalgalanmaları) binlerce küçük ama az-çok yuvarlak piksel grubu oluşturuyor — düşük çözünürlükte bunlar zaten tek bir piksele sıkışıp kayboluyordu, ama yüksek çözünürlükte her biri kendi başına "4 pikselden büyük, dairesel bir bölge" sayılıp sivilce olarak işaretleniyordu. Kullanıcının bildirdiği 85 ve 22 sayıları, muhtemelen farklı çözünürlükteki iki farklı fotoğraftan kaynaklanıyor.

## Bu sürümde ne değişti

Bütün piksel-tabanlı sabitler artık **mutlak sayı değil, o fotoğrafa özgü taranan yüz alanının (`face_scan_mask`) piksel sayısından türetilen bir ORAN**:

```
face_scale = sqrt(scan_piksel_sayısı)      # karakteristik "uzunluk"

kernel_boyutu     = clip(round(face_scale * 0.012), 3, 15) | tek sayı
min_kontur_alanı  = max(6, round(scan_piksel_sayısı * 0.00012))
```

Böylece yüksek çözünürlüklü bir fotoğrafta hem gürültü temizleme çekirdeği **orantılı olarak büyüyor** (küçük dokusal gürültüyü daha agresif siliyor) hem de "sivilce sayılmak için gereken minimum alan" **orantılı olarak büyüyor** — düşük çözünürlüktekiyle aynı **göreceli** büyüklükte bir leke gerekiyor, mutlak piksel sayısı değil.

`DETECTION_Z_THRESHOLD` da ek bir güvenlik payı olarak 1.0'dan 1.15'e yükseltildi.

`MAX_SPOT_AREA_RATIO` (sivilcenin taranan alana oranla üst boyut sınırı) zaten Akne.3'te bir orandı, değişmedi.

## Doğrulama — aynı test artık çözünürlükten bağımsız

Aynı fotoğraf, aynı 5 kaynak genişliğinde, Akne.4 ile tekrar çalıştırıldı:

| Kaynak fotoğraf genişliği | Akne.3 (önceki) | Akne.4 (bu sürüm) | kernel | min_alan |
|---|---|---|---|---|
| 170 px | 15 | **14** | 5 | 11 |
| 400 px | 14 | **14** | 5 | 11 |
| 700 px | 13 | **14** | 5 | 20 |
| 1200 px | 32 | **16** | 9 | 59 |
| 2000 px | 28 | **15** | 15 | 166 |
| 3000 px | 48 | **19** | 15 | 375 |

Aralık **13-48**'den **14-19**'a daraldı — hâlâ mükemmel sabit değil (çok yüksek çözünürlükte gerçek doku farkları biraz daha fazla ayrı bölge olarak sayılabiliyor) ama artık makul ve öngörülebilir. Yüksek çözünürlükte (2000px kaynak) işaretlenmiş görsel de kontrol edildi — geniş burun/çene taşması yok, küçük ve yerinde işaretlemeler (göz altı, ağız kenarı, çenedeki gerçek ben/lekeler) korunuyor.

## Önceki sürümle kıyaslama

| | Akne.3 | Akne.4 |
|---|---|---|
| Gürültü temizleme çekirdeği | Sabit 3×3 | **Taranan alana oranlı (3-15 px arası)** |
| Min. kontur alanı | Sabit 4 px | **Taranan alana oranlı (min. 6 px)** |
| Tespit eşiği | 1.0 | 1.15 (ek güvenlik payı) |
| Çözünürlük değişince tespit sayısı | 13 → 48 arası (13x fark) | **14 → 19 arası (1.4x fark)** |

## Maskeleme

**Değişmedi** — `face_mesh.py`, Akne.3 ile birebir aynı kopya. Kırmızılık+doygunluk birleşimi ve dairesellik/alan filtresi mantığı da (Akne.3'ten) değişmedi — sadece bu filtrelerin eşik DEĞERLERİ artık göreli.

## Ayarlanabilir parametreler (`akne.py` içinde)

- `WEIGHT_REDNESS = 0.6`, `WEIGHT_SATURATION = 0.4` — Akne.3 ile aynı.
- `Z_SCORE_CAP = 2.5` — Akne.3 ile aynı.
- `DETECTION_Z_THRESHOLD = 1.15` — Akne.3'ün 1.0'ından yükseltildi.
- `MIN_CIRCULARITY = 0.35`, `MAX_SPOT_AREA_RATIO = 0.012` — Akne.3 ile aynı.
- **YENİ**: `KERNEL_SIZE_RATIO = 0.012`, `MIN_KERNEL_SIZE = 3`, `MAX_KERNEL_SIZE = 15` — gürültü temizleme çekirdeğinin göreli boyutlandırması.
- **YENİ**: `MIN_AREA_RATIO = 0.00012`, `MIN_AREA_FLOOR = 6` — minimum kontur alanının göreli boyutlandırması.

## Bilinen sınırlama

- Çok yüksek çözünürlükte (3000px+ kaynak) tespit sayısı hâlâ hafifçe artıyor (19'a kadar) — tam sabit değil, çünkü gerçekten de yüksek çözünürlükte bazı gerçek küçük dokusal farklar daha net ayrılabiliyor. Bu, "hiç değişmeyen" bir sayı yerine, çözünürlükle **makul ve kontrollü** şekilde ölçeklenen bir davranış olarak kabul edildi.
- `MIN_AREA_RATIO` ve `KERNEL_SIZE_RATIO` sabitleri hâlâ ampirik/elle ayarlanmış; farklı kameralardan (farklı gürültü karakteristiğine sahip) gelen fotoğraflarda ince ayar gerekebilir.

## Nasıl çalıştırılır

```bash
cd "akne\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8304` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit edilen bölge sayısı | Akne.3 ile kıyas | Not |
|---|---|---|---|---|
| File.jpg (170x226, çeşitli ölçeklerde test edildi) | ~0.143 | 14-19 (çözünürlüğe göre) | 13-48 aralığından daraldı | Doğrulama testi, yukarıya bakınız |
| _(buraya kendi test sonuçlarını da ekleyebilirsin — özellikle 85/22 bulunan orijinal fotoğraflarla tekrar dene)_ | | | | |
