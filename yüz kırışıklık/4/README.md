# Yüz Kırışıklık.4 — Çözünürlük Tavanı Düzeltmesi + Dengelenmiş Eşikler

## Kullanıcının bildirdiği iki sorun

> "yaşlılarınki çok düştü buna bir ayar gerekiyor, bir sıkıntı var, düzenle... bir yerlerde patlak veriyor"

## Sorun 2: gerçek bir hata bulundu — çözünürlük tavanları çok düşük

"Patlak veriyor" ifadesi gerçek bir hataya işaret ediyordu. Yüz Kırışıklık.1-3'teki tüm çekirdek boyutları (`_relative_kernel`) yüz ölçeğine **oranlı** hesaplanıyordu ama bir **üst sınırla** (`max_k`) sınırlandırılmıştı. Bu tavanlar sadece düşük-orta çözünürlüklü test fotoğraflarıyla (170-1200px) kalibre edilmişti — gerçek yüksek çözünürlüklü fotoğraflarda (telefon kameraları kolayca 3000-4000px+ üretir) bu tavanlara **çoktan çarpılıyordu**:

| Çözünürlük | Parametre | İstenen | Tavan | Durum |
|---|---|---|---|---|
| 2000px | DETECTION_MARGIN | 82 | 65 | **çoktan çarpıyor** |
| 3000px | GABOR_KSIZE | 53 | 41 | çarpıyor |
| 3000px | CLOSING_KERNEL | 35 | 25 | çarpıyor |
| 3000px | DETECTION_MARGIN | 124 | 65 | çarpıyor |
| 4000px | GABOR_KSIZE | 71 | 41 | çarpıyor |

Tavana çarpan bir çekirdek artık çözünürlükle **orantılı büyümüyor** — bu, bu proje ailesi boyunca defalarca öğrenilen "çözünürlükten bağımsızlık" dersinin ihlaliydi. Sonuç: yüksek çözünürlükte kapama çekirdeği gerçekte olması gerekenden küçük kalıyor, yerel arka plan tahmini kırışıklığı düzgün "görmezden gelemiyor", tespit sayısı beklenmedik şekilde çöküyor — Yüz Kırışıklık.3'ün kendi ölçümünde: **2000px'te 10 tespit, 3000px'te sadece 3** (düzgün bir eğilim değil, ani bir çöküş).

**Düzeltme:** Tüm tavanlar, gerçekçi telefon kamerası çözünürlüklerini (6000px+) rahatça kapsayacak şekilde yükseltildi (`MAX_GABOR_KSIZE`: 41→121, `MAX_CLOSING_KERNEL`: 25→81, `MAX_DETECTION_MARGIN`: 65→181, `MAX_CLEAN_KERNEL`: 11→31).

## Sorun 1: hassasiyet dengesi yeniden ayarlandı

Yüz Kırışıklık.3'teki üç parametre (Gabor eşiği, koyuluk eşiği, minimum uzunluk) tek bir düz-tenli fotoğrafla kalibre edilip fazla sıkılaştırılmıştı — kullanıcı geri bildirimine göre bu, yaşlı/kırışıklıklı yüzlerdeki performansı olumsuz etkiledi. Üçü de Yüz Kırışıklık.2 (çok gevşek) ile Yüz Kırışıklık.3 (çok sıkı) arasında daha dengeli bir noktaya çekildi.

## Doğrulama

### 1. "Patlama" hatası düzeldi

| Kaynak genişlik | Skor | Tespit sayısı | Çekirdek tavana çarpıyor mu? |
|---|---|---|---|
| 170 px | %19.8 | 9 | Hayır |
| 700 px | %31.7 | 15 | Hayır |
| 1200 px | %43.6 | 25 | Hayır |
| 2000 px | %43.6 | 23 | Hayır |
| 3000 px | %39.9 | 22 | Hayır |
| 4000 px | %38.2 | 19 | Hayır |

Artık **hiçbir çözünürlükte** çekirdek tavana çarpmıyor (`capped=False`, ayrıca doğrulandı). Eğilim de düzgün: yükselip stabilize oluyor, Yüz Kırışıklık.3'teki gibi ani bir çöküş yok.

### 2. Hassasiyet dengesi

| | Yüz Kırışıklık.2 (çok gevşek) | Yüz Kırışıklık.3 (çok sıkı) | Yüz Kırışıklık.4 (dengeli) |
|---|---|---|---|
| Düz yüzde tespit sayısı | 30 | 6 | **9** |
| Sentetik yoğun-kırışıklık testinde tespit | 15 | 5 | **11** |

v4, düz yüzde v2'ye göre çok daha az yanlış tespit üretirken (30→9), sentetik yoğun-kırışıklık testinde v3'ün kaybettiği hassasiyetin çoğunu geri kazandı (5→11, v2'nin 15'ine yaklaşarak).

**Sıfır-bölge tutarlılığı** korundu: eşik yapay olarak imkansız hale getirildiğinde skor **tam %0.0** çıktı.

**Görsel doğrulama:** Düz yüzdeki 9 tespit, nazolabial kıvrım, göz altı çizgileri, burun köprüsü, ağız köşesi gibi anlamlı yerlerde — dağınık gürültü yok.

## Önceki sürümle kıyaslama

| | Yüz Kırışıklık.3 | Yüz Kırışıklık.4 |
|---|---|---|
| `MAX_GABOR_KSIZE` | 41 | **121** |
| `MAX_CLOSING_KERNEL` | 25 | **81** |
| `MAX_DETECTION_MARGIN` | 65 | **181** |
| `MAX_CLEAN_KERNEL` | 11 | **31** |
| Yüksek çözünürlükte davranış | Beklenmedik çöküş (2000px+'ten itibaren) | **Düzgün, öngörülebilir** |
| `GABOR_STRENGTH_PERCENTILE` | 83.0 | **80.0** |
| `DARK_DELTA_THRESHOLD` | 7.0 | **5.5** |
| `MIN_LENGTH_RATIO_OF_FACE` | 0.075 | **0.035** |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`kirisiklik.py` içinde)

- `MAX_GABOR_KSIZE = 121`, `MAX_CLOSING_KERNEL = 81`, `MAX_DETECTION_MARGIN = 181`, `MAX_CLEAN_KERNEL = 31` — yükseltilmiş çözünürlük tavanları.
- `GABOR_STRENGTH_PERCENTILE = 80.0`, `DARK_DELTA_THRESHOLD = 5.5`, `MIN_LENGTH_RATIO_OF_FACE = 0.035` — dengelenmiş hassasiyet eşikleri.
- Diğer parametreler (kapama tabanlı yerel referans, Gabor yönelimleri, uzama filtresi, uzunluk-tabanlı skor) — değişmedi.

## Bilinen sınırlama

- Dengeleme hâlâ tek bir referans fotoğraf + bir sentetik test ile yapıldı — gerçek yaşlı/kırışıklıklı bir fotoğrafla test edilip geri bildirim verilirse daha kesin bir kalibrasyon yapılabilir.
- Çok yüksek çözünürlüklerde (6000px+, tavanların üzerinde) davranış tekrar sınırlanabilir — ama bu, günümüz telefon kameralarının tipik çıktısının oldukça üzerinde bir değer.

## Nasıl çalıştırılır

```bash
cd "yüz kırışıklık\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8704` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit sayısı | v2 ile kıyas | v3 ile kıyas |
|---|---|---|---|---|
| File.jpg (düz ten, 170x226) | 0.198 | 9 | 30 (fazla) | 6 (az) |
| File.jpg + sentetik yoğun kırışıklık | 0.237 | 11 | 15 | 5 (az) |
| _(0 bölge simülasyonu)_ | 0.000 | 0 | — | — |
| _(kendi (özellikle yaşlı/kırışıklıklı) fotoğraflarınızla tekrar deneyip sonucu paylaşabilirsiniz)_ | | | | |
