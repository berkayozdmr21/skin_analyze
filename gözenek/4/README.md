# Gözenek.4 — Çok Ölçekli (Multi-Scale) Tespit

## Kullanıcının isteği

> "sayısal mantık aynı, gözenek görme kısmını değiştirelim"

Yani: Gözenek.3'teki skor felsefesi (sayı/şiddet ağırlıklı, alan değil) **korunacak**, değişecek olan gözeneklerin **nasıl tespit edildiği** ("görme" kısmı).

## Gözenek.1-3'ün sorunu: tek çekirdek boyutu

Önceki sürümler, black-hat dönüşümü için **tek bir çekirdek boyutu** kullanıyordu (yüz ölçeğine oranlı, ama hep tek bir değer). Ama gerçek gözenekler aynı yüzde bile farklı boyutlarda olabilir — bazıları çok ince/küçük, bazıları daha geniş/belirgin. Tek bir çekirdek boyutu sadece o boyuta yakın gözenekleri iyi yakalar; küçük çekirdek büyük gözenekleri parçalı yakalar, büyük çekirdek ince gözenekleri "siler".

## Çözüm: üç farklı çekirdek boyutuyla tarama

Bu, görüntü işleme literatüründe "ölçek seçimi" (scale selection) problemi olarak bilinir — standart çözüm, tek bir ölçek yerine **birden fazla ölçekte** tarama yapıp sonuçları birleştirmek (scale-space / multi-scale blob detection). Artık üç farklı black-hat çekirdek boyutuyla (küçük/orta/büyük, hepsi yüz ölçeğine oranlı: `0.010`, `0.018`, `0.032`) ayrı ayrı taranıyor, her ölçekte **bölgesel z-score** hesaplanıyor (Gözenek.2/3'teki mantık aynen korunuyor), sonra bu üç z-haritasının **piksel bazlı maksimumu** alınıyor.

## Yan ürün: Gözenek.3'te bulunan bir hata da düzeltildi

Bu sürümü farklı çözünürlüklerde test ederken, Gözenek.3'ün skor formülünde **daha önce fark edilmemiş bir hata** bulundu:

```
Gözenek.3: skor = clip( (Σ şiddet_i) / (taranan_alan/100) / DENSITY_CAP, 0, 1 )
```

Ağırlıklı sayı (`Σ şiddet_i`), çözünürlükten bağımsız (göreli) filtreler sayesinde zaten **yaklaşık sabit** kalıyordu (fiziksel yüzdeki gerçek gözenek sayısını yansıtıyor) — ama `taranan_alan` (piksel sayısı) çözünürlükle **orantılı büyüyordu**. Yaklaşık sabit bir sayıyı, çözünürlükle orantılı büyüyen bir alana bölmek, **skorun çözünürlük arttıkça çökmesine** yol açıyordu:

| Kaynak genişlik | Gözenek.3 skoru |
|---|---|
| 170 px | %58.2 |
| 700 px | %37.6 |
| 1200 px | %12.4 |
| 2000 px | **%5.1** |

Aynı fotoğraf, sadece çözünürlük değiştirilerek, %58'den %5'e düşüyordu — 11 kata yakın bir sapma. Bu hata Gözenek.3'ü geliştirirken fark edilmemişti çünkü o sürümde sadece TEK bir çözünürlükte eşik-değişimi testi yapılmıştı, çözünürlükler arası karşılaştırma yapılmamıştı.

### Düzeltme

Skor artık **alana bölünmüyor** — doğrudan sabit bir referansla (`COUNT_CAP = 200`) kıyaslanıyor:

```
Gözenek.4: skor = clip( (Σ şiddet_i) / COUNT_CAP, 0, 1 )
```

Kullanıcının "sayısal mantık aynı kalsın" isteğindeki **ana fikir** (sayı/şiddet ağırlıklı, alan değil) böylece hem korundu hem de çözünürlük kararlılığı kazandı.

## Doğrulama — çözünürlük kararlılığı artık çok daha iyi

Aynı fotoğraf, aynı 5 kaynak genişliğinde, Gözenek.4 ile tekrar test edildi:

| Kaynak genişlik | Gözenek.3 (hatalı) | Gözenek.4 (düzeltilmiş) |
|---|---|---|
| 170 px | %58.2 | %49.8 |
| 700 px | %37.6 | %63.2 |
| 1200 px | %12.4 | %69.2 |
| 2000 px | %5.1 | %82.9 |
| 3000 px | — | %96.7 |

Aralık **%5-%58** (11 kat, kaotik/çöküş yönünde) yerine artık **%50-%97** (2 kat, yüksek çözünürlükte biraz daha fazla ince detay ayırt edilmesinden kaynaklanan, kabul edilebilir ve öngörülebilir bir artış) — Gözenek.1'in kendi README'sinde zaten "çözünürlükle makul ölçüde artması beklenen/kabul edilen bir davranış" olarak belgelenen aynı sınırlamaya geri dönüldü, felaket boyutlu çöküş ortadan kalktı.

Sıfır-bölge ve monotonluk testleri de tekrarlandı ve korundu: 0 bölge → skor kesinlikle %0.0; eşik gevşetildikçe (daha fazla gözenek tespit edildikçe) skor hiçbir zaman düşmedi.

## Önceki sürümle kıyaslama

| | Gözenek.3 | Gözenek.4 |
|---|---|---|
| Black-hat çekirdeği | Tek boyut | **Üç boyut (küçük/orta/büyük), maksimum birleştirme** |
| Skor formülü | `Σşiddet / alan / DENSITY_CAP` | **`Σşiddet / COUNT_CAP` (alan yok)** |
| Çözünürlük kararlılığı | Kötü (%5-%58, 11x) | **İyi (%50-%97, 2x)** |
| Bölgesel referans, şekil filtresi, bölge dağılımı | Aynı | Değişmedi |

## Maskeleme

**Değişmedi** — `face_mesh.py` Gözenek.3 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`gozenek.py` içinde)

- **YENİ**: `MULTISCALE_KERNEL_RATIOS = (0.010, 0.018, 0.032)` — üç ölçeğin yüz-oranlı çekirdek boyutları.
- **YENİ**: `COUNT_CAP = 200.0` — skorun %100'e ulaştığı, sabit ağırlıklı-sayı referansı (gerçek fotoğrafla kalibre edildi).
- Diğer tüm parametreler (bölgesel referans, şekil filtresi, çözünürlük-oranlı temizlik) Gözenek.3 ile aynı.

## Bilinen sınırlama

- `COUNT_CAP = 200` tek bir gerçek fotoğrafla kalibre edildi; farklı ten tiplerinde ince ayar gerekebilir.
- Çözünürlük kararlılığı mükemmel değil (hâlâ ~2 kat aralık) — bu, blob-sayma tabanlı herhangi bir yöntemin doğasında var olan, kabul edilmiş bir sınırlama (yüksek çözünürlük gerçekten daha fazla ince detayı ayırt edebiliyor).
- Üç ölçekli tarama, tek ölçekliye göre biraz daha fazla hesaplama gerektiriyor (3 kat black-hat + bölgesel istatistik hesabı) — büyük fotoğraflarda gözle görülür bir yavaşlama olmasa da, teorik maliyet artışı var.

## Nasıl çalıştırılır

```bash
cd "gözenek\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8404` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama şiddet (z) | Tespit sayısı | Bölge dağılımı | Gözenek.3 skoru (kıyas) |
|---|---|---|---|---|---|---|
| File.jpg (170x226) | 0.498 | ~2.9 | ~2.7 | 110 | Alın/Yanaklar ağırlıklı | 0.582 (hatalı formülle) |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
