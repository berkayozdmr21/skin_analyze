# Gözenek.3 — Sayı (Yoğunluk) Tabanlı Skor

## Kullanıcının bildirdiği iki sorun

> "tutarlı değil, gözenek arttıkça skor artsın ve yüzdeleri az gibi geliyor, arttır"

## Kök neden

Gözenek.2'de skor, **alan-ağırlıklı bir "yük"** idi:

```
skor = Σ(bölge_i_şiddeti × bölge_i_ALANI) / (yüz_alanı × 0.15)
```

Bu formül **PİKSEL ALANINA** göre ağırlıklıydı, **SAYIYA** göre değil. Sonuç iki sorun:

1. **Tutarsızlık.** Az sayıda ama BÜYÜK gözenek/leke, çok sayıda ama KÜÇÜK gözenekten daha yüksek skor verebiliyordu — kullanıcının sezgisel beklentisi ("daha çok gözenek = daha yüksek skor") ile bu davranış uyuşmuyordu.
2. **Düşük yüzdeler.** `BURDEN_AREA_FRACTION = 0.15` ("yüzün %15'i kaplansa skor %100") gerçekçi hiçbir fotoğrafta yaklaşılamayacak kadar yüksek bir eşikti — gerçek bir yüzde görünür gözeneklerin kapladığı alan yüzdesi genelde %1-3 civarında kalıyor, bu da skoru sistematik olarak düşük gösteriyordu.

## Somut örnekle sorunun kanıtı

Varsayımsal iki senaryo (aynı ortalama şiddet=0.6, farklı boyut/sayı dağılımı):

| Senaryo | Gözenek sayısı | Her birinin alanı | Gözenek.2 (alan-ağırlıklı) | Gözenek.3 (sayı-ağırlıklı) |
|---|---|---|---|---|
| A: çok sayıda küçük gözenek | 50 | 20 px | yük = 50×20×0.6 = **600** | ağırlıklı sayı = 50×0.6 = **30** |
| B: az sayıda büyük gözenek | 5 | 300 px | yük = 5×300×0.6 = **900** | ağırlıklı sayı = 5×0.6 = **3** |

Gözenek.2'de **B, A'dan daha yüksek skor alıyordu** — 10 kat daha az gözenek olmasına rağmen! Gözenek.3'te bu düzeltiliyor: A (daha fazla gözenek) doğru şekilde daha yüksek skor alıyor.

## Bu sürümde ne değişti: skor artık SAYI (yoğunluk) tabanlı

Her tespit edilen gözenek, **alanı ne olursa olsun**, kendi şiddetiyle (0-1 arası, z-score'dan) toplama katkıda bulunuyor:

```
ağırlıklı_sayı = Σ şiddet_i                              [alan YOK, sadece şiddet]
yoğunluk       = ağırlıklı_sayı / (taranan_yüz_alanı / 100)   [yüzde-1'lik alana düşen ağırlıklı sayı]
skor           = clip(yoğunluk / 0.20, 0, 1)
```

**Önemli garanti:** Her tespit edilen gözenek, `DETECTION_Z_THRESHOLD`'u geçtiği için şiddeti her zaman **pozitif**tir. Bu yüzden yeni bir gözenek tespiti toplamı **kesinlikle artırır, asla azaltmaz** — "daha fazla gözenek → skor hiçbir zaman düşmez" garantisi artık matematiksel.

`DENSITY_CAP = 0.20` gerçek test fotoğrafıyla kalibre edildi: gözle görülür ölçüde gözenekli bir yüzde (119 tespit, ortalama şiddet ~0.91) yoğunluk ~0.116 çıkıyordu — bu da **%58**'e karşılık geliyor (Gözenek.2'deki %14.5 yerine).

## Doğrulama — monotonluk testi (gerçek fotoğrafla)

Aynı fotoğrafta tespit eşiği kademeli gevşetilerek daha fazla gözenek tespit edilmesi sağlandı, skorun tepkisi ölçüldü:

| Eşik | Tespit sayısı | Skor |
|---|---|---|
| 1.60 (sıkı) | 90 | %46.8 |
| 1.45 | 91 | %47.3 |
| 1.30 (varsayılan) | 119 | %58.2 |
| 1.15 | 121 | %59.0 |
| 1.00 (gevşek) | 148 | %65.5 |

Tespit sayısı arttıkça skor **kesintisiz olarak arttı** — hiçbir adımda düşmedi. Bu, kullanıcının istediği tutarlılığı doğruluyor.

Sıfır-bölge tutarlılığı da korundu: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

## Önceki sürümle kıyaslama

| | Gözenek.2 | Gözenek.3 |
|---|---|---|
| Skorun kaynağı | Alan-ağırlıklı yük (`Σ şiddet×alan`) | **Sayı-ağırlıklı yoğunluk (`Σ şiddet`)** |
| Daha fazla tespit → skor | Garanti yok (alana bağlı) | **Garanti: hiçbir zaman düşmez** |
| Aynı fotoğrafta tipik skor | %14.5 | **%58.2** |
| Tespit mantığı (black-hat, bölgesel referans, şekil filtresi) | Aynı | **Değişmedi** |

## Maskeleme ve tespit mantığı

**Değişmedi** — `face_mesh.py` ve tespit süreci (`_detect`) Gözenek.2 ile birebir aynı; sadece skor formülü (`pore_score`) değişti. `affected_area_pct` hâlâ raporlanıyor (şeffaflık için) ama artık skoru belirlemiyor.

## Ayarlanabilir parametreler (`gozenek.py` içinde)

- Gözenek.2'deki tespit parametreleri (`BLACKHAT_KERNEL_RATIO`, `Z_SCORE_CAP`, `DETECTION_Z_THRESHOLD`, `MIN_CIRCULARITY`, `MAX_SPOT_AREA_RATIO`, `MIN_REGION_PIXELS` vb.) — değişmedi.
- **YENİ**: `DENSITY_CAP = 0.20` — skorun %100'e ulaştığı, "yüzün %1'lik alanına düşen ağırlıklı gözenek sayısı" referansı.
- `BURDEN_AREA_FRACTION` (Gözenek.2'deki) artık kullanılmıyor, kaldırıldı.

## Bilinen sınırlama

- `DENSITY_CAP = 0.20` tek bir gerçek fotoğrafla kalibre edildi — farklı ten tiplerinde/yaşlarda (gözenek görünürlüğü doğal olarak değişir) çok yüksek veya çok düşük skorlar gözlemlenirse bu sabitin yeniden ayarlanması gerekebilir.
- Skor artık alanı hiç hesaba katmadığı için, tek bir ÇOK BÜYÜK (ör. genişlemiş/tıkanmış) gözenek ile birçok küçük normal gözenek aynı "1 adet" ağırlığında sayılabilir — şiddet (z-score) bunu kısmen telafi ediyor (daha belirgin/derin bir gözenek genelde daha yüksek z alır) ama boyut farkını doğrudan yansıtmıyor.

## Nasıl çalıştırılır

```bash
cd "gözenek\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8403` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama şiddet (z) | Tespit sayısı | Bölge dağılımı | Gözenek.2 skoru (kıyas) |
|---|---|---|---|---|---|---|
| File.jpg (170x226) | 0.582 | 2.35 | 2.73 | 119 | Alın: 65, Yanaklar: 54 | 0.145 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
