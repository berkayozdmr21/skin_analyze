# Göz Altı Morluk.4 — Mahalanobis Mesafesi (Robust, Çok Değişkenli)

## Önceki sürümlerin istatistiksel zayıflığı

Göz Altı Morluk.3, L* (koyuluk) ve b* (mavi-mor ton) kanallarını **ayrı ayrı** z-score'a çevirip, **sabit/elle seçilmiş ağırlıklarla** (`0.6 × z_L + 0.4 × z_b`) topluyordu. Bunun iki sorunu var:

1. **Korelasyon göz ardı ediliyor.** Kanallar arasında doğal bir ilişki varsa (örneğin bir kişinin teninde L* düşükken b*'nin de hafifçe düşük olma eğilimi varsa), bağımsızmış gibi toplamak sapmayı **olduğundan fazla büyütür** (aynı bilgiyi iki kere saymak gibi).
2. **Ağırlıklar veriden gelmiyor.** `0.6` / `0.4` sayılarının "doğru" olduğunu gösteren istatistiksel bir dayanak yok — deneme-yanılmayla seçilmiş.
3. Ten referans istatistiği (ortalama/std) **ham** hesaplanıyordu — ten maskesine kaçan birkaç aykırı piksel (parlama, kirpik gölgesi, saç teli) referansı bozabilirdi. Bu risk M.2/M.3'ün README'sinde zaten not edilmişti ama çözülmemişti.

## Bu sürümde ne değişti

### 1. Ayrı z-score toplamı → Mahalanobis mesafesi

Artık L*, a*, b* kanalları **tek bir 3-boyutlu vektör** olarak ele alınıyor. Ten bölgesinin gerçek kovaryans yapısı (`Σ`, kanallar arası korelasyonu da içeren 3×3 matris) çıkarılıp, her piksel için:

```
diff = ten_ortalama_vektörü − piksel_vektörü        (3 boyutlu: L, a, b)
D    = sqrt( diffᵀ · Σ⁻¹ · diff )
```

Bu, istatistikte "bir noktanın çok değişkenli bir dağılımın merkezinden kaç birim uzakta olduğunu" ölçmenin standart yoludur (Mahalanobis mesafesi). Kanallar arası korelasyon otomatik olarak hesaba katılır — elle seçilmiş ağırlıklara gerek kalmaz.

### 2. a* kanalı da eklendi

M.3 sadece L* ve b* kullanıyordu. Gerçek bir morlukta (damarların ince ciltten görünmesi) sadece maviye değil, hafifçe **kırmızı/magenta** yönüne de bir kayma olabilir (mor = mavi + kırmızı karışımı). a* kanalı da vektöre eklenip `Σ`'ya dahil edilerek, varsa bu bilgi de mesafeye yansıtılıyor.

### 3. Robust (winsorized) ten istatistiği

Ten bölgesinin ortalama/kovaryansı artık ham veriden değil, her kanal **%5–%95 persentillerine kırpılmış (winsorize edilmiş)** veriden hesaplanıyor. Böylece maskeye kaçan birkaç aşırı piksel referansı bozamıyor.

### 4. Yön kısıtlaması korundu

Mahalanobis mesafesi yönsüzdür — parlak bir piksel de yüksek mesafe verebilir. M.3'teki gibi, sadece **"tenden daha koyu (L düşük) VE daha mavi (b düşük)"** yönündeki sapmalar sayılıyor; aksi yöndeki sapmalar (örn. parlama) skora katkı yapmıyor.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(yön_kısıtlı_D[göz_altı_bandı]) / MAHALANOBIS_CAP, 0, 1 )
```

`MAHALANOBIS_CAP = 3.37`, z-score'daki "2.5 sigma" mantığının çok değişkenli karşılığı: 3 serbestlik dereceli bir ki-kare (χ²) dağılımında, D² değerinin **%99 persentili ≈ 11.3**'tür, yani D ≈ 3.37. Yani "3 kanalda birlikte, tesadüfen oluşma ihtimali %1'den küçük bir sapma" — M.2/M.3'teki 2.5σ'nın (tek kanalda %99'luk dilim dışı) doğal, çok değişkenli genellemesi.

**Tespit işaretleme**: `DETECTION_THRESHOLD = 2.15` (χ²(3) dağılımının %80 persentiline karşılık gelen mesafe) aşan pikseller "tespit edilen bölge" sayılıp sarı/magenta kontur ile işaretleniyor.

**Şeffaflık**: Arayüzde skorun yanında hem toplam Mahalanobis mesafesi hem de her kanalın (L, a, b) tek-kanallı ortalama sapması ayrı ayrı gösteriliyor — "bu bölge neden yüksek çıktı" sorusuna M.3'tekiyle aynı ruhta, ama 3 kanala genellenmiş bir cevap.

## Önceki sürümlerle kıyaslama

| | M.1 | M.2 | M.3 | M.4 |
|---|---|---|---|---|
| Kullanılan kanal(lar) | Sadece L* | Sadece L* | L* + b* (ayrı ayrı) | **L* + a* + b* (birlikte, kovaryanslı)** |
| Kanallar arası korelasyon | — | — | Yok sayılıyor | **Hesaba katılıyor (Σ⁻¹)** |
| Birleştirme | — | — | Sabit ağırlık (0.6/0.4) | **İstatistiksel (Mahalanobis)** |
| Referans istatistiği | Ortalama | Ortalama + std | Ortalama + std (ham) | **Winsorize edilmiş (aykırı değere dayanıklı)** |
| Ölçek | Sabit (30 L\*) | Z-score (2.5σ) | Z-score (2.5σ) | **Mahalanobis (χ²(3) tabanlı, 3.37)** |

## Bilinen sınırlama

Kovaryans matrisi ten bölgesindeki piksel sayısına ve o bölgenin renk çeşitliliğine duyarlıdır — çok küçük/düz bir ten bölgesi (örn. yüz büyük kısmı gölgedeyse) kovaryansı güvenilmez kılabilir; bunu önlemek için matrise küçük bir regularizasyon (`COV_REG_EPS`) eklendi ama bu tam bir çözüm değil. Ayrıca yön kısıtlaması hâlâ sadece L ve b eksenlerine bakıyor — a* ekseni mesafeye katkı yapıyor ama tek başına yön şartı olarak kullanılmıyor (çok sıkı bir şart, gerçek pozitif vakaları kaçırma riski taşır).

## Maskeleme

**Değişmedi** — `face_mesh.py` ve `preprocess.py`, Göz Altı Morluk.3 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`morluk.py` içinde)

- `MAHALANOBIS_CAP = 3.37` — skorun 1.0'a ulaştığı Mahalanobis mesafesi (χ²(3), %99).
- `DETECTION_THRESHOLD = 2.15` — işaretleme eşiği (χ²(3), %80).
- `WINSOR_PERCENTILES = (5, 95)` — ten referans istatistiği için aykırı değer kırpma aralığı.
- `COV_REG_EPS = 1e-3` — kovaryans matrisinin tersini alabilmek için küçük regularizasyon.
- `MIN_CONTOUR_AREA = 20` — gürültü temizliği.

## Nasıl çalıştırılır

```bash
cd "göz altı morluk\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8204` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Mahalanobis D | z_L | z_a | z_b | Tespit sayısı | M.3 skoru (kıyas) | Not |
|---|---|---|---|---|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | | | | | | |
