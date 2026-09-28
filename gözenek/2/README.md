# Gözenek.2 — Bölgesel (Alın/Burun/Çene/Yanak Ayrı) Referans

## Gözenek.1'in sorunu

Gözenek.1, black-hat tepkisinin z-score'unu **tek bir global referansla** hesaplıyordu: taranan yüzün TAMAMI (alın+burun+çene+yanak birlikte) üzerinden tek bir ortalama/std. Ama gözenek görünürlüğü yüzün farklı bölgelerinde **anatomik olarak farklı** olan bir şey — burun ve alın (T-bölgesi) doğal olarak daha fazla yağ bezi içerir, gözenekleri **sağlıklı bir tende bile** yanaklardan daha büyük/belirgin görünür.

Tek bir global referans kullanmak, bu doğal farkı "anormallik" gibi değerlendirebiliyordu:
- T-bölgesi genel ortalamayı yukarı çekiyorsa, yanaklardaki **gerçek** bir gözenek sorunu, global std'nin şişirilmesi yüzünden olduğundan düşük z-score alıp kaçırılabiliyordu.
- Tam tersi, T-bölgesindeki tamamen normal/sağlıklı gözenekler bile global ortalamadan sapma gösterip gereksiz yere tespit edilebiliyordu.

## Bu sürümde ne değişti

Yüzün her alt-bölgesi (alın, burun, çene, yanaklar) artık **kendi içinde** değerlendiriliyor:

```
her bölge r için ayrı ayrı:
    z_r = (blackhat_piksel − bölge_r_ortalama) / bölge_r_std
```

"Bu burun pikseli, DİĞER BURUN piksellerine göre ne kadar gözenekli" ve "bu yanak pikseli, DİĞER YANAK piksellerine göre ne kadar gözenekli" ayrı ayrı soruluyor — Göz Altı Morluk ailesindeki "tek referans → daha doğru/stabil referans" ilerlemesiyle aynı ruhta.

Bir bölgenin piksel sayısı çok azsa (`MIN_REGION_PIXELS = 300`'ün altında — örneğin yüz kadrajda kısmen kesiliyorsa), o bölge güvenilmez kendi istatistiğine düşmek yerine **global referansa** düşüyor (güvenli varsayılan).

## Doğrulama — bölgesel istatistikler gerçekten farklı mı?

Aynı test fotoğrafında (`File.jpg`), black-hat tepkisinin gerçek istatistikleri ölçüldü:

| Bölge | Piksel sayısı | Ortalama | Std |
|---|---|---|---|
| **Global (tüm yüz)** | 142.506 | 1.275 | **3.428** |
| Alın | 50.348 | 1.226 | 3.267 |
| Burun | 764 | 0.899 | **1.758** |
| Çene | 2.836 | 0.799 | **1.736** |
| Yanaklar | 88.558 | 1.306 | 3.494 |

Burun ve çenenin **kendi doğal varyansı, global varyansın yaklaşık yarısı** (std≈1.7 vs 3.4) — yani bu bölgelerde normalde çok daha az dalgalanma var. Global referansla değerlendirildiğinde, bu bölgelerdeki gerçek bir sapma "küçük" görünüp kaçırılabiliyordu; bölgesel referansla artık doğru büyüklükte değerlendiriliyor. Somut örnek: burun bölgesinde ortalamadan 3 birim sapan bir piksel, **global std ile z≈0.88** (eşiğin altında, kaçırılır) verirken, **bölgesel std ile z≈1.71** (eşiğin üstünde, doğru tespit edilir).

Bu test fotoğrafında burun/çene bölgeleri toplam alanın küçük bir kısmını oluşturduğu için (764+2.836 px, toplam 142.506 px'in ~%2.5'i), toplam skor/tespit sayısı büyük ölçüde değişmedi (alın+yanaklar zaten baskın) — ama T-bölgesi daha büyük olan fotoğraflarda bu düzeltme daha belirgin fark yaratacaktır.

## Yeni: bölge dağılımı raporlaması

Artık kaç gözeneğin hangi yüz bölgesinde tespit edildiği de gösteriliyor (ör. "Alın: 65 · Yanaklar: 54") — bu, T-bölgesi varsayımının gerçekten doğru olduğunu görmek için de faydalı bir şeffaflık katmanı.

## Önceki sürümle kıyaslama

| | Gözenek.1 | Gözenek.2 |
|---|---|---|
| Referans | Tek global (tüm yüz) | **Bölgesel (alın/burun/çene/yanak ayrı)** |
| Küçük/eksik bölge | — | Güvenli global fallback (`MIN_REGION_PIXELS`) |
| Raporlama | Skor + tespit sayısı | **+ bölge dağılımı** |
| Tespit mantığı (black-hat, şekil filtresi, çözünürlük-oranlı boyutlandırma, lezyon-yükü skoru) | Aynı | **Değişmedi** |

## Maskeleme

`face_mesh.py`'ye **yeni** bir fonksiyon eklendi: `make_region_scan_masks()` — `forehead`/`nose`/`chin`/`cheeks` bölgelerinin her birini, göz/kaş/dudak dışlaması uygulanmış halde ayrı ayrı döndürür. Diğer tüm maskeleme mantığı (Gözenek.1 ile) değişmedi.

## Ayarlanabilir parametreler (`gozenek.py` içinde)

- Gözenek.1'deki tüm parametreler değişmedi.
- **YENİ**: `MIN_REGION_PIXELS = 300` — bir bölgenin kendi istatistiğinin güvenilir sayılması için gereken minimum piksel sayısı.

## Bilinen sınırlama

- Bu test fotoğrafında burun/çene bölgeleri küçük kaldığı için düzeltmenin AGREGE skora etkisi sınırlı kaldı — daha büyük/yakın çekilmiş fotoğraflarda (T-bölgesi daha büyük bir alan kapladığında) etkisi daha belirgin olacaktır.
- 4 bölge arasındaki sınırlar (convex hull tabanlı) kesin anatomik sınırlar değil; bir pikselin "hangi bölgeye ait olduğu" bazen sınırda belirsiz olabilir.

## Nasıl çalıştırılır

```bash
cd "gözenek\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8402` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama şiddet (z) | Tespit sayısı | Bölge dağılımı | Gözenek.1 skoru (kıyas) |
|---|---|---|---|---|---|---|
| File.jpg (170x226) | 0.145 | 2.35 | 2.73 | 119 | Alın: 65, Yanaklar: 54 | 0.147 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
