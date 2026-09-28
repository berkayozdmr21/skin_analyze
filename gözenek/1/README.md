# Gözenek.1 — Black-Hat Tabanlı Gözenek Tespiti

## Bu proje nedir

Göz Altı Morluk ve Akne ailelerindeki **aynı mimari** (MediaPipe FaceMesh → ten bölgesi maskesi → göz/kaş/dudak hariç tutma → ısı haritası + kontur işaretleme → FastAPI + tek sayfalık arayüz) kullanılarak, **gözenek (pore)** tespiti için kurulmuş ilk sürüm.

## Akne ailesinden fark: yöntem neden farklı bir kanal kullanıyor

Akne öncelikle bir **renk** (kırmızılık/iltihap) belirtisiyken, gözenekler öncelikle bir **doku/gölge** belirtisidir — ten yüzeyinde çevresinden biraz daha koyu, küçük ve yuvarlak noktalar. Bu yüzden a\*/S (renk) kanalları yerine klasik bir görüntü işleme tekniği kullanılıyor: **black-hat morfolojik dönüşüm** (ana `skin-scan` projesindeki `blemishes.py`'de de benzer bir top-hat/black-hat yaklaşımı vardı).

```
black-hat = closing(görüntü) − görüntü
```

"Closing" (genişletme sonra aşındırma) işlemi küçük KOYU noktaları doldurur/yok eder, büyük yapıları (kaş, burun gölgesi gibi) korur. Bu doldurulmuş versiyondan orijinali çıkarınca, geriye sadece "closing tarafından doldurulmuş" — yani çevresinden belirgin şekilde koyu VE küçük olan — noktalar kalır.

## İstatistik: z-score (Akne ailesiyle aynı ruhta)

Black-hat çıkışı ham/mutlak değerlerle değil, o fotoğrafa özgü z-score ile değerlendiriliyor:

```
z = (blackhat_piksel − ten_blackhat_ortalama) / ten_blackhat_std
```

Böylece farklı fotoğraf/ışık koşullarında "ne kadar black-hat tepkisi normal sayılır" otomatik kalibre oluyor.

## Baştan doğru kurulan iki tasarım kararı (Akne ailesinin sonradan öğrendiği dersler)

Akne ailesinde iki sorun **sonradan** düzeltildi (Akne.4 ve Akne.5). Gözenek.1'de bu ikisi **baştan itibaren** doğru kuruldu:

1. **Çözünürlükten bağımsız (göreli) eşikler.** Black-hat çekirdeği, gürültü temizleme çekirdeği ve minimum bölge alanı — hepsi taranan yüz alanının piksel sayısından türetilen bir ORAN, sabit piksel sayısı değil.
2. **Skor, doğrudan tespit edilen bölgelerden hesaplanıyor.** Akne.5'teki "lezyon yükü" mantığıyla aynı: `skor = Σ(bölge_şiddeti × bölge_alanı) / (yüz_alanı × 0.15)`. 0 bölge tespit edilirse skor **matematiksel olarak** kesinlikle %0'dır — Akne.4'te yaşanan "0 tespit ama %6 skor" tutarsızlığı burada yapısal olarak imkansız.

`BURDEN_AREA_FRACTION` gözenek için Akne'nin `0.05`'inden daha yüksek (`0.15`) tutuldu — çünkü normal, sağlıklı bir tende bile onlarca görünür gözenek olması beklenir; akne için geçerli olan "yüzün %5'i = ciddi" ölçütü gözenek için çok düşük/hassas kalırdı.

## Doğrulama — çözünürlük ve sıfır-bölge testleri

Aynı test fotoğrafı (`test edilen görseller/File.jpg`) farklı çözünürlüklerde çalıştırıldı:

| Kaynak genişlik | Black-hat çekirdeği | Skor | Kaplanan alan | Tespit sayısı |
|---|---|---|---|---|
| 170 px | 5 | %14.7 | %2.35 | 119 |
| 700 px | 7 | %16.6 | %2.64 | 138 |
| 1200 px | 13 | %18.2 | %3.06 | 154 |
| 2000 px | 21 | %17.8 | %2.87 | 161 |

Skor %14.7-%18.2 arasında kaldı (akne.3'ün ilk halindeki 13x'lik sapmayla kıyaslanınca oldukça stabil) — tam sabit değil (yüksek çözünürlükte gerçekten daha fazla ince detay ayırt edilebiliyor, bu beklenen/kabul edilen bir davranış) ama makul.

Sıfır-bölge tutarlılığı da doğrulandı: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde, skor **tam olarak %0.0** çıktı.

## Görsel doğrulama

Düşük çözünürlükten (170px) 4 kata yakın büyütülen görüntüde ısı haritası aşırı gürültülü/damarsı görünüyordu — bu, gerçek gözenek dokusundan değil, **çok küçük bir kaynak görüntünün agresif şekilde büyütülmesinden** kaynaklanan bir interpolasyon artefaktı olarak değerlendirildi. Daha yüksek native çözünürlükte (2000px kaynak) test edildiğinde, işaretlemeler kaş kenarları, göz altı, yanaklar, çene ve burun kenarları gibi gözeneklerin gerçekten belirgin olduğu bölgelerde toplanıyor — anatomik olarak makul bir dağılım.

## Maskeleme

`face_mesh.py`, Akne ailesiyle **birebir aynı kopya** — `make_face_scan_mask()` (tüm ten, göz/kaş/dudak hariç) gözenek için de doğru hedef bölge, çünkü gözenekler de akne gibi yüzün herhangi bir yerinde olabilir.

## Ayarlanabilir parametreler (`gozenek.py` içinde)

- `BLACKHAT_KERNEL_RATIO = 0.018`, `MIN/MAX_BLACKHAT_KERNEL` — black-hat çekirdeğinin göreli boyutu (gözenek boyutunu "hedeflemek" için).
- `Z_SCORE_CAP = 2.5`, `DETECTION_Z_THRESHOLD = 1.3` — skor/tespit eşikleri.
- `CLEAN_KERNEL_RATIO`, `MIN_AREA_RATIO` — gürültü temizliği, çözünürlüğe oranlı.
- `MIN_CIRCULARITY = 0.3`, `MAX_SPOT_AREA_RATIO = 0.0025` — şekil filtresi (akne'dekinden daha küçük üst sınır, çünkü bir gözenek bir sivilceden çok daha küçük olmalı).
- `BURDEN_AREA_FRACTION = 0.15` — skorun %100'e ulaştığı "gözenek yükü" referansı.

## Bilinen sınırlama

- Çok düşük çözünürlüklü (ör. 170px) kaynak fotoğraflarda, agresif büyütme interpolasyon dokusu yaratıp gerçek gözenek sinyaliyle karışabiliyor — ısı haritası bu durumda gerçekte olduğundan daha "gürültülü" görünebilir, ama tespit sayısı/skor yine de makul aralıkta kalıyor (bkz. doğrulama tablosu).
- Black-hat, doğası gereği KOYU noktalara bakar — çok açık tenli/parlak (highlight) bölgelerdeki gözenekler (ışığın vurduğu yerlerde) daha az belirgin çıkabilir. Bu, akne ailesindeki "sadece pozitif yön" mantığıyla aynı bilinçli sınırlama.
- `BURDEN_AREA_FRACTION = 0.15` ampirik bir sabit; farklı ten tiplerinde/yaşlarda (gözenek görünürlüğü doğal olarak değişir) ince ayar gerekebilir.

## Nasıl çalıştırılır

```bash
cd "gözenek\1"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8401` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama şiddet (z) | Tespit sayısı | Not |
|---|---|---|---|---|---|
| File.jpg (170x226) | 0.147 | 2.35 | 2.79 | 119 | Doğrulama testi, yukarıya bakınız |
| _(0 bölge simülasyonu)_ | 0.000 | 0.00 | 0.00 | 0 | Skor/tespit tutarlılığı doğrulandı |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
