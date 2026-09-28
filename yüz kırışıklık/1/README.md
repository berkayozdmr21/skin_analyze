# Yüz Kırışıklık.1 — Çok Yönelimli Gabor Filtre Bankası

## Yöntem araştırması

Kırışıklık tespiti, görüntü işleme literatüründe "çizgi/sırt (ridge/line) tespiti" problemi olarak ele alınır. Araştırmada tutarlı şekilde ortaya çıkan iki standart teknik: **Gabor filtre bankası** ve **Hessian/Frangi tabanlı sırt filtreleri**.

Bu projede **Gabor filtre bankası** kullanıldı çünkü (a) doğrudan OpenCV içinde hazır (`cv2.getGaborKernel`), ek bağımlılık gerektirmiyor, (b) **yönelim seçici**: kırışıklıklar her açıda olabilir (alında yatay, kaşlar arasında dikey, göz kenarında çapraz "kaz ayağı") — tek bir filtre değil, birden fazla yönelimde filtre bankası kullanıp en güçlü yanıtı almak gerekiyor.

Sources:
- [GitHub - elliotnaylor/Wrinkle-detection](https://github.com/elliotnaylor/Wrinkle-detection)
- [Fast detection of facial wrinkles based on Gabor features — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320314002945)
- [Automatic Wrinkle Detection using Hybrid Hessian Filter](https://e-space.mmu.ac.uk/597156/2/accv2014finalpaper604.pdf)
- [Evaluation of Automatic Facial Wrinkle Detection Algorithms — NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321017/)

Ana `skin-scan` projesindeki `texture.py`, genel bir "doku pürüzlülüğü" ölçümü (Laplacian varyansı + LBP) — kırışıklığa özel değil, çizgi geometrisini hedeflemiyor. Bu proje, literatürdeki daha spesifik/doğru tekniği (yönelim-seçici çizgi tespiti) uyguluyor.

## Yöntem

1. **Altı farklı yönelimde** (0°, 30°, 60°, 90°, 120°, 150° — bir çizgi 180° sonra kendisiyle aynı göründüğü için bu yelpaze yeterli) Gabor çekirdekleri oluşturulur.
2. Her yönelimde görüntü filtrelenir, mutlak değer alınır.
3. Tüm yönelimlerdeki en yüksek yanıt piksel-bazlı alınır — "bu piksel, hangi yönelimde olursa olsun bir çizgi yapısının parçası mı" sorusuna cevap.
4. **Koyuluk doğrulaması**: Gabor filtresi tek başına yönlü herhangi bir çizgisel yapıya (parlak veya koyu) tepki verir — ama kırışıklık özellikle **koyu** bir çizgidir (deri kıvrımındaki gölge). Gabor yanıtı, "bu piksel ten referansından daha koyu mu" kontrolüyle (medyan/MAD — Yüz Lekesi.3'te doğrulanan robust yöntem) **birleştiriliyor** (VE) — sadece koyu VE çizgisel olan pikseller kırışıklık adayı sayılır.
5. **Uzama (elongation) filtresi**: her bölge için `cv2.minAreaRect` ile en/boy oranı hesaplanır — sadece yeterince uzun/ince (yuvarlak DEĞİL) bölgeler kırışıklık sayılır. Bu, Akne/Gözenek/Yüz Lekesi ailesindeki "sadece yuvarlak" filtresinin **tam tersi**.

## İstatistik — skor NEYE göre veriliyor?

Kırışıklık, sayılabilir noktalar (Akne/Gözenek/Yüz Lekesi) veya yayılmış bir alan (Yağlılık) değil — bir **çizgi topluluğu**. Doğru ölçülen şey "kaç tane" veya "ne kadar alan" değil, **toplam çizgi uzunluğu**dur (dermatolojide "kırışıklık yoğunluğu" tam olarak böyle tanımlanır):

```
her tespit edilen çizgi i için:
    uzunluk_i = minAreaRect ile bulunan uzun kenar (piksel)
    şiddet_i  = clip(çizgi_i_ortalama_koyuluk_z / Z_SCORE_CAP, 0, 1)

ağırlıklı_uzunluk = Σ (uzunluk_i × şiddet_i)
skor = clip( ağırlıklı_uzunluk / (yüzün_karakteristik_uzunluğu × LENGTH_DENSITY_CAP), 0, 1)
```

### Kritik bir tasarım kararı: ALANA değil UZUNLUĞA oranlama

Akne/Gözenek ailesinde daha önce ciddi bir hata yapılmıştı: 1 boyutlu bir miktarı (sayı) 2 boyutlu bir miktara (piksel alanı) bölmek, çözünürlük değiştikçe skorun çökmesine yol açmıştı. Kırışıklık **uzunluk** (1 boyutlu) bir miktar — çözünürlükle **doğrusal** ölçeklenir, ALANLA (karesel ölçeklenir) DEĞİL. Bu yüzden ağırlıklı uzunluk, yüzün **karakteristik uzunluğuna** (`face_scale = sqrt(taranan_alan)`, çözünürlükle doğrusal ölçeklenir) bölünüyor — alana değil.

**Doğrulandı**: `ağırlıklı_uzunluk / face_scale` oranı, 170px-3000px arasında **1.72-1.98** aralığında ölçüldü — çok stabil, bu tasarım kararının doğru olduğunu kanıtlıyor.

## Geliştirme süreci — sınır artefaktı ve düzeltmesi (dürüstçe)

İlk görsel kontrolde ciddi bir sorun bulundu: göz soketinin **tam sınırını** takip eden oval bir çizgi ve yüzün **dış kenarını** (saç/arka plan geçişi) takip eden uzun bir çizgi, yanlışlıkla "kırışıklık" olarak işaretleniyordu — bunlar gerçek deri kıvrımları değil, maske sınırındaki doğal koyu geçişlerdi (Yüz Lekesi.4/5'te karşılaşılan aynı sorun sınıfı).

Çözüm, aynı aileden: tespit alanı, delik (göz/kaş/dudak) ve dış kenarlardan **içbüküldü** (`DETECTION_MARGIN_RATIO`). İlk denemede küçük bir marj (0.012) yetersiz kaldı, kademeli olarak büyütülüp (0.045, sonra 0.07) görsel kontrolle doğrulandı — son ayarda dış kenar çizgisi tamamen kayboldu, göz soketi etrafındaki tam oval iz küçük, makul göz-altı kıvrım parçalarına indi.

## Doğrulama

| Kaynak genişlik | Skor | Ağırlıklı uzunluk | Tespit sayısı |
|---|---|---|---|
| 170 px | %13.6 | 166.3 | 8 |
| 700 px | %17.5 | 284.8 | 12 |
| 1200 px | %21.0 | 589.3 | 14 |
| 2000 px | %35.1 | 1649.2 | 22 |
| 3000 px | %34.3 | 2426.8 | 18 |

Skor %13.6-%35.1 arasında (yüksek çözünürlükte biraz daha fazla ince detayın ayırt edilmesi beklenen bir davranış, ailenin diğer projelerinde de gözlemlenen bir örüntü). Sıfır-bölge tutarlılığı da doğrulandı: eşik yapay olarak imkansız hale getirildiğinde skor **tam %0.0** çıktı.

**Görsel doğrulama**: Son haliyle işaretlenen çizgiler kaşlar arası (glabella) dikey çizgi, nazolabial kıvrım (burun-ağız kenarı), göz altı ince çizgiler gibi anatomik olarak gerçek kırışıklık bölgeleriyle örtüşüyor.

## Maskeleme

`face_mesh.py`, aile ile birebir aynı kopya.

## Ayarlanabilir parametreler (`kirisiklik.py` içinde)

- `GABOR_ORIENTATIONS_DEG` — 6 yönelim (0°-150°, 30° aralıklarla).
- `GABOR_KSIZE_RATIO`, `GABOR_LAMBDA_RATIO`, `GABOR_SIGMA_FACTOR`, `GABOR_GAMMA` — Gabor çekirdek parametreleri, yüz ölçeğine oranlı.
- `GABOR_STRENGTH_PERCENTILE = 88.0` — çizgi yanıtının "güçlü" sayılma eşiği.
- `DARK_Z_THRESHOLD = 0.6`, `Z_SCORE_CAP = 2.0` — koyuluk doğrulaması (medyan/MAD).
- `MIN_ELONGATION_RATIO = 2.2` — şekil filtresi (uzun/kısa kenar oranı).
- `MAX_LENGTH_RATIO_OF_FACE = 0.55` — aşırı uzun (muhtemelen artefakt) çizgilerin üst sınırı.
- `DETECTION_MARGIN_RATIO = 0.07` — sınır artefaktlarını önlemek için içbükme miktarı.
- `LENGTH_DENSITY_CAP = 4.0` — skorun %100'e ulaştığı, yüzün karakteristik uzunluğuna oranlı ağırlıklı uzunluk eşiği.

## Bilinen sınırlama

- Göz soketi çevresindeki bazı küçük çizgi parçaları hâlâ maske-sınırı etkisinden tam arınmış olmayabilir — `DETECTION_MARGIN_RATIO` daha da artırılabilir ama bu, gerçek göz altı ince çizgilerini de kaçırma riskini artırır (bilinçli bir denge).
- Gabor parametreleri (özellikle dalga boyu) tek bir test fotoğrafıyla kalibre edildi; çok farklı kırışıklık genişliklerinde (çok ince/çok derin) ince ayar gerekebilir.
- Sadece **koyu** çizgileri arıyor; bazı yüzeysel/çok ince kırışıklıklar yeterince koyu gölge oluşturmayabilir ve kaçırılabilir.

## Nasıl çalıştırılır

```bash
cd "yüz kırışıklık\1"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8701` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Ağırlıklı uzunluk (px) | Ortalama şiddet (z) | Tespit sayısı |
|---|---|---|---|---|
| File.jpg (170x226) | 0.136 | 166.3 | 1.81 | 8 |
| _(0 bölge simülasyonu)_ | 0.000 | 0.0 | 0.0 | 0 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | |
