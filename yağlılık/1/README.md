# Yağlılık.1 — HSV Specular Highlight (Parlama) Tespiti

## Yöntem araştırması

Cilt yağlılığını OpenCV ile tespit etmenin standart yöntemini araştırdım. Hem web araştırması hem de masaüstündeki **ana `skin-scan` projesinde zaten mevcut olan** `src/pipeline/maps/oiliness.py` modülü aynı tekniğe işaret ediyor: **specular highlight (aynasal/parlak yansıma) tespiti**.

**Temel prensip:** Yağlı/parlak ten, üzerine düşen ışığı neredeyse doğrudan yansıtır (aynasal yansıma) — bu da HSV renk uzayında **yüksek V (parlaklık) + düşük S (doygunluk)** olarak kendini gösterir (yansıma noktası, rengin "yıkanıp" beyaza yakınlaşmasına neden olur). Bu, fotoğrafçılık ve görüntü işlemede "specular highlight" olarak bilinen klasik bir optik özellik.

Bunu güvenilir kılmak için tek başına HSV eşiklemesi yeterli değil (beyaz bir nesne veya aşırı parlak arka plan da aynı kritere uyabilir) — bu yüzden **gradyan büyüklüğü** (Sobel) ile doğrulanıyor: gerçek bir "parlama" ten üzerinde **yumuşak/düz** bir geçiş yapar (düşük gradyan), keskin kenarlı parlak bölgeler nesne sınırı olabilir, parlama değildir.

Sources:
- [GitHub - DurtyDhiana/skin-scan](https://github.com/DurtyDhiana/skin-scan)
- [Specular Highlights detection - OpenCV Q&A Forum](https://answers.opencv.org/question/92887/specular-highlights-detection/)
- [Tutorial: Skin Detection Example using Python and OpenCV — PyImageSearch](https://pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/)

Ana projedeki `oiliness.py` da tam olarak bu yöntemi (HSV eşikleme + Sobel gradyan doğrulaması) kullanıyor — bu, hem literatürle hem de kullanıcının kendi projesiyle tutarlı, kanıtlanmış bir teknik olduğunu doğruladı.

## Bu proje ailesinden iki önemli fark

### 1. Koyu değil, PARLAK bölgeler aranıyor

Göz Altı Morluk/Akne/Gözenek/Yüz Lekesi hep **koyu** lekeler arıyordu (düşük V veya yüksek "delta koyuluk"). Yağlılık tam tersine **parlak** bölgeler arıyor (yüksek V) — kavramsal olarak "top-hat" mantığında, ama burada doğrudan HSV eşikleme + gradyan doğrulaması ile yapılıyor (literatürdeki standart yöntem, black-hat/top-hat morfolojisi değil).

### 2. Skor, SAYI değil ALAN YÜZDESİ

Akne/Gözenek/Yüz Lekesi'nde skor, tespit edilen **ayrık** lekelerin sayısı/şiddetinden hesaplanıyordu — çünkü o lekeler doğası gereği küçük, ayrı, sayılabilir noktalardı. Yağlılık/parlama **farklı bir fiziksel doğaya** sahip: T-bölgesi boyunca birbirine bağlanan, düzensiz/yaygın bir **örtüşme (coverage)** olgusu — "kaç tane parlama var" sorusu pek anlamlı değil. Bu yüzden skor burada **kaplanan alan yüzdesine** dayanıyor:

```
skor = clip( (yağlı_piksel_sayısı / taranan_yüz_alanı) / OILY_COVERAGE_CAP, 0, 1)
```

## Sabit eşikten yüzdelik dilime geçiş — gerçek test fotoğrafıyla bulunan sorun

Ana projedeki `oiliness.py`, **sabit** eşikler kullanıyordu (V>200, S<80). Bu proje de başlangıçta aynı sabitlerle kuruldu — ama gerçek test fotoğrafında (`File.jpg`) **medyan V değerinin zaten 202** olduğu görüldü! Yani sabit "V>200" eşiği, bu fotoğrafta **yüzün neredeyse yarısını** "yağlı" sayıyordu (ısı haritası neredeyse tüm yüzü kaplıyordu — açıkça yanlış kalibrasyon, çünkü fotoğraflar arası genel pozlama/parlaklık çok değişebilir).

Bu proje ailesinde defalarca işe yarayan çözüm burada da uygulandı: **sabit değer yerine, o fotoğrafın kendi parlaklık dağılımına göre yüzdelik dilim eşiği**:

```
V_PERCENTILE = 92.0   # yüzün en parlak %8'i
S_PERCENTILE = 35.0   # yüzün en az doygun %35'i
```

"Bu yüzün en parlak %8'i VE en az doygun %35'i" — hangi fotoğrafta olursa olsun, o fotoğrafın **kendi** özel parlama noktalarını buluyor, sabit bir sayıya güvenmek yerine.

## Doğrulama

Aynı test fotoğrafı farklı çözünürlüklerde çalıştırıldı:

| Kaynak genişlik | Skor | Kaplanan alan | Tespit sayısı |
|---|---|---|---|
| 170 px | %55.7 | %4.46 | 20 |
| 700 px | %63.8 | %5.10 | 20 |
| 1200 px | %64.9 | %5.20 | 23 |
| 2000 px | %59.7 | %4.78 | 24 |
| 3000 px | %59.1 | %4.73 | 23 |

Kaplanan alan yüzdesi **%4.5-5.2** arasında kaldı — çok stabil. Sıfır-bölge tutarlılığı da doğrulandı: eşik yapay olarak imkansız hale getirilip (yüzdelik dilim %100'e çekilerek) 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

**Görsel doğrulama:** Tespit edilen bölgeler kaşlar arası (glabella), burun köprüsü, çene gibi klasik "T-bölgesi" parlama noktalarında yoğunlaştı — cilt bakımı/dermatoloji literatüründe yağlılığın en çok göründüğü bölgelerle örtüşüyor, anatomik olarak makul bir sonuç.

## Maskeleme

`face_mesh.py`, aile ile birebir aynı kopya — `make_face_scan_mask()` (tüm ten, göz/kaş/dudak hariç) burada da doğru hedef bölge.

## İşaretleme

Bu projeye özgü **yeni** bir görselleştirme: `draw_filled_regions()` — parlama bölgeleri, önceki ailedeki gibi numaralı dairelere sadeleştirilmiyor, çünkü doğaları gereği düzensiz/yaygın şekilli. Bunun yerine tespit edilen konturların **gerçek şekli** yarı saydam renkle dolduruluyor.

## Ayarlanabilir parametreler (`yaglilik.py` içinde)

- `V_PERCENTILE = 92.0`, `S_PERCENTILE = 35.0` — parlama eşiğinin yüzdelik dilim referansları.
- `GRADIENT_THRESHOLD = 50.0` — "yumuşak/düz" sayılma sınırı (Sobel gradyan büyüklüğü).
- `CLEAN_KERNEL_RATIO`, `MIN_AREA_RATIO` — gürültü temizliği, çözünürlüğe oranlı.
- `OILY_COVERAGE_CAP = 0.08` — skorun %100'e ulaştığı kaplanan alan oranı.

## Bilinen sınırlama

- Gözlük camı yansıması, terleme parıltısı gibi cilt-dışı parlak yüzeyler de (eğer yüz maskesi sınırları içine düşerse) yanlışlıkla "yağlılık" sayılabilir.
- `V_PERCENTILE=92`/`S_PERCENTİLE=35` tek bir gerçek fotoğrafla kalibre edildi; çok farklı ışık koşullarında (çok karanlık/çok parlak ortam) ince ayar gerekebilir.
- Bu yöntem doğrudan "yağ miktarını" değil, ışığın cilt üzerindeki **yansıma davranışını** ölçüyor — kuru ama parlak (örn. nemlendirici sürülmüş) bir cilt de yanlışlıkla yağlı sayılabilir.

## Nasıl çalıştırılır

```bash
cd "yağlılık\1"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8601` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Tespit sayısı | Not |
|---|---|---|---|---|
| File.jpg (1024px) | 0.649 | 5.20 | 23 | Doğrulama testi, yukarıya bakınız |
| _(0 bölge simülasyonu)_ | 0.000 | 0.00 | 0 | Skor/tespit tutarlılığı doğrulandı |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | |
