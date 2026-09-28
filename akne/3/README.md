# Akne.3 — Kırmızılık (a*) + Doygunluk (HSV S) Birleşimi

## Kullanıcının isteği

> "sivilcelere benziyor ya onları da akne saysın, düzgün çalışsın, daha iyi ayarlarmısın"

Yani: gerçek fotoğraflarda gözle görülür sivilce/leke gibi duran bölgelerin sistem tarafından **kaçırılmaması**, tespit edilen bölge sayısının açıkça gösterilmesi, ve tespitlerin **doğru** yerlerde (gerçekten sivilceye benzeyen küçük lekelerde, geniş yüz bölgelerinde değil) olması isteniyordu.

## Bulunan iki gerçek sorun (kullanıcının kendi fotoğrafıyla test edilerek tespit edildi)

Test için "test edilen görseller" klasöründeki gerçek bir yüz fotoğrafı (`File.jpg`, 170×226 piksel — küçük bir vesikalık boyutu) kullanıldı, Akne.2 ile çalıştırıldı:

### Sorun 1: düşük çözünürlük, tespiti neredeyse tamamen siliyordu

Akne ailesinin miras aldığı `preprocess.py`, sadece **büyük** fotoğrafları küçültüyordu, **küçük** fotoğrafları büyütmüyordu. 170×226 gibi düşük bir çözünürlükte, sabit boyutlu (3×3) morfolojik gürültü temizliği orantısız agresif kalıyordu:

- Eşiği aşan piksel sayısı: **914**
- Gürültü temizliği + kontur filtresinden sonra kalan bölge sayısı: **sadece 1**

Yani gerçek sivilceye benzeyen onlarca piksel grubu, sırf düşük çözünürlük yüzünden tek bir bölgeye eziliyordu.

### Sorun 2: tek kanal (a*) bazı gerçek sivilceleri kaçırıyordu

Akne.2 sadece a\* (kırmızılık) kanalına bakıyordu. Ama bazı gerçek sivilceler a\*'da çok yüksek çıkmadan, **doygunlukta** (HSV S — ne kadar "canlı/parlak" renk) belirgin şekilde öne çıkabiliyor.

## Bu sürümde ne değişti

### 1. `preprocess.py`'ye çözünürlük düzeltmesi eklendi

Artık `MIN_WORKING_SIZE = 700` altındaki fotoğraflar `INTER_CUBIC` ile büyütülüyor (aile içindeki diğer projelerin "sadece büyükleri küçült" mantığına **ek** olarak). Aynı test fotoğrafında:

- Önce: preprocess çıktısı 170×226 (değişmemiş) → 1 bölge
- Sonra: preprocess çıktısı 700×526 → morfolojik işlemler artık tutarlı çalışıyor

### 2. Kırmızılık + doygunluk birleşimi (Göz Altı Morluk.3 ile birebir aynı yapı)

```
z_a: (a - ten_a_ortalama) / ten_a_std          -> kırmızılık
z_s: (S - ten_S_ortalama) / ten_S_std          -> doygunluk/canlılık

combined_z = 0.6 * z_a + 0.4 * z_s
```

Bu bir **VE (AND)** şartı değil, **ağırlıklı toplam** — sadece kırmızılıkta veya sadece doygunlukta güçlü çıkan bir piksel de hâlâ yüksek `combined_z` alabiliyor. Bu, hassasiyeti **düşürmüyor**, iki farklı belirti türünü kapsayarak **artırıyor**.

### 3. YENİ SORUN ve çözümü: "sivilce şekli" filtresi

Çözünürlük düzeltmesi + iki kanal birleşimiyle ilk denemede tespit sayısı 1'den **28**'e çıktı — ama görsel kontrolde bunların çoğunun **burun köprüsü, göz altı, çene hattı gibi geniş anatomik bölgeleri** takip eden uzun/ince şekiller olduğu görüldü, gerçek sivilce değil. Sadece eşiği düşürmek/kanal eklemek yetmiyordu; sistem "kırmızı olan her yeri" değil, **sivilceye benzeyen bölgeleri** bulmalıydı.

Çözüm: her aday bölge için **dairesellik** (`4π×alan/çevre²`, 1.0 = mükemmel daire) ve **maksimum alan** kontrolü eklendi:

```
MIN_CIRCULARITY = 0.35        # uzun/ince şeritler (dairesellik düşük) elenir
MAX_SPOT_AREA_RATIO = 0.012   # taranan yüz alanının %1.2'sinden büyük bölgeler elenir
```

Gerçek bir sivilce şekli ne olursa olsun **yaklaşık yuvarlak bir leke**dir — yüzün doğal hatlarını (burun köprüsü, çene gölgesi, nazolabial kıvrım) takip eden uzun konturlar bu tanıma uymaz ve artık otomatik eleniyor.

## Doğrulama — gerçek fotoğrafta adım adım sonuç

Aynı test fotoğrafında (`test edilen görseller/File.jpg`), her değişiklik sırayla eklenip gerçek çalıştırmayla ölçüldü:

| Aşama | Tespit edilen bölge sayısı | Not |
|---|---|---|
| Akne.2 (orijinal çözünürlük, sadece a\*) | **1** | 914 aday piksel varken neredeyse tamamı morfolojik işlemde kayboldu |
| + çözünürlük düzeltmesi + a\*+S birleşimi + düşürülmüş eşik | **28** | Hassasiyet arttı ama burun köprüsü/çene hattı gibi geniş, sivilce-olmayan bölgeler de işaretlendi |
| + dairesellik/alan filtresi (bu sürümün nihai hali) | **15** | Geniş anatomik bölgeler elendi, kalanlar gerçekten küçük/yuvarlak lekeler |

Bu üç aşama, ısı haritası ve işaretlenmiş görsel karşılaştırmalarıyla görsel olarak da doğrulandı (geniş burun/çene konturları filtre sonrası kayboldu, küçük ben/leke işaretleri kaldı).

## Arayüzdeki değişiklik

Kullanıcı isteği doğrultusunda, tespit edilen bölge sayısı artık **ayrı, belirgin bir kutu** içinde gösteriliyor ("**15 adet sivilce/akne tespit edildi**" gibi), skorun hemen altında — önceki sürümlerde bu bilgi küçük bir alt yazıydı.

## Önceki sürümle kıyaslama

| | Akne.2 | Akne.3 |
|---|---|---|
| Kanal(lar) | Sadece a\* | **a\* + HSV S (birlikte)** |
| Küçük fotoğraf desteği | Yok (bozuk sonuç riski) | **Var (`MIN_WORKING_SIZE` büyütme)** |
| Şekil filtresi | Yok | **Var (dairesellik + maks. alan)** |
| Tespit eşiği | 1.2 | 1.0 (iki kanal birlikte kullanıldığı için gevşetildi) |
| Min. kontur alanı | 6 | 4 |

## Maskeleme

**Değişmedi** — `face_mesh.py`, Akne.2 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`akne.py` içinde)

- `WEIGHT_REDNESS = 0.6`, `WEIGHT_SATURATION = 0.4` — iki bileşenin ağırlıkları.
- `Z_SCORE_CAP = 2.5` — skorun 1.0'a ulaştığı birleşik z-score eşiği.
- `DETECTION_Z_THRESHOLD = 1.0` — işaretleme eşiği.
- `MIN_CONTOUR_AREA = 4` — gürültü temizliği.
- `MIN_CIRCULARITY = 0.35` — "sivilce şekli" sayılmak için minimum dairesellik.
- `MAX_SPOT_AREA_RATIO = 0.012` — taranan yüz alanına oranla maksimum sivilce boyutu.
- `preprocess.py` içindeki `MIN_WORKING_SIZE = 700` — küçük fotoğrafların büyütüleceği alt sınır.

## Bilinen sınırlama

- Dairesellik filtresi, birbirine çok yakın birden fazla sivilcenin **birleşip** uzun bir şekil oluşturduğu durumlarda (ör. yoğun bir akne kümesi) bu kümeyi tek seferde eleyebilir — bu, "az sayıda ama net" tespit ile "çok sayıda ama bazı kümelerin kaçırılması" arasında bilinçli bir denge.
- `MAX_SPOT_AREA_RATIO` sabit bir oran; çok yakın çekilmiş (yüzün kadrajı doldurduğu) fotoğraflarda büyük/şişkin tek bir sivilce yanlışlıkla elenebilir.

## Nasıl çalıştırılır

```bash
cd "akne\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8303` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | z_redness | z_saturation | Tespit edilen bölge sayısı | Akne.2 skoru (kıyas) | Not |
|---|---|---|---|---|---|---|
| File.jpg (170x226, gerçek yüz fotoğrafı) | 0.143 | 0.37 | 0.34 | 15 | (Akne.2: 1 bölge, çözünürlük sorunu) | Doğrulama testi, yukarıya bakınız |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
