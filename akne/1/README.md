# Akne.1 — Yanak Referanslı Kırmızılık (a*) Farkı

## Bu proje nedir

**Göz Altı Morluk.1** ile **birebir aynı mimari ve mantık** kullanılarak, akne/sivilce/kızarıklık benekleri tespiti için uyarlanmış ilk (en basit) sürüm. Masaüstündeki `göz altı morluk` klasöründeki projelerle aynı aile: her sürümde maskeleme mantığı sabit kalıp, sadece hesaplama (istatistik) değişecek — bu 1. sürüm, ailenin en basit/sabit ölçekli üyesi.

## Yöntem

1. Lab renk uzayına çevrilir, **a\*** (yeşil↔kırmızı ekseni) kanalı alınır — göz altı morluk'un L* (parlaklık) kanalı yerine, çünkü akne öncelikle bir **koyuluk** değil, bir **renk (kızarıklık/iltihap)** belirtisi.
2. **Referans bölge**: yanaklar (`cheeks` — yüzün sol/sağ yarısının convex hull'u, göz altı morluk ailesindeki aynı mantık). Yanağın a\* ortalaması (`cheek_a_mean`) hesaplanır — "bu kişinin tipik/iltihaplanmamış ten rengi" örneği olarak.
3. **Hedef (taranan) bölge**: göz altı morluk'ta hedef sabit ve dar bir bandı (göz altı) kapsıyordu, çünkü morluk anatomik olarak hep aynı yerde oluşur. **Akne yüzün herhangi bir yerinde çıkabilir** — bu yüzden hedef bölge, taranabilir **tüm yüz cildi** (alın+burun+çene+yanak birleşimi).
4. **YENİ — dışlama (exclusion) maskesi**: Hedef bölge tüm yüze genişleyince, doğal olarak akne OLMAYAN ama akneye benzer biçimde "kırmızı"/farklı renkte olan bölgeler (başta **dudaklar**, ayrıca **gözler** ve **kaşlar**) devreye giriyor. Bunlar dışarıda bırakılmazsa örneğin dudaklar doğrudan "akne" olarak işaretlenir. `face_mesh.py`'deki `make_exclusion_mask()` bu bölgeleri (göz+kaş+dudak, kenarlarda pay bırakmak için hafifçe genişletilmiş/`dilate` edilmiş) hedef bölgeden çıkarıyor.
5. Her piksel için:
   ```
   delta = a(piksel) - cheek_a_mean
   ```
   Pozitif ve büyük değer = o piksel, yanaktan **belirgin şekilde daha kırmızı** → akne/kızarıklık adayı.
6. **Isı haritası**: `delta` (sadece pozitif kısmı) `[0, 18]` aralığında `[0,1]`'e ölçeklenir, taranan bölge dışında sıfırlanır.
7. **Skor**: taranan bölgedeki ortalama pozitif delta, aynı şekilde ölçeklenir.

## Tespit edilen bölgeleri işaretleme

Göz altı morluk.1'deki gibi, sadece bir ısı haritası/skor değil, **`delta > 7` (a\* birimi)** olan pikseller ayrıca:
1. İkili (binary) maskeye çevrilir, taranan yüz bölgesi ile kesişimi alınır.
2. Küçük gürültü noktaları morfolojik açma (`MORPH_OPEN`) ile temizlenir.
3. `cv2.findContours` ile bu bölgelerin **konturları** bulunur (alanı 6 pikselden küçük olanlar elenir — akne benekleri göz altı torbalarından küçük olduğu için göz altı morluk.1'deki 20 yerine daha küçük bir eşik).
4. Her kontur, orijinal fotoğraf üzerine **sarı kontur çizgisi + dikdörtgen çerçeve** ile çizilir.

Arayüzde bu, ısı haritasının yanında **ikinci bir görsel** olarak (`Tespit Edilen Akne Bölgeleri`) gösteriliyor, ayrıca kaç ayrı bölgede tespit yapıldığı da yazıyor.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(clip(a - cheek_a_mean, 0, None)[taranan_bölge]) / 18, 0, 1 )
```

`18` değeri ampirik — dermatoloji/eritem çalışmalarında referans alınan aralıklara yakın bir üst sınır olarak seçildi (bkz. `akne.py` içindeki `DELTA_VISUAL_CAP`), göz altı morluk.1'deki `DELTA_VISUAL_CAP=30` mantığının a* kanalına uyarlanmış hali.

**Tespit işaretleme** ayrı bir eşikle (`DETECTION_THRESHOLD = 7`) çalışıyor — bu, skoru etkilemiyor, sadece "somut olarak nerede bulundu" görselleştirmesi için.

## Göz Altı Morluk.1 ile kıyaslama

| | Göz Altı Morluk.1 | Akne.1 |
|---|---|---|
| Kanal | L* (koyuluk) | **a\* (kırmızılık)** |
| Referans bölge | Yanak | Yanak (**değişmedi**) |
| Hedef bölge | Dar bant (göz altı) | **Tüm yüz cildi (alın+burun+çene+yanak)** |
| Dışlama gerekiyor mu | Hayır (hedef zaten dudak/kaştan uzak) | **Evet — göz+kaş+dudak dışlanmalı** |
| Min. kontur alanı | 20 px (torba büyüklüğünde) | **6 px (sivilce büyüklüğünde)** |

## Maskeleme

- `cheeks`: göz altı morluk ailesindeki aynı yöntem (yüzü ortadan ikiye bölüp her yarının convex hull'u) — referans bölgesi olarak.
- `forehead`/`nose`/`chin`/`cheeks` birleşimi (`make_skin_tone_mask`): göz altı morluk.2-5'teki "tüm ten" mantığıyla aynı — hedef bölgenin temeli.
- `make_exclusion_mask`: **bu projeye özgü, yeni eklenen** — göz (tam kontur) + kaş + dudak (dış kontur) convex hull'larının birleşimi, kenar payı için hafifçe genişletilmiş (`dilate`).
- `make_face_scan_mask`: ten birleşimi **eksi** dışlama maskesi = nihai taranan bölge.

## Ayarlanabilir parametreler (`akne.py` içinde)

- `DELTA_VISUAL_CAP = 18.0` — skor/ısı haritası ölçeklemesi için üst sınır (a\* birimi).
- `DETECTION_THRESHOLD = 7.0` — bir pikselin "tespit edilen akne noktası" sayılması için gereken minimum delta.
- `MIN_CONTOUR_AREA = 6` — gürültü temizliği için minimum kontur alanı (piksel).
- `face_mesh.py` içindeki `dilate_px = 9` — dışlama maskesinin kenar payı (piksel).

## Bilinen sınırlama

- Sadece **kırmızılığa** bakıyor; koyu/iltihapsız komedonlar (siyah nokta/beyaz nokta) veya iyileşmekte olan, artık kızarık olmayan izler bu yöntemle yakalanmaz — bu, göz altı morluk ailesinin sonraki sürümlerinde (z-score, çok kanallı, vb.) olduğu gibi ileride ele alınabilir.
- Test görsellerinde doğrulandığı üzere, **ben/çil gibi pigmentasyon lekeleri** de bazen a* değerini yükselttiği için yanlışlıkla "akne adayı" olarak işaretlenebiliyor — yöntem sadece renk farkına bakıyor, lezyonun akne mi yoksa başka bir pigmentasyon türü mü olduğunu ayırt etmiyor.
- Dudak dışlama maskesi convex hull tabanlı olduğu için dudak köşelerinde/kenarlarında hafif bir dışlama taşması veya eksik kapama olabilir; `dilate_px` bunu bir miktar telafi ediyor. Test edilen görsellerde dudaklar doğru şekilde dışlandı (bkz. Test sonuçları).
- `make_region_masks`'taki "cheeks" birleştirmesi artık **tek taraf** (sol veya sağ) yeterli piksele sahip olsa bile çalışıyor — yüz karede tam ortalanmamış fotoğraflarda bile referans/skor hesaplanabilsin diye (göz altı morluk.1'in orijinal `make_cheeks_mask()` davranışıyla aynı dayanıklılık).

## Nasıl çalıştırılır

```bash
cd "akne\1"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8301` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit edilen bölge sayısı | Not |
|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | |
