# Göz Altı Morluk.1 — Bölgesel L* Farkı (Yanak Referanslı)

## Yöntem

1. Lab renk uzayına çevrilir, **L\*** (parlaklık) kanalı alınır.
2. **Referans bölge**: yanaklar (`cheeks` — yüzün sol/sağ yarısının convex hull'u, ana skin-scan projesindeki aynı mantık). Yanağın L\* ortalaması (`cheek_L_mean`) hesaplanır.
3. **Göz altı bandı**: alt göz kapağı landmark zinciri (dış köşeden iç köşeye) alınıp, göz genişliğinin **yarısı kadar aşağı** kaydırılarak bir şerit (bant) oluşturulur. Bu bant, gözün hemen altındaki tipik "morluk/torba" bölgesini kapsar.
4. Her piksel için:
   ```
   delta = cheek_L_mean - L(piksel)
   ```
   Pozitif ve büyük değer = o piksel, yanaktan **belirgin şekilde daha koyu** → morluk/gölge adayı.
5. **Isı haritası**: `delta` (sadece pozitif kısmı) [0, 30] aralığında [0,1]'e ölçeklenir, göz altı bandı dışında sıfırlanır.
6. **Skor**: göz altı bandındaki ortalama pozitif delta, aynı şekilde ölçeklenir.

## Tespit edilen bölgeleri işaretleme (yeni — kullanıcının istediği özellik)

Sadece bir ısı haritası/skor değil, **`delta > 12` (L\* birimi)** olan pikseller ayrıca:
1. İkili (binary) maskeye çevrilir, göz altı bandı ile kesişimi alınır.
2. Küçük gürültü noktaları morfolojik açma (`MORPH_OPEN`) ile temizlenir.
3. `cv2.findContours` ile bu bölgelerin **konturları** bulunur (alanı 20 pikselden küçük olanlar elenir).
4. Her kontur, orijinal fotoğraf üzerine **sarı kontur çizgisi + dikdörtgen çerçeve** ile çizilir → **"tam olarak burada tespit ettim"** görseli.

Arayüzde bu, ısı haritasının yanında **ikinci bir görsel** olarak (`Tespit Edilen Bölgeler`) gösteriliyor, ayrıca kaç ayrı bölgede tespit yapıldığı da yazıyor.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip( mean(clip(cheek_L_mean - L, 0, None)[göz_altı_bandı]) / 30, 0, 1 )
```
Yani göz altı bandındaki **ortalama koyuluk farkı**, sabit bir ölçekle (0-30 L\* birimi) [0,1]'e sıkıştırılıyor. `30` değeri ampirik — çok koyu bir morluk için makul bir üst sınır olarak seçildi, `morluk.py` içinde `DELTA_VISUAL_CAP` olarak ayarlanabilir.

**Tespit işaretleme** ayrı bir eşikle (`DETECTION_THRESHOLD = 12`) çalışıyor — bu, skoru etkilemiyor, sadece "somut olarak nerede bulundu" görselleştirmesi için.

## Maskeleme

- `cheeks`: ana skin-scan projesindeki aynı yöntem (yüzü ortadan ikiye bölüp her yarının convex hull'u).
- `under_eye`: ana projede daha önce eklediğimiz aynı yöntem (alt göz kapağı landmark'ları + göz genişliğinin yarısı kadar aşağı offset).

Maskeleme yöntemi **değişmedi** — sadece bu klasörde bağımsız bir kopya olarak duruyor.

## Ayarlanabilir parametreler (`morluk.py` içinde)

- `DELTA_VISUAL_CAP = 30.0` — skor/ısı haritası ölçeklemesi için üst sınır (L\* birimi).
- `DETECTION_THRESHOLD = 12.0` — bir pikselin "tespit edilen morluk noktası" (işaretlenecek) sayılması için gereken minimum delta.
- `MIN_CONTOUR_AREA = 20` — gürültü temizliği için minimum kontur alanı (piksel).

## Nasıl çalıştırılır

```bash
cd "göz altı morluk\1."
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8201` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit edilen bölge sayısı | Not |
|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | |
