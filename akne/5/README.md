# Akne.5 — Tespit Edilen Bölgelere Dayalı Skor

## Kullanıcının bildirdiği sorun (ekran görüntüsüyle)

Akne.4'te bir test şu sonucu verdi: **"0 adet sivilce/akne tespit edildi"** ama skor **"%6"** gösteriyordu. Bu, kullanıcı açısından anlamsız: hiç sivilce bulunmadıysa skor neden sıfır değil?

## Kök neden

Akne.1'den Akne.4'e kadar tüm sürümlerde **skor** ve **tespit sayısı**, birbirinden **tamamen bağımsız iki ayrı hesaplamaydı**:

- **Skor**: taranan YÜZÜN TAMAMINDAKİ ortalama `combined_z` — binlerce piksel (neredeyse hepsi normal ten) üzerinden alınan bir ortalama. Bu ortalama, tek tek pikseller eşiği geçmese bile hafifçe pozitif çıkabiliyordu (örneğin hafif ışık/doku farkları yüzünden).
- **Tespit sayısı**: sadece `DETECTION_Z_THRESHOLD`'u aşan, sivilce-şeklinde **bağımsız bölgelerin** sayısı.

Bu iki hesap farklı eşiklerle/farklı kapsamlarla çalıştığı için tutarsız sonuçlar üretebiliyordu.

## Bu sürümde ne değişti: skor artık TESPİT EDİLEN BÖLGELERDEN hesaplanıyor

Artık iki ayrı hesap yok — tek bir tespit süreci (`_detect()`) hem kontur listesini (işaretleme için) hem de skoru üreten veriyi birlikte döndürüyor:

```
her tespit edilen bölge i için:
    şiddet_i = clip(bölge_i_ortalama_combined_z / Z_SCORE_CAP, 0, 1)
    katkı_i  = şiddet_i × bölge_i_alanı (piksel)

toplam_yük = Σ katkı_i
skor = clip(toplam_yük / (taranan_yüz_alanı × 0.05), 0, 1)
```

Bu, dermatolojideki "lezyon yükü" (lesion burden) mantığıyla aynı fikir: çok sayıda/geniş/şiddetli bölge → yüksek skor; az sayıda/küçük/hafif bölge → düşük skor; **SIFIR bölge → skor MATEMATİKSEL OLARAK SIFIR** (artık ayrı bir hesap tutarsızlığı imkansız, çünkü skor doğrudan bölge listesinden türetiliyor).

`BURDEN_AREA_FRACTION = 0.05`: "yüzün %5'i maksimum şiddette sivilcelerle kaplı olsa, skor %100'e ulaşır" anlamına gelen ampirik bir referans — dermatolojik değerlendirmelerde yüzeyin %5'inin lezyonlu olması zaten belirgin/ciddi bir akne durumu sayılır.

## Doğrulama — bildirilen senaryo artık düzeldi

Akne.5'te, tespit eşiğini yapay olarak çok yükseğe çekip "0 bölge bulunması" durumu simüle edildi:

| Durum | Skor | Tespit sayısı |
|---|---|---|
| Normal (gerçek fotoğraf) | %35.1 | 14 |
| Yapay olarak 0 bölge bulunacak şekilde | **%0.0** | **0** |

Artık 0 bölge → her zaman %0 skor. Bildirilen "%6 ama 0 bölge" tutarsızlığı yapısal olarak imkansız hale geldi.

## Skorun yeni ölçeği

Skor artık FARKLI bir şeyi ölçtüğü için (yüzün tamamının ortalaması yerine tespit edilen bölgelerin yükü), sayı ölçeği de değişti — aynı test fotoğrafında:

| | Akne.4 (eski yöntem) | Akne.5 (yeni yöntem) |
|---|---|---|
| Skor | %14.3 | **%35.1** |
| Tespit sayısı | 14 | 14 |

Bu bir hata değil — Akne.4'ün skoru "yüzün genel ortalama kızarıklığı" idi (çoğu piksel normal ten olduğu için doğal olarak düşük), Akne.5'in skoru ise "gerçekten tespit edilen lekelerin ne kadar yer kapladığı ve ne kadar şiddetli olduğu" — kavramsal olarak kullanıcının sorduğu soruya ("akne ne kadar büyük/ciddi") çok daha yakın.

## Arayüzdeki değişiklik

- Skorun altındaki kutu artık her zaman skorla **tutarlı**: 0 bölge → "0 adet sivilce/akne tespit edildi" + %0 skor.
- Yeni bir istatistik eklendi: **"Kaplanan alan"** (%) — tespit edilen bölgelerin taranan yüz alanına oranı, skorun NEYE dayandığını somutlaştırıyor.
- **"Ortalama şiddet (z)"** de gösteriliyor — tespit edilen bölgelerin ortalama standart sapma cinsinden ne kadar belirgin olduğu.

## Önceki sürümle kıyaslama

| | Akne.4 | Akne.5 |
|---|---|---|
| Skorun kaynağı | Yüzün tamamının ortalama z'si | **Tespit edilen bölgelerin yükü (alan × şiddet)** |
| 0 bölge → skor | Sıfır OLMAYABİLİR (tutarsızlık riski) | **Her zaman kesinlikle sıfır** |
| Yeni istatistik | — | **Kaplanan alan (%), ortalama şiddet (z)** |
| Tespit mantığı (a*+S, dairesellik, göreli boyutlandırma) | Aynı | **Değişmedi** |

## Bilinen sınırlama

- Skor artık tespit edilen bölgelerin toplam alanına dayandığı için, çözünürlük değiştikçe **skor** de biraz dalgalanabiliyor (test edilen fotoğrafta %13-%42 arası) — tespit SAYISI (14-19) daha stabil kalıyor ama alan ölçümü, farklı ölçeklerdeki interpolasyon farklarından hafifçe etkileniyor. Bu, gelecekteki bir sürümde ele alınabilecek ayrı bir iyileştirme alanı.
- `BURDEN_AREA_FRACTION = 0.05` ampirik bir sabit; gerçek kullanıcı fotoğraflarıyla daha fazla test edilip ince ayar gerekebilir (çok düşük/yüksek çıkan skorlar için önce bu sabiti tartışalım).

## Maskeleme ve tespit mantığı

**Değişmedi** — `face_mesh.py` Akne.4 ile birebir aynı kopya; kırmızılık+doygunluk birleşimi, dairesellik/alan filtresi ve çözünürlükten bağımsız boyutlandırma da (Akne.4'ten) aynen korundu.

## Ayarlanabilir parametreler (`akne.py` içinde)

- Akne.4'teki tüm parametreler (`WEIGHT_REDNESS`, `WEIGHT_SATURATION`, `Z_SCORE_CAP`, `DETECTION_Z_THRESHOLD`, `MIN_CIRCULARITY`, `MAX_SPOT_AREA_RATIO`, `KERNEL_SIZE_RATIO`, `MIN_AREA_RATIO` vb.) — değişmedi.
- **YENİ**: `BURDEN_AREA_FRACTION = 0.05` — skorun %100'e ulaştığı "lezyon yükü" referansı.

## Nasıl çalıştırılır

```bash
cd "akne\5"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8305` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama şiddet (z) | Tespit sayısı | Akne.4 skoru (kıyas) | Not |
|---|---|---|---|---|---|---|
| File.jpg (170x226) | 0.351 | 2.23 | 1.73 | 14 | 0.143 | Doğrulama testi, yukarıya bakınız |
| _(0 bölge simülasyonu)_ | 0.000 | 0.00 | 0.00 | 0 | — | Bildirilen tutarsızlık artık imkansız |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
