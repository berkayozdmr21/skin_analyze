# skin_analyze — Yüz Cilt Analizi Deneyleri (OpenCV + MediaPipe)

Tek bir fotoğraftan **klasik görüntü işleme** (derin öğrenme modeli olmadan) ile yüz cildindeki farklı durumları ölçen, birbirinden bağımsız 6 mini proje ve toplam **27 sürüm**. Her proje kendi klasöründe, numaralı alt klasörlerde (`1`, `2`, `3`, ...) ilerler; her yeni sürüm bir öncekinin **istatistiğini, skor mantığını veya görselleştirmesini bilinçli olarak değiştirir** ve neden değiştirdiğini kendi `README.md` dosyasında ölçümlerle belgeler.

| Proje | Ölçtüğü şey | Sürüm | Port |
|---|---|---|---|
| [`göz altı morluk`](göz%20altı%20morluk) | Göz altı koyu halka / morluk | 1-5 | 8201-8205 |
| [`akne`](akne) | Akne / kırmızı sivilce ve lezyonlar | 1-5 | 8301-8305 |
| [`gözenek`](gözenek) | Gözenek tespiti | 1-4 | 8401-8404 |
| [`yüz lekesi`](yüz%20lekesi) | Pigmentasyon / leke / çil | 1-5 | 8501-8505 |
| [`yağlılık`](yağlılık) | Cilt parlaması / yağlılık + cilt tipi | 1-4 | 8601-8604 |
| [`yüz kırışıklık`](yüz%20kırışıklık) | Kırışıklık çizgileri | 1-4 | 8701-8704 |

## Ortak mimari

Her sürüm klasörü aynı dosya setine sahiptir:

| Dosya | Görev |
|---|---|
| `face_mesh.py` | MediaPipe FaceMesh (478 nokta) ile yüz bölgeleri (alın, burun, çene, yanaklar) ve **hariç tutulan** alanlar (göz, kaş, dudak) için maskeler |
| `preprocess.py` | Gri-dünya renk normalizasyonu, CLAHE, gamma, düşük çözünürlüklü fotoğraflar için yükseltme |
| `<konu>.py` | Projenin çekirdek algoritması (`morluk.py`, `akne.py`, `gozenek.py`, `leke.py`, `yaglilik.py`, `kirisiklik.py`) |
| `visualize.py` | Isı haritası ve işaretleme çizimleri |
| `app.py` | FastAPI sunucusu (`/health`, `/scan`) |
| `client.html` | Fotoğraf yükleyip sonucu gösteren tek sayfalık arayüz |
| `README.md` | O sürümün yöntemi, gerekçesi ve test sonuçları |

## Kurulum ve çalıştırma

Python 3.9 ile test edildi.

```bash
git clone https://github.com/berkayozdmr21/skin_analyze.git
cd skin_analyze
pip install -r requirements.txt
cd "yüz kırışıklık/4"
python app.py
```

Sonra `http://127.0.0.1:8704` adresini açıp bir yüz fotoğrafı yükleyin. Her sürüm kendi portunu kullanır (tabloya bakın), yani hepsini aynı anda çalıştırıp karşılaştırabilirsiniz.

> Test fotoğrafları ve ekran görüntüleri gizlilik nedeniyle repoya **konmadı** (`.gitignore`). Kendi fotoğraflarınızla deneyin.

## Projelerin evrimi

### Göz altı morluk (1-5)
1. Göz altı bölgesi ile yanak arasındaki L* (koyuluk) farkı.
2. Ten rengine göre z-skoru.
3. Koyuluk + mavi-mor ton (renk imzası).
4. Robust Mahalanobis mesafesi (çok değişkenli).
5. Yerel aydınlatma düzeltmeli Mahalanobis.

### Akne (1-5)
1. Yanak referanslı kırmızılık (a*) farkı.
2. Tüm ten referanslı z-skoru.
3. a* + HSV doygunluk birleşimi.
4. **Çözünürlükten bağımsız tespit**: aynı fotoğraf farklı çözünürlükte 13 ile 48 arasında bölge veriyordu; tüm mutlak piksel eşikleri yüz ölçeğine bağlandı.
5. **Skor, tespit edilen bölgelerden hesaplanır**: "0 bölge" gösterirken skorun sıfırdan büyük çıkması hatası giderildi.

### Gözenek (1-4)
1. Black-hat morfolojisi.
2. Bölgesel (alın/burun/çene/yanak) referans.
3. Sayı (yoğunluk) tabanlı skor.
4. Çok ölçekli tespit. Skoru alana bölmek yüksek çözünürlükte skoru çökertiyordu, sabit sayı referansına geçildi.

### Yüz lekesi (1-5)
1. Yanak referanslı L* farkı.
2. Tüm yüz z-skoru.
3. Robust (medyan/MAD) referans. Vitiligo benzeri büyük lekeleri k-means ile ayırma denemesi **daha kötü çıktı** ve README'de başarısız deney olarak belgelendi.
4. Yerel uyarlamalı eşikleme; algılama alanı, referans alanından ayrı aşındırıldı.
5. Morfolojik kapama ile arka plan tahmini: yoğun çillerin kendi referansını bozması engellendi.

### Yağlılık (1-4)
1. HSV parlama (specular highlight) tespiti.
2. Bölgesel yüzdelik dilim referansı; sabit eşik yerine fotoğrafa göre yüzdelik dilim kullanılır.
3. Çok ölçekli tespit + **cilt tipi sınıflandırması** (Yağlı / Kombine / Normal). T-bölgesinde ortalama yerine maksimum kullanmak, küçük ama yağlı burun/çene sinyalinin sulanmasını önledi.
4. Parlama + doku pürüzsüzlüğü birleşimi.

### Yüz kırışıklık (1-4)
1. Çok yönelimli Gabor filtre bankası (6 yön) + robust koyuluk.
2. Morfolojik kapama ile yerel arka plan referansı (yoğun kırışıklıkta kendi kendini maskelemeyi önler).
3. Minimum uzunluk filtresi: kırışıksız yüzde çıkan yanlış tespitler 30'dan 6'ya indi, ama yaşlı yüzlerde duyarlılık aşırı düştü.
4. **Çözünürlük tavanı hatası düzeltildi** + eşikler dengelendi (aşağıya bakın).

## Bu projelerden çıkan ortak dersler

1. **Mutlak piksel eşikleri çözünürlükle bozulur.** Çekirdek boyutu, minimum alan ve minimum uzunluk gibi değerler `yüz ölçeği = sqrt(tarama alanı)` üzerinden türetilmeli. Çekirdek boyutlarına konan üst sınırlar (tavan) da gerçek çözünürlük aralığını kapsamalı: kırışıklık.3'te tavanlar 2000 piksel civarında aşılıyor, tespit sayısı 10'dan 3'e çöküyordu; kırışıklık.4'te tavanlar yükseltilerek düzeltildi.
2. **Sayı benzeri niceliği alana bölme.** Alan çözünürlüğün karesiyle, sayı ise (göreli filtrelerle) neredeyse sabit ölçeklenir. Bu oran çözünürlük arttıkça skoru çökertir. Uzunluk için yüz ölçeğine (doğrusal), sayı için sabit referansa bölmek gerekir.
3. **Skor, gösterilen tespitle aynı hesaptan çıkmalı.** Ayrı bir "tüm yüz ortalaması" skoru, "0 bölge" yazarken sıfırdan büyük yüzde göstermeye yol açar.
4. **Öz-referanslı eşikler** (medyan/MAD, yüzdelik dilim, yerel arka plan) sabit sayılardan daha iyi genelleşir; ama referansı bozan yapılar (yoğun çil, yoğun kırışıklık, göz/dudak delikleri) için ayrıca önlem gerekir (morfolojik kapama, inpaint, aşındırılmış algılama marjı).
5. **Hassasiyet ile kesinlik arasında takas vardır.** Tek bir kırışıksız fotoğrafla ayarlanan eşik, yaşlı yüzlerde duyarlılığı öldürür (kırışıklık.3). Her sürüm bu dengeyi ölçerek belgeler.
6. **Başarısız deneyleri de belgelemek** (yüz lekesi.3'teki k-means) sonraki sürümlerin aynı hataya düşmesini önler.

## Bilinen sınırlamalar

- Doğrulama büyük ölçüde tek bir düşük çözünürlüklü (170x226) referans fotoğraf, birkaç kullanıcı ekran görüntüsü ve sentetik testlerle yapıldı. Geniş bir veri kümesiyle ölçülmüş bir doğruluk iddiası yoktur.
- Bunlar **tıbbi teşhis aracı değildir**; deney ve eğitim amaçlıdır.
- Farklı ten renkleri, aydınlatma ve kamera koşulları için sistematik bir değerlendirme yapılmadı.
