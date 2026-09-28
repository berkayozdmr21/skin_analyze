# Yüz Lekesi.1 — Yanak Referanslı Koyuluk (L*) Farkı

## Bu proje nedir

Göz Altı Morluk / Akne / Gözenek aileleriyle **aynı mimari** kullanılarak, yüz lekesi (güneş lekesi, melazma, hiperpigmentasyon) tespiti için kurulmuş ilk sürüm. Masaüstündeki diğer projelerle aynı aile: MediaPipe FaceMesh → ten bölgesi maskesi → göz/kaş/dudak hariç tutma → ısı haritası + kontur işaretleme → FastAPI + tek sayfalık arayüz.

## Yöntem

1. Lab renk uzayına çevrilir, **L\*** (parlaklık) kanalı alınır — lekeler öncelikle bir **koyuluk/pigmentasyon** belirtisi (akne'nin renk/kırmızılık belirtisinden farklı olarak, göz altı morluk'un koyuluk mantığına daha yakın).
2. **Referans bölge**: yanaklar (`cheeks`) — göz altı morluk.1/akne.1 ile birebir aynı mantık, "tipik/lekesiz ten" örneği.
3. **Hedef (taranan) bölge**: akne.1'deki gibi tüm taranabilir yüz cildi (alın+burun+çene+yanak, göz/kaş/dudak hariç) — lekeler göz altı morluk'taki gibi sabit bir bantta değil, yüzün herhangi bir yerinde olabilir.
4. Her piksel için:
   ```
   delta = cheek_L_mean - L(piksel)
   ```
   Pozitif ve büyük değer = piksel, yanaktan belirgin şekilde daha **koyu** → leke adayı.

## Baştan doğru kurulan tasarım kararları

Akne ve Gözenek ailelerinde iki önemli sorun **sonradan** ortaya çıkıp düzeltildi — bu proje baştan bu derslerle kuruldu:

1. **Çözünürlükten bağımsız (göreli) eşikler.** Gürültü temizleme çekirdeği ve minimum kontur alanı, taranan yüz alanının piksel sayısından türetilen bir oran — sabit piksel sayısı değil.
2. **Skor, ALANA değil SAYIYA/ŞİDDETE dayalı.** Akne.5 ve Gözenek.3'te skor alan-ağırlıklıydı, bu da çözünürlük değiştikçe skorun çökmesine yol açıyordu (Gözenek.3: %58→%5). Gözenek.4'te doğrulanan çözüm — skoru sabit bir SAYI referansına (`COUNT_CAP`) bölmek, alana değil — burada baştan uygulanıyor.

## Şekil filtresi

Lekeler (güneş lekesi, melazma) akne benekleri veya gözeneklerden daha **büyük** ve daha **düzensiz/blotchy** sınırlara sahip olabilir — bu yüzden `MIN_CIRCULARITY` akne'dekinden (0.35) daha düşük (**0.22**) ve `MAX_SPOT_AREA_RATIO` daha yüksek (**0.02**) tutuldu.

## İstatistik — skor NEYE göre veriliyor?

```
her tespit edilen leke i için:
    şiddet_i = clip(bölge_i_ortalama_delta / 30, 0, 1)

ağırlıklı_sayı = Σ şiddet_i
skor           = clip(ağırlıklı_sayı / COUNT_CAP, 0, 1)
```

`COUNT_CAP = 25` gerçek bir test fotoğrafıyla kalibre edildi. Her tespit edilen leke şiddeti her zaman pozitif olduğu için, yeni bir leke tespiti skoru **hiçbir zaman düşürmez** — 0 leke tespit edilirse skor **kesinlikle %0**'dır.

## Doğrulama — çözünürlük kararlılığı ve tutarlılık

Aynı test fotoğrafı (`test edilen görseller/File.jpg`) farklı çözünürlüklerde çalıştırıldı:

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 170 px | %37.3 | 12 |
| 700 px | %36.1 | 12 |
| 1200 px | %52.4 | 16 |
| 2000 px | %39.8 | 13 |
| 3000 px | %37.4 | 13 |

Skor **%36-%52** arasında kaldı — akne/gözenek ailesinin ilk (hatalı) sürümlerindeki 10 kat farka kıyasla oldukça stabil. Sıfır-bölge tutarlılığı da doğrulandı: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

## Görsel doğrulama

Tespit edilen bölgeler, fotoğraftaki gerçek görünür ben/leke izleriyle (sol yanak, burun kenarı, ağız köşesi, çene) örtüşüyor. Isı haritasında yüz tarama maskesinin **kenarında** (saç/çene hattı sınırında) parlak bir "halka" efekti gözlemlendi — bu, maskenin sınırındaki doğal gölge/geçiş bölgelerinden kaynaklanıyor. Şekil filtresi (dairesellik) bu ince/uzun sınır izini **doğru şekilde** gerçek tespitlerden ayırıyor — işaretlenen bölgeler hep kompakt/yuvarlak lekeler, sınır halkası değil.

## Maskeleme

`face_mesh.py`, Akne/Gözenek ailesiyle **birebir aynı kopya** — `make_face_scan_mask()` (tüm ten, göz/kaş/dudak hariç) ve `make_cheeks_mask()` (referans bölgesi) burada da doğru seçimler.

## Ayarlanabilir parametreler (`leke.py` içinde)

- `DELTA_VISUAL_CAP = 30.0` — göz altı morluk.1 ile aynı sabit (aynı kanal, aynı ölçek).
- `DETECTION_THRESHOLD = 12.0` — göz altı morluk.1 ile aynı.
- `CLEAN_KERNEL_RATIO`, `MIN_AREA_RATIO` — gürültü temizliği, çözünürlüğe oranlı.
- `MIN_CIRCULARITY = 0.22`, `MAX_SPOT_AREA_RATIO = 0.02` — şekil filtresi (akne/gözenekten daha gevşek, lekeler daha büyük/düzensiz olabileceği için).
- `COUNT_CAP = 25.0` — skorun %100'e ulaştığı, sabit ağırlıklı-sayı referansı.

## Bilinen sınırlama

- Sadece **koyuluğa** bakıyor; bazı pigmentasyon türleri (örn. kızarık/pembe post-inflamatuar izler) bu yöntemle yakalanmayabilir — akne ailesindeki renk (a\*) sinyali burada kullanılmıyor, çünkü klasik "leke" tipik olarak öncelikle bir koyuluk belirtisidir.
- Isı haritasında maske sınırında görülen parlak halka, gerçek tespitleri etkilemiyor (şekil filtresi tarafından eleniyor) ama görsel olarak dikkat dağıtıcı olabilir.
- `COUNT_CAP = 25` tek bir gerçek fotoğrafla kalibre edildi; farklı ten tiplerinde/yaşlarda (doğal ben/çil sayısı büyük ölçüde değişir) ince ayar gerekebilir.

## Nasıl çalıştırılır

```bash
cd "yüz lekesi\1"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8501` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama koyuluk farkı | Tespit sayısı | Not |
|---|---|---|---|---|---|
| File.jpg (170x226) | 0.373 | 1.07 | 30.3 | 12 | Doğrulama testi, yukarıya bakınız |
| _(0 bölge simülasyonu)_ | 0.000 | 0.00 | 0.0 | 0 | Skor/tespit tutarlılığı doğrulandı |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
