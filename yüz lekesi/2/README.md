# Yüz Lekesi.2 — Tüm Yüz Referanslı Z-Score

## İstenen değişiklik

> "istatistik kısmını azıcık değiştir ve işaretleme şeklini değiştir"

## 1. İstatistik değişikliği: yanak referansı → tüm yüz referanslı z-score

Göz Altı Morluk.1→.2 ve Akne.1→.2 geçişleriyle birebir aynı motivasyon:

- **Referans genişletildi**: sadece yanak yerine, dışlama uygulanmış tüm ten (`face_scan_mask` — alın+burun+çene+yanak, göz/kaş/dudak hariç) referans alınıyor. Yanak convex hull'u dudak bölgesini de içine alabiliyordu; artık bu kontaminasyon yok.
- **Sabit ölçek → z-score**: skor artık sabit "30 L* birimi" yerine kişinin kendi teninin doğal koyuluk dalgalanmasına (standart sapmasına) göre ölçekleniyor:
  ```
  z = (ten_L_ortalama - L_piksel) / ten_L_std
  ```

Bu, "küçük/azıcık" bir istatistik değişikliği — aynı temel mantık (koyuluk farkı), sadece referans ve ölçek kişiye özelleşti.

## 2. İşaretleme şekli değişikliği: kontur+çerçeve → şiddet-renkli numaralı daire

Önceki sürümlerde (Göz Altı Morluk/Akne/Gözenek/Yüz Lekesi.1) tüm tespitler **aynı sabit sarı renkte**, kontur+dikdörtgen çerçeve ile işaretleniyordu — "kaç tane bulundu" sorusuna cevap veriyordu ama "hangisi daha ciddi" sorusuna cevap vermiyordu.

Artık her leke, **kendi şiddetine göre renklendirilmiş** (sarı=hafif → kırmızı=belirgin) yarı saydam bir daire ile işaretleniyor, üstüne **sıra numarası** yazılıyor:

- Daire, lekenin düzensiz konturunu değil, onu kapsayan kompakt bir alanı gösteriyor — "yaklaşık burada" şeklinde daha okunaklı.
- Renk, doğrudan `leke_score()`'daki severity (şiddet) hesabından geliyor — görselleştirme ve sayısal veri artık **aynı kaynağa** dayanıyor, ayrı/tutarsız olma riski yok.
- Numaralandırma, gelecekte bir "leke listesi" (örn. "3. leke: yanak, şiddet 0.6") eklenmesini kolaylaştırıyor.

## Doğrulama

Aynı test fotoğrafı (`test edilen görseller/File.jpg`) farklı çözünürlüklerde çalıştırıldı:

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 170 px | %39.6 | 13 |
| 700 px | %39.0 | 13 |
| 1200 px | %33.7 | 11 |
| 2000 px | %50.9 | 16 |
| 3000 px | %53.9 | 17 |

Skor %34-%54 arasında kaldı — Yüz Lekesi.1 ile benzer düzeyde stabil. Sıfır-bölge tutarlılığı da doğrulandı: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

Yeni işaretleme görsel olarak da doğrulandı: sol yanaktaki gerçek ben belirgin/turuncu-kırmızı renkli büyük bir daire ile (#7), diğer küçük lekeler daha açık sarı-turuncu tonlarında işaretlendi — renk farkı gerçekten şiddet farkını yansıtıyor.

## Önceki sürümle kıyaslama

| | Yüz Lekesi.1 | Yüz Lekesi.2 |
|---|---|---|
| Referans bölge | Sadece yanak | **Tüm ten (dışlama uygulanmış)** |
| Ölçek | Sabit (30 L* birimi) | **Z-score (kişinin kendi std sapması)** |
| İşaretleme | Sabit sarı kontur+çerçeve | **Şiddete göre renkli, numaralı daire** |
| Skor formülü (sayı/şiddet ağırlıklı, alan değil) | Aynı | Değişmedi |

## Maskeleme

**Değişmedi** — `face_mesh.py`, Yüz Lekesi.1 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`leke.py` içinde)

- `Z_SCORE_CAP = 2.5`, `DETECTION_Z_THRESHOLD = 1.2` — Göz Altı Morluk/Akne ailesiyle aynı sabitler.
- `COUNT_CAP = 22.0` — z-score birimine göre yeniden kalibre edildi (Yüz Lekesi.1'in L*-delta biriminde kalibre edilmiş `25.0`'inden farklı).
- Şekil filtresi (`MIN_CIRCULARITY`, `MAX_SPOT_AREA_RATIO`) ve çözünürlük-oranlı boyutlandırma — değişmedi.

`visualize.py`'de yeni: `draw_severity_circles()` — şiddet→renk dönüşümü ve daire+numara çizimi. `draw_detection_markers()` eski yöntem olarak geriye dönük uyumluluk için tutuldu (artık kullanılmıyor).

## Bilinen sınırlama

- Daire, lekenin gerçek (düzensiz) şeklini değil, onu kapsayan bir çemberi gösteriyor — çok uzun/ince bir leke için daire gereğinden büyük bir alanı kaplayabilir.
- Yarı saydam dairelerin üst üste bindiği durumlarda (birbirine çok yakın lekeler) renkler karışıp okunması zorlaşabilir.

## Nasıl çalıştırılır

```bash
cd "yüz lekesi\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8502` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Ortalama Z-score | Tespit sayısı | Yüz Lekesi.1 skoru (kıyas) |
|---|---|---|---|---|---|
| File.jpg (170x226) | 0.396 | 1.25 | 1.68 | 13 | 0.373 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
