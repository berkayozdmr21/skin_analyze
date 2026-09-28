# Yağlılık.2 — Bölgesel (Alın/Burun/Çene/Yanak Ayrı) Yüzdelik Dilim Referansı

## İstenen değişiklik

> "2. yap farklılaştırarak teknikleri güncelleyerek"

## Yağlılık.1'in sorunu (Gözenek.1 → .2 ile birebir aynı motivasyon)

Yağlılık.1, "parlama" eşiğini (V/S yüzdelik dilimlerini) **tek bir global referansla** hesaplıyordu: taranan yüzün tamamı üzerinden tek bir yüzdelik dilim kesme değeri. Ama yağlılık/parlama görünürlüğü yüzün farklı bölgelerinde **anatomik olarak farklı** — T-bölgesi (alın+burun+çene) doğal olarak daha fazla yağ bezi içerir, sağlıklı bir tende bile yanaklardan daha parlak görünür.

Tek bir global yüzdelik dilim kullanmak bu doğal farkı yanlış değerlendirebiliyordu:
- T-bölgesi genel dağılımı "parlaklık" yönünde çektiği için, yanaklardaki **gerçek** bir yağlılık, global eşiğin yüksek kalması yüzünden kaçırılabiliyordu.
- T-bölgesindeki tamamen normal/sağlıklı parlaklık bile (yanaklara kıyasla) "olağan dışı" sayılıp gereksiz yere işaretlenebiliyordu.

## Bu sürümde ne değişti

Yüzün her alt-bölgesi (alın, burun, çene, yanaklar) artık **kendi içinde** değerlendiriliyor:

```
her bölge r için ayrı ayrı:
    V_eşik_r = persentil(V[bölge_r], V_PERCENTILE)
    S_eşik_r = persentil(S[bölge_r], S_PERCENTILE)
```

"Bu alın pikseli, DİĞER ALIN piksellerine göre ne kadar parlak" ve "bu yanak pikseli, DİĞER YANAK piksellerine göre ne kadar parlak" ayrı ayrı soruluyor.

## Doğrulama — bölgesel eşikler gerçekten farklı mı?

Aynı test fotoğrafında, gerçek V/S yüzdelik dilim kesme değerleri ölçüldü:

| Bölge | Piksel sayısı | V eşiği (yüzdelik dilim) | S eşiği |
|---|---|---|---|
| **Global (tüm yüz)** | 92.968 | 234.0 | 33.0 |
| Alın | 50.348 | 237.0 | 34.0 |
| **Burun** | 764 | **255.0** | 33.0 |
| **Çene** | 2.836 | **250.0** | 27.0 |
| Yanaklar | 88.558 | 233.0 | 33.0 |

Burun ve çenenin "parlak sayılma" eşiği (255, 250), yanakların eşiğinden (233) **gerçekten yüksek** — yani bu bölgelerde bir pikselin "olağan dışı parlak" sayılması için ÇOK DAHA parlak olması gerekiyor, tıpkı anatomik olarak beklendiği gibi. Bu, düzeltmenin gerçek ve ölçülebilir bir etkisi olduğunu doğruluyor.

## YENİ: bölge bazlı kaplama raporlaması

Artık yağlı/parlak alanın hangi yüz bölgesinde ne kadar kapladığı da raporlanıyor. Test fotoğrafında ilginç bir bulgu: **çene %32.8** kaplı çıkarken, yanaklar sadece **%5.3** — bu, T-bölgesi varsayımının bu fotoğrafta gerçekten doğru olduğunu somut olarak gösteriyor.

## Doğrulama — çözünürlük kararlılığı

| Kaynak genişlik | Skor | Kaplanan alan | Alın | Burun | Çene | Yanaklar |
|---|---|---|---|---|---|---|
| 170 px | %64.3 | %5.14 | %5.9 | %17.0 | %32.1 | %5.0 |
| 1200 px | %67.6 | %5.41 | %6.0 | %13.3 | %32.8 | %5.3 |
| 3000 px | %62.3 | %4.98 | %5.5 | %10.0 | %32.6 | %4.8 |

Genel skor/kaplama stabil kaldı; burun (en küçük piksel havuzuna sahip bölge) yüzdelik dilim tahmininde biraz daha fazla dalgalanma gösteriyor — beklenen bir davranış (küçük örneklem, daha gürültülü persentil tahmini).

**Sıfır-bölge tutarlılığı** korundu: eşik yapay olarak imkansız hale getirildiğinde skor **tam olarak %0.0** çıktı.

## Önceki sürümle kıyaslama

| | Yağlılık.1 | Yağlılık.2 |
|---|---|---|
| Referans | Tek global (tüm yüz) | **Bölgesel (alın/burun/çene/yanak ayrı)** |
| Küçük/eksik bölge | — | Güvenli global fallback (`MIN_REGION_PIXELS`) |
| Raporlama | Skor + kaplama yüzdesi | **+ bölge bazlı kaplama dağılımı** |
| Tespit mantığı (HSV+gradyan, kaplama-alanı skoru) | Aynı | **Değişmedi** |

## Maskeleme

`face_mesh.py`'deki `make_region_scan_masks()` (Akne/Gözenek ailesinden hazır) burada da doğrudan kullanıldı — göz/kaş/dudak dışlanmış, bölge bazlı maskeler.

## Ayarlanabilir parametreler (`yaglilik.py` içinde)

- Yağlılık.1'deki tüm parametreler değişmedi.
- **YENİ**: `MIN_REGION_PIXELS = 300` — bir bölgenin kendi yüzdelik diliminin güvenilir sayılması için gereken minimum piksel sayısı.

## Bilinen sınırlama

- Burun gibi küçük piksel havuzuna sahip bölgelerde yüzdelik dilim tahmini daha gürültülü olabilir (çözünürlüğe göre %10-17 arası dalgalanma gözlendi).
- 4 bölge arasındaki sınırlar (convex hull tabanlı) kesin anatomik sınırlar değil.

## Nasıl çalıştırılır

```bash
cd "yağlılık\2"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8602` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Kaplanan alan (%) | Bölge dağılımı | Tespit sayısı | Yağlılık.1 skoru (kıyas) |
|---|---|---|---|---|---|
| File.jpg (1024px) | 0.676 | 5.41 | Alın:6.0 Burun:13.3 Çene:32.8 Yanak:5.3 | 23 | 0.649 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
