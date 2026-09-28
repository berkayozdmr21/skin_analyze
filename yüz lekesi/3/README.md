# Yüz Lekesi.3 — Robust (Medyan/MAD) Referans

## Kullanıcının bildirdiği sorun (ekran görüntüsüyle)

Büyük, açık renkli bir pigmentasyon alanına (vitiligo/melazma benzeri) sahip bir yüzde, Yüz Lekesi.2 "bazı yerleri karıştırıyordu" — ısı haritasında geniş, dağınık parlak bölgeler ve büyük alanın sınırını takip eden çizgiler görünüyordu, sadece küçük/nokta şeklindeki gerçek lekeler değil.

## Kök neden

Yüz Lekesi.2, referans istatistiğini **ortalama + standart sapma** ile (ham, "robust olmayan" biçimde) hesaplıyordu. Eğer ten bölgesinin büyük bir kısmı (örneğin vitiligo gibi geniş bir depigmentasyon alanı) normal tenden çok farklı bir parlaklıkta ise, bu büyük "aykırı" grup ortalamayı kendi yönüne çeker — sonuç: geri kalan normal ten bile, bu kaydırılmış ortalamaya göre "anormal derecede farklı" görünür, bu da yüzün geniş bölgelerinin yanlışlıkla "leke" olarak işaretlenmesine yol açar.

## Bu sürümde denenenler

### 1. Medyan + MAD (Median Absolute Deviation) — uygulandı

```
ten_medyan = medyan(L[ten_bölgesi])
ten_mad    = medyan(|L[ten_bölgesi] - ten_medyan|)
z = (ten_medyan - L) / (ten_mad × 1.4826)
```

Medyan, ortalamanın aksine, verinin **%50'sinden azı** aykırı/farklı olduğu sürece o "tipik/çoğunluk" değerine sabit kalır — birkaç (hatta yüzeyin bir kısmını kaplayan) aykırı bölge onu kolayca kaydıramaz.

### 2. İki-kümeli (k-means) ayrıştırma — denendi, İPTAL EDİLDİ

Daha da güçlü bir fikir olarak, ten bölgesindeki L* değerlerini k-means ile iki kümeye ayırıp (varsa) büyük/atipik kümeyi referans hesabından tamamen çıkarmayı denedim. **Test sonucu bu fikri iptal ettim**: yapay bir "vitiligo" testinde (fotoğrafın ~%32'sini kaplayan parlatılmış bir alan), k-means'in bulduğu "çoğunluk" kümesinin medyanı (224) gerçek orijinal medyandan (188) **daha uzak** çıktı — çünkü yüzdeki doğal parlaklık varyasyonu (burun/alın parlaması, çene gölgesi) ile eklenen atipik bölgeyi güvenilir şekilde ayırt edemiyordu. Daha basit olan medyan/MAD'ın tahmin edilebilir davranışı tercih edildi.

### 3. Tarama sınırının içbükülmesi (erosion) — uygulandı

Isı haritasında gözlemlenen "sınır çizgisi" artefaktlarını azaltmak için `face_scan_mask` artık hafifçe içbükülmüş (`EDGE_ERODE_RATIO`) — saç çizgisi/çene hattı gibi geçiş bölgelerindeki pikseller artık taramaya dahil edilmiyor.

## Doğrulama — dürüst sonuç

Kullanıcının kendi fotoğrafı elimde olmadığı için, aynı başarısızlık modelini simüle eden **yapay bir test** kullanıldı: referans fotoğrafın sol yarısına (~%32'lik alan) +55 L* birimlik yapay bir "vitiligo" yaması eklendi, v2 ve v3 bu görüntüde karşılaştırıldı.

| | Yüz Lekesi.2 | Yüz Lekesi.3 |
|---|---|---|
| Yapay vitiligo görüntüsünde tespit sayısı | 8 | 11 |
| Vitiligo yamasının içi işaretlendi mi? | Hayır | Hayır |
| Vitiligo sınırında işaret var mı? | Sınıra YAKIN bir işaret (#7) vardı | Sınırdan biraz daha uzak |

**Dürüst değerlendirme:** Bu spesifik yapay testte, hem v2 hem v3 büyük yamanın İÇİNİ doğru şekilde işaretlemedi (ikisi de "daha koyu" arıyor, yama daha AÇIK olduğu için zaten tetiklenmiyordu) — asıl risk sınır/geçiş bölgesindeydi ve v3'ün erosion+medyan/MAD kombinasyonu bunu biraz daha iyi engelledi ama farkı **çarpıcı değil, ölçülü**. Gerçek fotoğrafınızla tekrar test edip hâlâ karışıklık varsa, parametreleri (özellikle `EDGE_ERODE_RATIO` ve `MIN_CIRCULARITY`) birlikte ince ayar yapabiliriz.

Normal (vitiligo olmayan) referans fotoğrafta regresyon kontrolü de yapıldı: skor 0.396→0.386, tespit sayısı 13→12 — anlamlı bir bozulma yok.

## Önceki sürümle kıyaslama

| | Yüz Lekesi.2 | Yüz Lekesi.3 |
|---|---|---|
| Referans istatistiği | Ortalama + std | **Medyan + MAD (robust)** |
| Tarama sınırı | Ham (tam maske) | **Hafifçe içbükülmüş (erosion)** |
| İşaretleme, skor formülü | Aynı | Değişmedi |

## Maskeleme

`face_mesh.py` değişmedi. Tarama bölgesi artık `leke.py` içinde `_erode_scan_mask()` ile ek olarak içbükülüyor.

## Ayarlanabilir parametreler (`leke.py` içinde)

- `COUNT_CAP = 22.0` — Yüz Lekesi.2 ile aynı (z-score birimi değişmedi).
- **YENİ**: `EDGE_ERODE_RATIO = 0.01`, `MIN/MAX_EDGE_ERODE_KERNEL` — tarama sınırının içbükülme miktarı.
- Diğer parametreler (şekil filtresi, çözünürlük-oranlı boyutlandırma) değişmedi.

## Bilinen sınırlama

- Medyan/MAD, aykırı bölge yüzeyin **%50'sine yakın veya daha fazlasını** kapladığında (yapay testimizdeki gibi ~%32-43 gibi aşırı durumlarda) tam koruma sağlamaz — hiçbir tek-değişkenli istatistik bu durumda mükemmel çalışamaz. Gerçekten çok büyük/baskın bir pigmentasyon farkı olan fotoğraflarda hâlâ dikkatli değerlendirme gerekebilir.
- k-means tabanlı bölge ayrıştırma denendi ve güvenilmez bulunduğu için kullanılmadı — bu, gelecekte daha gelişmiş bir yöntemle (örn. gerçek bir segmentasyon modeli) tekrar ele alınabilir.

## Nasıl çalıştırılır

```bash
cd "yüz lekesi\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8503` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit sayısı | Not |
|---|---|---|---|
| File.jpg (normal, regresyon kontrolü) | 0.386 | 12 | Yüz Lekesi.2: 0.396/13 — anlamlı fark yok |
| File.jpg + yapay vitiligo yaması | 0.297 | 11 | Yama içi/sınırı doğru şekilde işaretlenmedi |
| _(kendi fotoğrafınızla tekrar deneyip sonucu paylaşabilirsiniz)_ | | | |
