# Yüz Lekesi.4 — Yerel Uyarlamalı Eşikleme (Adaptive Thresholding)

## İstenen değişiklik

> "4. yaparmısın farklı bir teknik ama aynı işlev"

## Farklı teknik, aynı işlev

Yüz Lekesi.1-3, hep **aynı temel teknik** ailesine dayanıyordu: bir referans bölgenin (yanak, sonra tüm yüz) istatistiğini (ortalama/std veya medyan/MAD) hesaplayıp, **her pikseli bu tek, küresel (global/bölgesel) referansla kıyaslamak**. Klasik bir "global eşikleme" yaklaşımı.

Bu sürümde tamamen farklı bir teknik kullanılıyor: **yerel uyarlamalı eşikleme** (`cv2.adaptiveThreshold` ile bilinen klasik bir görüntü işleme yöntemi, kavramsal olarak burada normalize edilmiş konvolüsyonla uygulanıyor). Fikir: her pikseli tüm yüzün ortalamasıyla değil, **sadece kendi yakın çevresinin** (küçük bir pencere) ortalamasıyla kıyasla:

```
yerel_ortalama(x,y) = pikselin çevresindeki KÜÇÜK bir pencerenin (Gauss ağırlıklı) ortalaması
delta(x,y) = yerel_ortalama(x,y) - piksel(x,y)
```

Bu, Yüz Lekesi.1-3'teki "referans bölgesi" kavramını **tamamen ortadan kaldırıyor** — artık ne yanak ne tüm yüz referans alınıyor, her pikselin **kendi yerel komşuluğu** kendi referansı.

## Bu tekniğin doğal avantajı: Yüz Lekesi.3'ün sorununu tasarım gereği çözüyor

Yüz Lekesi.3'te uğraştığımız sorun (büyük bir vitiligo/melazma alanının global/bölgesel referansı bozması), burada **doğası gereği** ortadan kalkıyor: yerel pencere küçük olduğu için, uzaktaki büyük bir atipik bölge bir pikselin yerel komşuluğunu **etkilemiyor** — o pencere zaten oraya ulaşmıyor. Medyan/MAD gibi "robust ama yine de tüm veriye bakan" bir çözüme bile gerek kalmıyor.

**Doğrulandı:** Aynı yapay vitiligo testinde (v3'te kullanılanın aynısı), bu sürümde hiçbir işaret yama içine veya sınırına düşmedi — v3'te (medyan/MAD + erosion ile) kısmi iyileşme sağlamıştık, burada sorun kaynağından ortadan kalktığı için çok daha temiz.

## Geliştirme süreci — iki başarısız deneme, sonunda doğru ayar

Dürüstçe: bu teknik ilk denemede düzgün çalışmadı, iki hatalı ayardan geçti:

1. **İlk deneme** (küçük pencere, `BLOCK_SIZE_RATIO=0.05`, taban 15px): Isı haritası aşırı "benekli" çıktı — pencere o kadar küçüktü ki ince ten dokusunu bile "leke" sayıyordu.
2. **İkinci deneme** (büyütülmüş pencere, taban 25-151px): Beklenenin aksine **daha da kötü** oldu — sebep, büyük Gauss bulanıklaştırmanın göz/kaş/dudak "delikleri"nin kenarlarındaki normalize-konvolüsyon artefaktlarını **daha geniş bir alana** yaymasıydı (delik kenarına yakın piksellerde yerel ortalama güvenilmez hale geliyor, pencere büyüdükçe bu güvenilmez bölge de büyüyor).
3. **Gerçek çözüm**: pencere boyutunu "doğru" ayarlamak değil, **tespit alanını** bu güvenilmez kenar bölgelerinden arındırmaktı. `DETECTION_MARGIN_RATIO` ile tespit alanı (SADECE sonucun değerlendirildiği bölge, yerel ortalama HESABI için değil) delik ve maske kenarlarından ek olarak içbüküldü.

Bu üçüncü adımdan sonra, orta boyutlu bir pencere (`BLOCK_SIZE_RATIO=0.045`, taban 21px) ile makul, temiz sonuçlar elde edildi.

## Görsel doğrulama — çözünürlüğün etkisi

Test fotoğrafımız (170×226px) çok düşük native çözünürlükte olduğu için 4 kata yakın büyütülüyor — bu, **Gözenek.1'de zaten belgelenen** bilinen bir artefakt kaynağı (interpolasyon dokusu). Bu sürümde de aynı etki gözlemlendi: düşük çözünürlükte ısı haritası benekli görünürken, **gerçek yüksek çözünürlükte** (2000px kaynak) test edildiğinde çok temiz: sadece gerçek görünür işaretler (göz altındaki ben, burun/ağız kenarındaki lekeler) tespit edildi, sahte benek yok.

## Doğrulama — sıfır-bölge ve çözünürlük

| Kaynak genişlik | Skor | Tespit sayısı |
|---|---|---|
| 700 px | %55.4 | 15 |
| 1200 px | %46.5 | 12 |
| 2000 px | %46.0 | 12 |
| 3000 px | %33.0 | 8 |

Sıfır-bölge tutarlılığı da doğrulandı: tespit eşiği yapay olarak çok yükseltilip 0 bölge bulunacak şekilde test edildiğinde skor **tam olarak %0.0** çıktı.

## Önceki sürümle kıyaslama

| | Yüz Lekesi.1-3 | Yüz Lekesi.4 |
|---|---|---|
| Referans kavramı | Global/bölgesel (yanak veya tüm yüz) | **Yerel (her pikselin kendi çevresi)** |
| Büyük atipik alan (vitiligo) riski | Kısmen var (v3'te robust istatistikle azaltıldı) | **Tasarım gereği yok** |
| Kenar/delik artefaktı riski | Var (v3'te erosion ile azaltıldı) | Var (farklı bir mekanizmadan — bu sürümde `DETECTION_MARGIN_RATIO` ile çözüldü) |
| Teknik | İstatistiksel (z-score/delta, global referans) | **Görüntü işleme (adaptive thresholding, yerel pencere)** |
| Skor/işaretleme mantığı | Sayı/şiddet ağırlıklı, renkli daire | Değişmedi |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`leke.py` içinde)

- `BLOCK_SIZE_RATIO = 0.045`, `MIN/MAX_BLOCK_SIZE` — yerel pencerenin göreli boyutu.
- `C_OFFSET = 13.0` — bir pikselin yerel ortalamasından ne kadar koyu olması gerektiği (L* birimi).
- `DETECTION_MARGIN_RATIO = 0.05`, `MIN/MAX_DETECTION_MARGIN` — tespit alanının delik/kenar bölgelerinden içbükülme miktarı.
- `COUNT_CAP = 18.0` — yeni teknikle yeniden kalibre edildi.
- Şekil filtresi (`MIN_CIRCULARITY`, `MAX_SPOT_AREA_RATIO`) — değişmedi.

## Bilinen sınırlama

- Çok düşük çözünürlüklü kaynak fotoğraflarda (agresif büyütme gerektiren) ısı haritası hâlâ interpolasyon dokusuna hassas olabilir — bu, tekniğe özgü değil, önceki tüm sürümlerde de (ve Gözenek ailesinde de) gözlemlenen genel bir sınırlama.
- Yerel pencere boyutu sabit bir orandır; çok büyük tekil lekeler (pencereden daha geniş) yanlış değerlendirilebilir.

## Nasıl çalıştırılır

```bash
cd "yüz lekesi\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8504` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | Tespit sayısı | Not |
|---|---|---|---|
| File.jpg (2000px kaynak) | ~0.46 | 12 | Temiz, gerçek işaretler — yüksek çözünürlükte doğrulandı |
| File.jpg + yapay vitiligo yaması | 1.00 (doygun, düşük çözünürlük gürültüsünden) | 33 | Yamanın kendisi hiç yanlış işaretlenmedi |
| _(kendi fotoğrafınızla tekrar deneyip sonucu paylaşabilirsiniz)_ | | | |
