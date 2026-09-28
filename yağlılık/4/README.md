# Yağlılık.4 — Specular + Doku Pürüzsüzlüğü Birleşimi

## İstenen değişiklik

> "4 de yap farklılaştırarak"

## Farklılaştırma: HSV'ye ek bağımsız bir sinyal

Yağlılık.1'in araştırma bölümünde bulunan bir ipucu tam kullanılmamıştı: web araştırmasında karşıma çıkan bir öneri, "specular highlight tespitinin **ötesinde**, Local Binary Patterns veya **Laplacian varyansı** ile TAMAMLANABİLECEĞİ" idi. Yağlılık.1-3 sadece HSV specular highlight (parlaklık+doygunluk) sinyaline dayanıyordu. Bu sürümde araştırmanın önerdiği **tamamlayıcı sinyal** eklendi: **doku pürüzsüzlüğü** (Laplacian varyansı).

**Neden ayrı bir sinyal gerekiyor:** HSV specular testi sadece "çok parlak VE çok az doygun" (aynasal yansıma) noktalarını bulur. Ama gerçek yağlı cilt sadece birkaç parlak nokta değil, genelde daha geniş bir alanda "donuk parlaklık" ve düşük doku (az görünür gözenek) sergiler — bu, HSV parlaklık eşiğini geçmeyebilir ama yine de tipik bir "yağlılık" görünümüdür. Laplacian varyansı (klasik bir netlik/bulanıklık ölçüsü) bunu yakalar: pürüzsüz/yağlı bir yüzeyde yerel Laplacian enerjisi düşüktür.

```
özel_aday = (HSV specular: yüksek V + düşük S + çok ölçekli düzlük)
doku_aday = (belirgin parlaklık) VE (çok düşük yerel Laplacian enerjisi — bölgesel yüzdelik dilim)

nihai_aday = özel_aday VEYA doku_aday
```

Bu bir "VE" değil **"VEYA"** birleşimi — iki bağımsız kriterden biri yeterli. Bu, hassasiyeti artırır: Yağlılık.1-3'ün sadece parlak nokta arayıp kaçırdığı "donuk ama pürüzsüz" bölgeleri de yakalar.

## Kalibrasyon hatası ve düzeltmesi (dürüstçe)

İlk denemede doku kriterini gevşek tuttum (`TEXTURE_SMOOTH_PERCENTILE=15`, `TEXTURE_MIN_V_PERCENTILE=50` — "en az orta parlaklık"). Sonuç **aşırı tespitti**: kaplanan alan %4.9'dan %14.5'e sıçradı, yanaklar yanlışlıkla "yağlı" işaretlendi (görsel kontrol: yanaklarda büyük, bağlantılı mavi bloklar), cilt tipi hatalı şekilde "Yağlı Cilt"e döndü (doğrusu "Kombine Cilt"ti — v3'te doğrulanmıştı).

Sebep: "ortalamanın üstü parlaklık + en pürüzsüz %15" kriteri, normal yanak dokusunun önemli bir kısmını bile kapsıyordu — doku tek başına çok genel bir kriter, sıkı bir eşleştirme gerekiyordu.

**Düzeltme:** `TEXTURE_SMOOTH_PERCENTILE=8.0` (sadece en pürüzsüz %8) ve `TEXTURE_MIN_V_PERCENTILE=75.0` (belirgin şekilde parlak) — HSV testine daha yakın bir sıkılıkta, ama hâlâ ayrı/tamamlayıcı bir sinyal.

## Doğrulama — düzeltme sonrası

| | Gevşek (hatalı) | Sıkı (düzeltilmiş) |
|---|---|---|
| Kaplanan alan | %14.5 | **%7.5** |
| Cilt tipi | Yağlı Cilt (yanlış) | **Kombine Cilt (v3 ile tutarlı)** |
| Yanak kaplaması | %14.1 | **%7.2** |

Çözünürlük kararlılığı (düzeltme sonrası):

| Kaynak genişlik | Kaplanan alan | Cilt tipi | Parlama sinyali | Doku sinyali |
|---|---|---|---|---|
| 170 px | %7.48 | Kombine Cilt | %5.41 | %4.33 |
| 700 px | %6.59 | Kombine Cilt | %5.28 | %4.30 |
| 1200 px | %7.12 | Kombine Cilt | %5.69 | %4.89 |
| 2000 px | %6.08 | Kombine Cilt | %5.33 | %4.89 |
| 3000 px | %5.07 | Kombine Cilt | %5.14 | %4.81 |

Cilt tipi sınıflandırması tüm çözünürlüklerde **Kombine Cilt** olarak tutarlı kaldı (v3 ile aynı sonuç, doğrulandı). Sıfır-bölge tutarlılığı da korundu: her iki sinyal de imkansız hale getirildiğinde skor **tam %0.0**, tip **"Normal Cilt"** çıktı.

**Görsel doğrulama:** Düzeltme sonrası işaretli görsel, v3'ün genel desenini (glabella, burun, çene, birkaç yanak noktası) koruyor, aşırı geniş yanak blokları kayboldu.

## Yeni: sinyal kaynağı raporlaması

Artık hangi tespitlerin HANGİ sinyalden geldiği ayrı ayrı raporlanıyor (`specular_pct` / `texture_pct`) — "bu yağlılık belirgin parlama noktalarından mı, yoksa genel pürüzsüz görünümden mi geliyor" sorusuna cevap. Test fotoğrafında ikisi de benzer büyüklükte katkı yapıyor (%5.1-5.7 parlama, %4.3-4.9 doku) — iki sinyal de anlamlı, biri diğerini domine etmiyor.

## Önceki sürümle kıyaslama

| | Yağlılık.3 | Yağlılık.4 |
|---|---|---|
| Sinyal sayısı | 1 (HSV specular) | **2 (HSV specular VEYA doku pürüzsüzlüğü)** |
| Kaplanan alan | ~%4.9-5.4 | **~%5.1-7.5 (doku katkısıyla artan hassasiyet)** |
| Raporlama | T-bölgesi/yanak/cilt tipi | **+ sinyal kaynağı dökümü (specular_pct/texture_pct)** |
| Cilt tipi sınıflandırması | Kombine Cilt | Kombine Cilt (**değişmedi, doğrulandı**) |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`yaglilik.py` içinde)

- **YENİ**: `TEXTURE_WINDOW_RATIO`, `MIN/MAX_TEXTURE_WINDOW` — Laplacian enerjisi için yerel pencere boyutu.
- **YENİ**: `TEXTURE_SMOOTH_PERCENTILE = 8.0` — bölge başına en pürüzsüz kabul edilen yüzdelik dilim (gerçek fotoğrafla sıkılaştırılarak kalibre edildi).
- **YENİ**: `TEXTURE_MIN_V_PERCENTILE = 75.0` — doku adayının aynı zamanda yeterince parlak olması şartı.
- `OILY_COVERAGE_CAP = 0.10` — doku sinyali eklenince toplam kaplama arttığı için Yağlılık.3'ün 0.08'inden yükseltildi.
- Diğer parametreler (V/S yüzdelik dilimleri, çok ölçekli gradyan, T-bölgesi/yanak eşiği) — değişmedi.

## Bilinen sınırlama

- Doku sinyali, kalibrasyonu ne kadar sıkı tutulursa tutulsun, doğası gereği HSV specular testinden daha "genel" bir kriter — çok farklı ışık koşullarında (örn. çok yumuşak/dağınık ışık) yeniden ayar gerekebilir.
- İki sinyal arasındaki örtüşme (aynı pikselin her iki kritere de uyması) şu an ayrı raporlanmıyor — `specular_pct + texture_pct`, toplam kaplamadan biraz büyük çıkabilir (örtüşen pikseller iki kez sayılıyor raporlamada, ama nihai maskede sadece bir kez).

## Nasıl çalıştırılır

```bash
cd "yağlılık\4"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8604` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Cilt Tipi | T-bölgesi % | Yanak % | Toplam kaplama % | Parlama % | Doku % |
|---|---|---|---|---|---|---|
| File.jpg (170px) | Kombine Cilt | 31.2 | 7.2 | 7.48 | 5.41 | 4.33 |
| _(0 sinyal simülasyonu)_ | Normal Cilt | 0.0 | 0.0 | 0.00 | 0.0 | 0.0 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | | |
