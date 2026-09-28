# Göz Altı Morluk.3 — Renk İmzası (Koyuluk + Mavi-Mor Ton)

## Fikir

Göz Altı Morluk.1 ve .2, göz altı bölgesini sadece **koyuluğa** (L\* parlaklık kanalı) bakarak değerlendiriyordu. Ama bu, gerçek bir "morluk" (damarların ince ciltten görünmesiyle oluşan **mavi-mor** renkli bölge) ile sıradan bir **gölge** (örneğin üst göz kapağının düşmesiyle oluşan, aynı tonda ama daha koyu bir alan) arasında ayrım yapamıyor. Gölge **nötr/gri** renktedir — sadece koyulaşır, renk kaymaz. Gerçek morluk ise hem koyulaşır **hem de renk olarak maviye/mora kayar**.

## Yöntem

Artık iki ayrı z-score hesaplanıp birleştiriliyor:

1. **Koyuluk z-score (`z_L`)** — Göz Altı Morluk.2 ile birebir aynı: `(ten_L_ortalama - L) / ten_L_std`.
2. **Mavi-mor ton z-score (`z_b`)** — Lab **b\*** kanalı (mavi↔sarı ekseni) kullanılıyor: `(ten_b_ortalama - b) / ten_b_std`. Pozitif değer = o piksel, tenin normal sarımsı tonundan **daha maviye** kaymış.

İkisi ağırlıklı birleştiriliyor:
```
combined_z = 0.6 × z_L + 0.4 × z_b
```

**Sonuç:** Hem koyu **hem** mavimsi olan bölgeler en yüksek skoru alır. Sadece koyu ama renk kaymamış (muhtemelen düz gölge) bölgeler daha düşük skor alır — çünkü sadece `z_L` katkısı var, `z_b` katkısı yok/düşük.

## İstatistik — skor NEYE göre veriliyor?

```
skor = clip(mean(combined_z[göz_altı_bandı]) / 2.5, 0, 1)
```
Ölçek (`Z_SCORE_CAP=2.5`) ve tespit eşiği (`DETECTION_Z_THRESHOLD=1.2`), Göz Altı Morluk.2 ile aynı tutuldu — böylece iki versiyonun skorları kabaca karşılaştırılabilir kalıyor, tek değişken renk bileşeninin eklenmesi.

**Şeffaflık:** Arayüzde artık üç ayrı sayı gösteriliyor: koyuluk bileşeni, mavi-mor ton bileşeni, ve birleşik skor. Bu, "bu bölge neden yüksek çıktı — sadece koyu olduğu için mi, gerçekten renk de kaymış olduğu için mi" sorusuna cevap veriyor. Örneğin `z_L` yüksek ama `z_b` sıfıra yakınsa, bu muhtemelen **gölge**, gerçek morluk değil.

## Önceki versiyonlarla kıyaslama

| | Morluk.1 | Morluk.2 | Morluk.3 |
|---|---|---|---|
| Kullanılan kanal | Sadece L* | Sadece L* | **L\* + b\* (renk)** |
| Referans | Yanak | Tüm ten | Tüm ten (M.2 ile aynı) |
| Ölçek | Sabit | Z-score | Z-score (M.2 ile aynı) |
| Gölge/gerçek morluk ayrımı | Yok | Yok | **Var (kısmen)** |

## Bilinen sınırlama

Bu hâlâ kesin bir "gölge mi morluk mu" sınıflandırıcısı değil — sadece bir **ağırlıklandırma**. Çok güçlü bir gölge (örneğin çok yönlü/sert ışıkta) hâlâ yüksek `z_L` ile skoru yükseltebilir, mavi bileşen düşük olsa bile `0.6` ağırlığıyla hâlâ önemli bir katkı yapıyor. Ağırlıkları (`WEIGHT_DARKNESS` / `WEIGHT_BLUE_TONE`) test sonuçlarına göre ayarlamak gerekebilir — özellikle mavi bileşenin ağırlığını artırmak, gölgeleri daha güçlü elemek isteniyorsa denenebilir.

## Maskeleme

**Değişmedi** — `face_mesh.py`, Göz Altı Morluk.2 ile birebir aynı kopya.

## Ayarlanabilir parametreler (`morluk.py` içinde)

- `WEIGHT_DARKNESS = 0.6`, `WEIGHT_BLUE_TONE = 0.4` — iki bileşenin ağırlıkları.
- `Z_SCORE_CAP = 2.5` — skorun 1.0'a ulaştığı birleşik z-score eşiği.
- `DETECTION_Z_THRESHOLD = 1.2` — işaretleme eşiği.

## Nasıl çalıştırılır

```bash
cd "göz altı morluk\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8203` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Skor | z_L (koyuluk) | z_b (mavi ton) | Tespit sayısı | M.2 skoru (kıyas) | Not |
|---|---|---|---|---|---|---|
| _(buraya test sonuçlarını ekle)_ | | | | | | |
