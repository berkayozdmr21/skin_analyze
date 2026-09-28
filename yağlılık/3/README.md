# Yağlılık.3 — Çok Ölçekli Tespit + Cilt Tipi Sınıflandırması

## İstenen değişiklik

> "3. yap farklılaştırarak yüzde göstermeyi de farklılaştır"

İki ayrı farklılaştırma yapıldı: **(1) teknik** — gradyan doğrulaması artık çok ölçekli, **(2) yüzde gösterimi** — tek bir toplam yüzde yerine T-bölgesi/yanak ikili gösterimi + cilt tipi etiketi.

## 1. Teknik farklılaştırma: çok ölçekli gradyan doğrulaması

Yağlılık.1-2, specular highlight adaylarını **tek** bir Sobel çekirdeği (ksize=3) ile doğruluyordu. Gözenek.4'teki "çok ölçekli black-hat" ile aynı mantık burada uygulandı: **iki** farklı Sobel çekirdek boyutuyla (küçük=3, büyük=9) gradyan büyüklüğü hesaplanıp, bir pikselin "yumuşak" sayılması için **her iki ölçekte de** düşük gradyan göstermesi gerekiyor — tek ölçekli kontrolden daha sıkı bir doğrulama.

### Bir kalibrasyon hatası ve düzeltmesi (dürüstçe)

İlk denemede büyük çekirdek için sabit bir eşik (`GRADIENT_THRESHOLD_LARGE=90`) seçtim — ama bu **her şeyi sıfıra düşürdü**. Sebep: Sobel çekirdekleri normalize edilmemiş, çekirdek büyüdükçe gradyan büyüklüğü ölçeği **patlıyor** (ölçtüm: ksize=3'te medyan ≈25, ksize=9'da medyan ≈88.000 — 3.500 kat fark!). Sabit bir sayı böyle tamamen farklı ölçeklere uygulanamazdı.

Çözüm, bu dosyadaki V/S eşikleriyle **aynı felsefe**: sabit sayı yerine **yüzdelik dilim** — her ölçekte, o ölçeğin kendi gradyan dağılımının en "yumuşak" %70'i "düz/parlama adayı" sayılıyor. Ölçek ne olursa olsun otomatik uyum sağlıyor.

## 2. Yüzde gösterimi farklılaştırma: T-bölgesi/Yanak ikili gösterimi + cilt tipi

Dermatolojide cilt yağlılığı genelde tek bir sayı olarak raporlanmaz — **"kombine cilt"** (T-bölgesi yağlı, yanaklar normal/kuru) ve **"yağlı cilt"** (her yerde yağlı) gibi standart sınıflandırmalar, T-bölgesi ve yanak kaplamasının **ayrı ayrı** karşılaştırılmasına dayanır.

```
T-bölgesi_% = max(alın_%, burun_%, çene_%)   [en yüksek alt-bölge]
Yanak_%     = yanakların kendi kaplama yüzdesi

Cilt tipi:
    T-bölgesi YÜKSEK VE Yanak YÜKSEK  -> "Yağlı Cilt"
    T-bölgesi YÜKSEK VE Yanak DÜŞÜK   -> "Kombine Cilt"
    T-bölgesi DÜŞÜK VE Yanak DÜŞÜK    -> "Normal Cilt"
```

### Bir tasarım düzeltmesi (dürüstçe)

İlk denemede T-bölgesi yüzdesini alın+burun+çene'nin **havuzlanmış (pooled)** birleşimi olarak hesapladım — ama bu **yanıltıcı** çıktı: alın (50.000 piksel), burun+çene'den (764+2.836 piksel) kat kat büyük olduğu için, havuzlanmış ortalama neredeyse tamamen alının (daha az yağlı) değerini yansıtıyordu (T-bölgesi≈5.5%, yanak≈4.7% — neredeyse ayırt edilemez, "Normal Cilt" çıkıyordu, gerçek burun %17/çene %32 sinyali kayboluyordu).

Düzeltme: T-bölgesi yüzdesi artık alt-bölgelerin **maksimumu** — "T-bölgesinde HERHANGİ BİR yerde belirgin yağlılık var mı" sorusuna cevap veriyor, büyük ama daha az yağlı bir alt-bölge tarafından seyreltilmiyor.

## Doğrulama

Aynı test fotoğrafı farklı çözünürlüklerde çalıştırıldı:

| Kaynak genişlik | T-bölgesi | Yanak | Cilt tipi | Tespit sayısı |
|---|---|---|---|---|
| 170 px | %29.6 | %4.7 | Kombine Cilt | 22 |
| 700 px | %27.3 | %4.1 | Kombine Cilt | 20 |
| 1200 px | %25.6 | %4.6 | Kombine Cilt | 22 |
| 2000 px | %17.1 | %3.1 | Kombine Cilt | 26 |
| 3000 px | %13.7 | %2.4 | Kombine Cilt | 26 |

**Sınıflandırma (Kombine Cilt) tüm çözünürlüklerde tutarlı kaldı** — T-bölgesi yüzdesinin ham değeri çözünürlükle dalgalansa da (küçük piksel havuzlu burun/çene bölgesinden kaynaklanan beklenen bir durum), eşiği (%8) rahatça aşıyor, yanak ise hiçbir zaman yaklaşmıyor. Sıfır-bölge tutarlılığı da korundu: eşik yapay olarak imkansız hale getirildiğinde skor **tam %0.0**, tip **"Normal Cilt"** (güvenli varsayılan) çıktı.

## Önceki sürümle kıyaslama

| | Yağlılık.2 | Yağlılık.3 |
|---|---|---|
| Gradyan doğrulama | Tek ölçek (ksize=3, sabit eşik) | **Çok ölçekli (ksize 3+9, yüzdelik dilim)** |
| Ana gösterim | Tek toplam kaplama % | **T-bölgesi % + Yanak % + Cilt Tipi etiketi** |
| T-bölgesi hesabı | — | Alt-bölgelerin **maksimumu** (havuzlanmış ortalama değil) |
| Bölgesel V/S eşikleri, kaplama-alanı skoru | Aynı | Değişmedi (genel skor hâlâ hesaplanıp dönüyor) |

## Maskeleme

`face_mesh.py` değişmedi.

## Ayarlanabilir parametreler (`yaglilik.py` içinde)

- `GRADIENT_KERNEL_SMALL=3`, `GRADIENT_KERNEL_LARGE=9`, `GRADIENT_SMOOTH_PERCENTILE=70.0` — çok ölçekli gradyan doğrulaması.
- `OILY_REGION_THRESHOLD_PCT = 8.0` — bir bölgenin "yağlı" sayılması için gereken minimum kaplama yüzdesi (cilt tipi sınıflandırması için).
- Diğer parametreler (V/S yüzdelik dilimleri, bölgesel referans, kaplama-alanı skoru) — Yağlılık.2 ile aynı.

## Bilinen sınırlama

- `OILY_REGION_THRESHOLD_PCT=8.0` tek bir gerçek fotoğrafla kalibre edildi; farklı ten tiplerinde ince ayar gerekebilir.
- T-bölgesi için "maksimum alt-bölge" yaklaşımı, tek bir küçük bölgede (örn. sadece burun ucunda) lokalize aşırı parlamaya karşı hassas olabilir — bu, geniş bir T-bölgesi yağlılığından ziyade tek bir noktasal ışık yansımasını da "Kombine Cilt" olarak sınıflandırabilir.

## Nasıl çalıştırılır

```bash
cd "yağlılık\3"
"C:\Users\User\Desktop\skin-scan\.venv39\Scripts\python.exe" app.py
```

Sunucu `http://127.0.0.1:8603` üzerinde açılır.

## Test sonuçları

| Fotoğraf | Cilt Tipi | T-bölgesi % | Yanak % | Toplam kaplama % | Tespit sayısı |
|---|---|---|---|---|---|
| File.jpg (170px) | Kombine Cilt | 29.6 | 4.7 | 4.89 | 22 |
| _(0 bölge simülasyonu)_ | Normal Cilt | 0.0 | 0.0 | 0.00 | 0 |
| _(buraya kendi test sonuçlarını da ekleyebilirsin)_ | | | | | |
