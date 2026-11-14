# Unified Fire Activity Calendar – Harmonized MODIS & VIIRS

NASA Space Apps challenge uygulaması: FIRMS formatındaki MODIS ve VIIRS sıcak noktalarını (hot spots) **1 km** mesafe ve **±6 saat** zaman penceresinde harmonize eder, haritada ve aktivite takviminde gösterir.

## Özellikler

1. **FIRMS CSV yükleme** — `latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `satellite`
2. **Harmonizasyon** — eşleşen MODIS+VIIRS → *Matched (High Confidence)*; tek sensör → *MODIS-only* / *VIIRS-only*
3. **Folium harita** — yeşil / turuncu / mavi
4. **Yangın aktivitesi takvimi** — günlük veya haftalık çizgi grafik
5. **Özet istatistikler** — toplam, matched, MODIS-only, VIIRS-only

## Çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py --server.port 8529
```

Tarayıcıda açılan arayüzde örnek veriyi kullanabilir veya kendi FIRMS CSV’nizi yükleyebilirsiniz.

## Proje yapısı

```
app.py                 # Streamlit arayüzü
src/harmonize.py       # Mesafe/zaman eşleme mantığı
data/sample_firms.csv  # Örnek hot spotlar
requirements.txt
```

## Harmonizasyon kuralı

Bir MODIS noktası ile bir VIIRS noktası şu koşulların ikisini de sağlarsa *Matched (High Confidence)* olur:

- Haversine mesafe ≤ 1 km (kenar çubuğundan değiştirilebilir)
- `|acq_datetime_MODIS − acq_datetime_VIIRS|` ≤ 6 saat

Eşleme açgözlüdür: her MODIS için zaman penceresindeki en yakın, henüz eşleşmemiş VIIRS seçilir.

## Uydu tanıma

| Sensör | Örnek `satellite` değerleri      |
|--------|----------------------------------|
| MODIS  | Terra, Aqua, MODIS               |
| VIIRS  | N, N20, N21, NOAA-20, Suomi NPP  |

İsteğe bağlı `instrument` sütunu da okunur.

## Veri kaynağı

[NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) — Fire Information for Resource Management System.
