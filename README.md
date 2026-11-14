# Unified Fire Activity Calendar

NASA Space Apps Challenge 2026 – MODIS ve VIIRS hot spot harmonizasyonu.

Uygulama, [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/) üzerinden
canlı sıcak nokta verisi çeker; **1 km** mesafe ve **±6 saat** zaman penceresinde
MODIS ↔ VIIRS eşleşmelerini *Matched (High Confidence)* olarak işaretler.

## Özellikler

1. **FIRMS_MAP_KEY** — `.env` dosyasından okunur (Git’e eklenmez)
2. **Bölge** — Türkiye, Akdeniz, Ege, İç Anadolu veya manuel bbox
3. **Sensörler** — `MODIS_NRT`, `VIIRS_NOAA20_NRT`, `VIIRS_NOAA21_NRT` (çoklu)
4. **Gün sayısı** — 1–10 slider
5. **Harmonizasyon** — Matched / MODIS-only / VIIRS-only
6. **Folium harita** — yeşil / turuncu / mavi
7. **Yangın aktivitesi takvimi** — günlük veya haftalık Plotly grafik
8. **İstatistik kartları** — toplam, matched, MODIS-only, VIIRS-only

## Çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # FIRMS_MAP_KEY değerini yazın
streamlit run app.py --server.port 8529
```

1. [FIRMS MAP Key](https://firms.modaps.eosdis.nasa.gov/api/key/) alın.
2. `.env` içine `FIRMS_MAP_KEY=...` yazın.
3. Bölge/sensör/gün seçip **Veriyi Getir**’e basın.

## Proje yapısı

```
app.py                 # Streamlit arayüzü
src/firms_api.py       # FIRMS Area API istemcisi
src/harmonize.py       # Mesafe/zaman eşleme
data/sample_firms.csv  # Harmonizasyon birim testleri için örnek
.env.example           # Anahtar şablonu (.env Git'te yok)
requirements.txt
```

## API

```
https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SENSOR}/{min_lon},{min_lat},{max_lon},{max_lat}/{days}
```

## Harmonizasyon kuralı

Bir MODIS ile bir VIIRS noktası şu koşulların ikisini de sağlarsa *Matched* olur:

- Haversine mesafe ≤ 1 km
- `|acq_datetime_MODIS − acq_datetime_VIIRS|` ≤ 6 saat

Eşleme açgözlüdür: her MODIS için zaman penceresindeki en yakın, henüz
eşleşmemiş VIIRS seçilir.

## Testler

```bash
python tests/test_harmonize.py
python tests/test_firms_api.py
```

## Güvenlik notu

`FIRMS_MAP_KEY` `.env` dosyasındadır ve `.gitignore` ile hariç tutulur.
Anahtar loglanmaz, `session_state`’e yazılmaz ve hata mesajlarında gösterilmez.
