# İlerleme Günlükleri (Progress Logs)

Bu klasör, yarışma boyunca **günlük ilerleme raporlarını** tutar. Her dosya o günün
hikâyesini baştan sona anlatır: veri incelemesi, alınan kararlar ve gerekçeleri, denenen
modeller, ne işe yaradı / ne yaramadı, LB sonuçları ve sonraki adımlar.

Teknik deney-deney kayıt için `EXPERIMENTS.md` (kök dizin), sayısal log için `results.csv`'ye bakın.

## Günlükler

| Tarih | Rapor | Özet |
|---|---|---|
| 2026-06-11 | [2026-06-11.md](2026-06-11.md) | Gün 1: pipeline kuruldu, exp001–011 + blend; en iyi LB ~83.6. Tabular frontier kapandı (residual analizi); BERT düzeltildi. Gece exp012/exp013 koşuyor. |
| 2026-06-12 | [2026-06-12.md](2026-06-12.md) | Gün 2: text push tuttu → **LB 83.17** (yeni en iyi). Multi-seed ve pseudo-labeling reddedildi; basit-leak denetimi temiz. Akşam: exp018 mDeBERTa + exp019 CatBoost@ks-v3. |
| 2026-06-13 | [2026-06-13.md](2026-06-13.md) | Gün 3: **signal floor kanıtlandı** (7 bağımsız test, hepsi `reports/eda/floor_analysis.py` ile tekrar üretilebilir + 4 figür). CV→LB farkı %100 yıl-dağılımı; tepeye açık büyük oranda public-LB gürültüsü. Karar: public'i kovalama, private için sağlam blend kilitle. 5 final blend varyantı hazır. |
