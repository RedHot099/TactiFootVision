# xg-temporal: wyniki w skrócie

Faza zamknięta 2026-10-28 decyzją właściciela, bez odczytu etykiet. Pełny raport z liczbami sprawdzanymi przez `scripts/check_xgd_audit.py`: [REPORT.md](REPORT.md).

- **Pytanie:** czy liniowa głowa czasowa na embeddingach G-combo z okna przed kontaktem obniża out-of-fold log-loss G-combo.
- **Odpowiedź:** nie wiadomo. H1 i H2 nie były testowane, bo metoda trzy razy nie przeszła własnej kontroli pozytywnej (zasiany efekt ruchu, wyrocznia T − S równa progowi praktycznemu). Liczby wszystkich trzech prób są w sekcji 5 raportu.
- **Wniosek o metodzie:** na obecnej populacji liniowa głowa odzyskuje mniej niż połowę zasianego efektu, a żaden badany efekt nie osiąga mocy 80%. Zera syntetyczne są w porządku, więc problemem jest moc, nie fałszywe odrzucenia.
- **To nie jest dowód braku informacji o ruchu.**
- **Rekomendacja:** nie wracać do tej głowy na obecnych danych. Ruch lepiej badać przez niskowymiarowe cechy z detektora albo na znacznie większej liczbie meczów (ekstrapolacja w sekcji 9 raportu).

![Odzyskany efekt i moc trzech prób](figures/recovery_power.png)

![Kontrole zerowe trzech prób](figures/null_rejections.png)

Figury: `scripts/xgt_report.py`, z `results/xgt/controls/*/controls.json`. Duże wyjścia (embeddingi, repliki) zostają w `outputs/` i `results/`, poza repozytorium.
