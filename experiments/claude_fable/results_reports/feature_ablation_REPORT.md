# Wdrożenie cech P0/P1.1: wyniki ablacji

Data: 2026-08-31. Plan: `research/feature_implementation_plan.md`; burza mózgów:
`research/feature_brainstorm.md`. Baza: D = głowa na zamrożonym DINOv2-L (3 seedy,
ensemble), GBM pozycyjny. Test 2 229 strzałów / 100 meczów; `penalty_xg` = średnie
xG modelu na karnych z testu (rzeczywista konwersja ~0.76).

## Wyniki (dwa niezależne przebiegi bazowe pokazują szum seedów D ≈ ±0.005-0.01 AUC)

| Konfiguracja | D AUC | D log-loss | D pen-xG | GBM AUC | GBM log-loss | GBM pen-xG |
|---|---|---|---|---|---|---|
| base (przebieg 1 / 2) | 0.738 / 0.733 | 0.3551 / 0.3574 | 0.257 / 0.248 | 0.6149 | 0.3825 | 0.157 |
| + is_penalty | 0.7435 | 0.3536 | 0.263 | **0.6379** | **0.3698** | **0.587** |
| + occlusion (światło bramki) | 0.7346 | 0.3571 | 0.253 | 0.6115 | 0.3830 | 0.157 |
| + GK-proxy | 0.7344 | 0.3573 | 0.251 | 0.6086 | 0.3834 | 0.157 |
| + ball_in_air | 0.7403 | **0.3533** | 0.276 | 0.6191 | 0.3811 | 0.165 |
| + fixdef (naprawione cechy obrońców) | 0.7417 | 0.3545 | 0.258 | 0.6164 | 0.3813 | 0.164 |
| + all | 0.7391 | 0.3535 | 0.296 | 0.6360 | 0.3709 | 0.563 |
| + sel (pen+air+fixdef) | 0.7339 | 0.3578 | 0.250 | 0.6367 | 0.3706 | 0.556 |

## Werdykty (kryteria: nie pogarsza log-loss + poprawia cel)

| Cecha | GBM | Głowa D | Decyzja |
|---|---|---|---|
| `is_penalty` | **+2.3 pp AUC, log-loss −0.013, xG karnych 0.157→0.59** — powtarzalne w 2 przebiegach | +0.005-0.01 AUC, w granicach szumu seedów; xG karnych ledwo drgnęło (0.25→0.26-0.30) — obraz częściowo już "widzi" karne | **PRZYJĘTA** (obowiązkowa w GBM; w D neutralna-lekko dodatnia) |
| `ball_in_air` | +0.4 pp | najlepszy log-loss i ECE w przebiegu 1, niepowtórzone w combo | **PRZYJĘTA warunkowo** (tania, nie szkodzi, pełni też rolę flagi jakości pozycji piłki) |
| `fixdef` (bug-fix obrońców) | +0.15 pp | w szumie | **PRZYJĘTA** jako poprawka poprawności (bug realny), bez oczekiwań metrycznych |
| `occlusion` (światło bramki) | −0.3 pp | −0.4 pp | **ODRZUCONA przy obecnej ekstrakcji** — sygnał ginie w szumie drużyn/pozycji (~1 s anotacji = obrońcy 3-6 m dalej). Wraca przy pozycjach GT z SoccerTrack. |
| `GK-proxy` | −0.6 pp | −0.4 pp | **ODRZUCONA** — najgłębszy obrońca to za słabe przybliżenie GK przy szumie drużyn. |

## Wnioski

1. Największy pojedynczy zysk wdrożenia: **kalibracja karnych w GBM (0.157→0.59)**
   i +2.3 pp AUC — mechanizm oczywisty i powtarzalny.
2. **Głowa D jest już blisko nasycenia informacją tanio dostępną**: żadna cecha nie
   daje na niej zysku ponad szum seedów. To spójne z hipotezą, że DINOv2-L widzi
   większość tej geometrii w pikselach.
3. Propozycje "światło bramki" i "GK" są dobre merytorycznie (literatura: część
   pakietu +0.03-0.05 AUC na czystym trackingu), ale **przy naszej ekstrakcji
   pogarszają wyniki** — właściwa kolejność prac: najpierw jakość pozycji
   (SoccerTrack GT / lepszy keypoint model), potem te cechy.
4. dl672 (cechy DINOv2-L @672×378) porzucone po 3 próbach: ekstraktor wymaga
   ~29 GB RAM przy zapisie (do przebudowy na zapis strumieniowy, niska wartość
   oczekiwana przy silnej bazie 448).

## Stan produkcyjnego zestawu cech po wdrożeniu

- GBM: 20 cech + `is_penalty` + `ball_in_air(+valid)` + poprawka obrońców.
- D (rekomendowany model): bez zmian architektury; `is_penalty` przekazywana do
  głowy (neutralna metrycznie, wymagana dla spójności produktowej xG karnych) —
  wariant `AttentionPoolHead(aux_dim=...)` w `d_foundation.py`.
- Kod: `xgpoc/features/aux.py` (+testy), `scripts/build_aux_table.py`,
  `scripts/run_feature_ablation.py`; tabela: `results/feature_ablation/ablation.csv`.
