# Runda przedmerge'owa: wiarygodność i jakość wyników

Data: 2026-08-31. Plan: `../research/pre_merge_plan.md`. Test: 2 229 strzałów /
100 meczów, split po meczach, częstość bazowa 0.133. Wszystkie porównania między
modelami są **sparowane** i bootstrapowane po meczach (2 000 losowań).

Cel rundy nie brzmiał „podnieś AUC", tylko **„najpierw upewnij się, że dotychczasowe
liczby są uczciwe, dopiero potem je poprawiaj"**. Poniżej wyniki w tej kolejności.

## Wynik w jednym akapicie

Audyt nie znalazł przecieku, a zależność od grafiki telewizyjnej okazała się pozorna.
Z czterech hipotez poprawy jedna zadziałała i to mocno (**fine-tuning DINOv2 — po
naprawieniu learning rate**), jedna jest darmowa i zostaje ze względów odporności
(**maskowanie nakładki**), jedna została odrzucona pomiarem (**TTA**), a jedna okazała
się niezbędna w składzie mieszanym mimo słabej samodzielnej jakości (**model pozycyjny
— karne i górny ogon rozkładu**). Finalny ensemble, dobrany wyłącznie na walidacji:
**AUC 0.758 [0.734, 0.781], BSS 11.7%, log-loss 0.340** wobec poprzedniego wyniku
0.746 / 8.4% / 0.351 — poprawa **istotna** (Δlog-loss +0.0107 [+0.0058, +0.0157],
P = 1.000).

---

## 1. Defekt repozytorium wykryty przy okazji

Główny `.gitignore` zawiera reguły `data/` i `models/`. Dopasowują się na dowolnym
poziomie drzewa, więc **cały `xgpoc/data/` i `xgpoc/models/` był wykluczony z każdego
dotychczasowego commita** — 11 plików źródłowych, w tym wszystkie modele i datasety.
Branch w postaci zapisanej w gicie nie dawał się uruchomić.

Naprawa: negacje `!xgpoc/data/` i `!xgpoc/models/` w `.gitignore` eksperymentu.
Ujawniło to też trzy błędy lintera, których `ruff` nie widział, bo respektuje
`.gitignore`.

## 2. Audyt przecieku między splitami — **PASS**

`scripts/audit_leakage.py` → `results/audit/leakage.json`.

| Kontrola | Wynik |
|---|---|
| Duplikaty `shot_id` | 0 |
| Wspólne mecze train/valid/test | 0 (300 / 100 / 100 meczów) |
| Ten sam mecz pod dwiema nazwami | 0 |
| Klatki nieczytelne / zdegenerowane | 0 / 0 |
| **Klatki testowe z „bliźniakiem" w treningu** | **0 (0.00%)** |
| Pary z tego samego meczu w różnych splitach | 0 |

**Metodologicznie ważne:** pierwsza wersja używała 64-bitowego dHasha i dała wynik
bezużyteczny — na transmisjach piłkarskich rozkład dystansów par *tej samej akcji*
(t0 vs −400 ms; mediana 13) nachodzi na rozkład par losowych (percentyl 1 = 21).
Każdy próg albo gubi prawdziwe bliźniaki, albo flaguje tysiące niezwiązanych ujęć
ogólnych. Werdykt oparto więc na cosinusie osadzeń CLS DINOv2, które separują się
czysto (ta sama akcja p5 = 0.941 vs pary losowe p99 = 0.921). Obie kalibracje zostają
w raporcie — widać, dlaczego wybrano to narzędzie.

**Znalezisko poboczne, nie przeciek:** 48% klatek testowych ma w treningu klatkę
„podobnej sceny". Inspekcja (`cross_split_pairs.jpg`) i analiza par meczów pokazują,
że to **ten sam stadion, inna akcja** — 85 ze 100 meczów testowych ma drużynę
gospodarzy występującą też w treningu. To kwestia generalizacji, nie etykiet;
zmierzona osobno w sekcji 6.

## 3. Scoreboard: czy model czyta boisko, czy pasek z wynikiem?

Nakładkę zlokalizowano z danych (`scripts/find_overlay_region.py`): piksele, które
między t0 a t−1600 ms nie zmieniają się **w stosunku do własnego detalu przestrzennego**
— to odróżnia grafikę od jednolitej murawy, która też „się nie zmienia", ale tylko
dlatego, że nie ma czego zmieniać. Wykryto scoreboard `[80, 38, 409, 95]` i logo
kanału `[1017, 31, 1210, 116]`.

### Sondy kontrolne (model C uczony wyłącznie na wycinku)

| Sonda | Co widzi | AUC | BSS |
|---|---|---|---|
| `overlay_only` | grafika **+ cienkie statyczne pasy przy krawędziach kadru** | 0.606 | 1.1% |
| `graphics_only` | **tylko** scoreboard i logo kanału | **0.530** | **−0.2%** |

Pierwsza sonda wyglądała alarmująco — 0.606 AUC to tyle, co cały pipeline pozycyjny.
Druga rozstrzyga: **sama tablica wyników nie niesie sygnału** (BSS ujemny, model
degeneruje się do stałej). Cały sygnał pierwszej sondy pochodził z pasów przy
krawędziach kadru, zawierających bandy reklamowe i piksele murawy, czyli pośrednio
kadrowanie kamery. To zupełnie inna diagnoza niż „model czyta stan meczu".

### Retrening na zamaskowanych klatkach

| Model | AUC | BSS | log-loss | ΔAUC vs baza [95% CI] |
|---|---|---|---|---|
| C baza | 0.7361 | 8.0% | 0.3518 | — |
| C + maska | 0.7346 | 7.9% | 0.3520 | −0.0016 [−0.0080, +0.0048] |
| D-L baza | 0.7396 | 8.1% | 0.3533 | — |
| D-L + maska | 0.7353 | 8.2% | 0.3525 | −0.0042 [−0.0118, +0.0030] |

**Werdykt: maskowanie jest darmowe.** Zero mierzalnego kosztu, a usuwa całą klasę
ryzyka domenowego (inny realizator, materiał bez nakładki). Zostaje włączone domyślnie,
mimo że w tym zbiorze niczego nie zmienia.

## 4. Fine-tuning DINOv2 — **działa, ale dopiero po naprawieniu learning rate**

`FinetuneViTXG`: te same wagi DINOv2, odmrożone ostatnie N bloków, gradient
checkpointing, osobny (niższy) LR dla backbone'u. Koszt okazał się dużo niższy niż
zakładałem: pełne odmrożenie ViT-L mieści się w 6.2 GiB i 3.4 min/epokę.

### Pierwszy sweep głębokości (1 seed, `lr` 3e-4, mnożnik backbone'u 0.1)

| Konfiguracja | valid log-loss | test AUC | test BSS |
|---|---|---|---|
| DINOv2-L, 4 bloki | 0.3987 | 0.7465 | 6.5% |
| DINOv2-L, 12 bloków | 0.4221 | 0.678 | −0.2% |
| DINOv2-L, 24 bloki | 0.4219 | 0.574 | −0.2% |
| DINOv2-B, 12 bloków | 0.4220 | 0.553 | −0.2% |

Głębsze warianty **zapadły się do stałej predykcji**. Napisanie w tym miejscu
„głębsze odmrażanie nie działa" byłoby błędem: zapaść przy jednym ustawieniu LR to
awaria optymalizacji, nie własność metody. Dlatego przed werdyktem przeprowadzono
sweep LR.

### Sweep learning rate (1 seed) — zapaść była artefaktem LR

| Konfiguracja | valid LL | test AUC | test BSS | test log-loss |
|---|---|---|---|---|
| L/4, lr 3e-4, mult 0.02 | 0.3913 | 0.7435 | 8.1% | 0.3518 |
| L/4, lr 1e-4, mult 0.1 | 0.3937 | 0.7304 | 7.4% | 0.3539 |
| L/4, lr 1e-4, mult 0.02 | 0.4042 | 0.7409 | 8.3% | 0.3512 |
| **L/12, lr 1e-4, mult 0.02** | 0.4150 | **0.7483** | **8.8%** | **0.3502** |

Ta sama głębokość (12 bloków), która wcześniej dawała AUC 0.678 i BSS −0.2%, przy
mnożniku LR backbone'u 0.02 daje 0.748 i 8.8%. **Winny był learning rate, nie
głębokość.**

Zarazem widać tu granicę selekcji na jednym seedzie: wariant najlepszy na teście miał
**najgorszy** log-loss walidacyjny. Dlatego dwie czołowe konfiguracje przetrenowano na
3 seedach i wybrano po **średniej walidacyjnej z seedów** (L/4: 0.3891, L/12: 0.3932).

### Wynik końcowy (3 seedy, ensemble seedów)

| Model | AUC | BSS | log-loss | ECE | koszt |
|---|---|---|---|---|---|
| D-L zamrożony + maska | 0.7353 | 8.2% | 0.3525 | 0.019 | 3 × 40 s |
| C ConvNeXt + maska | 0.7346 | 7.9% | 0.3520 | 0.014 | 3 × 4 min |
| **G fine-tune L/4** | 0.7543 | **9.6%** | **0.3458** | 0.015 | 3 × 17 min |
| G fine-tune L/12 | 0.7583 | 9.5% | 0.3474 | 0.023 | 3 × 21 min |

Testy sparowane:

| Porównanie | ΔAUC [95% CI] | P | Δlog-loss [95% CI] | P |
|---|---|---|---|---|
| G-L4 vs D-L zamrożony | **+0.0192 [+0.0006, +0.0366]** | 0.977 | **+0.0067 [+0.0011, +0.0121]** | 0.993 |
| G-L4 vs C | +0.0198 [−0.0009, +0.0391] | 0.970 | +0.0062 [−0.0002, +0.0123] | 0.973 |
| G-L12 vs G-L4 | +0.0039 [−0.0048, +0.0131] | 0.81 | −0.0016 [−0.0048, +0.0018] | 0.18 |

**Fine-tuning istotnie bije zamrożoną głowę.** Obie głębokości są między sobą
nierozróżnialne, więc do produkcji wskazuję **L/4** — tańszą i lepiej skalibrowaną
(ECE 0.015 vs 0.023).

## 5. TTA (odbicie lustrzane) — **hipoteza odrzucona**

| Wariant vs jego baza | ΔAUC [95% CI] | P | Δlog-loss [95% CI] |
|---|---|---|---|
| C + TTA vs C | **−0.0065 [−0.0121, −0.0006]** | **0.014** | −0.0022 [−0.0042, −0.0002] |
| D-L + odbicie (aug + TTA) vs D-L | +0.0014 [−0.0058, +0.0089] | 0.65 | +0.0013 [−0.0008, +0.0034] |
| D-L + maska + TTA vs D-L + maska | −0.0030 [−0.0114, +0.0048] | 0.23 | **−0.0035 [−0.0064, −0.0007]** |

Dla C odbicie **istotnie szkodzi** obu metrykom, dla D z maską istotnie pogarsza
log-loss. C i głowa D uczą się już z losowym odbiciem, więc TTA nie dokłada
informacji — uśrednia natomiast logity dwóch widoków, co spłaszcza pewne predykcje.
**Nie wdrażać.**

## 6. Ocena na podgrupach

`results/subgroups/subgroups.csv`, krzywe niezawodności: `figures_premerge/reliability.png`.

### Pamiętanie stadionów

| Model | AUC (gospodarz znany z treningu, n=1880) | AUC (gospodarz nieznany, n=349) | Δ |
|---|---|---|---|
| C + maska | 0.7346 | 0.7340 | −0.001 |
| D-L + maska | 0.7389 | 0.7173 | −0.022 |
| **G fine-tune** | **0.7620** | 0.7119 | **−0.050** |
| Ensemble finalny | 0.7631 | 0.7333 | −0.030 |
| GBM pozycyjny | 0.6236 | 0.6185 | −0.005 |

C generalizuje na nieznane stadiony bez spadku; **fine-tuning traci ~5 pp** — czyli
część swojej przewagi kupuje adaptacją do miejsc widzianych w treningu. Przy n = 349
to obserwacja, nie dowód, ale kierunek jest spójny i wart monitorowania przy wdrożeniu
na nowe ligi.

### Gdzie modele działają (AUC)

| Podgrupa | n | C | D-L | G | GBM | Ensemble |
|---|---|---|---|---|---|---|
| 0–6 m | 120 | 0.760 | **0.838** | 0.807 | 0.587 | 0.807 |
| 6–11 m | 208 | 0.719 | 0.714 | 0.715 | 0.669 | 0.744 |
| 11–16 m | 290 | 0.623 | 0.614 | 0.655 | 0.486 | 0.646 |
| 16–25 m | 586 | 0.736 | 0.694 | 0.743 | 0.635 | 0.751 |
| 25+ m | 506 | 0.783 | 0.807 | 0.808 | 0.598 | 0.821 |
| **geometria niedostępna** | 519 | 0.707 | 0.727 | 0.724 | **0.558** | 0.743 |

Dwa wnioski produkcyjne: pas **11–16 m** (krawędź pola karnego) jest najtrudniejszy dla
wszystkich modeli, a na **23% zbioru, gdzie homografia lub detekcja piłki zawodzą**,
modele obrazowe degradują się łagodnie, podczas gdy pozycyjny przestaje działać.

### Karne i górny ogon rozkładu

| Model | średnie xG na karnych (n=30) | log-loss na karnych |
|---|---|---|
| C | 0.304 | 1.026 |
| D-L | 0.288 | 1.088 |
| G | 0.333 | 0.980 |
| GBM z `is_penalty` | 0.549 | 0.652 |
| **Ensemble finalny** | **0.574** | **0.632** |

Rzeczywista konwersja karnych: 0.767. Krzywa niezawodności pokazuje to samo
systemowo: modele obrazowe **kompresują górny ogon** — ich predykcje rzadko przekraczają
0.35, choć obserwowana częstość w najwyższym kubełku sięga 0.58. To jedyny powód, dla
którego model pozycyjny — mimo AUC 0.62 — zostaje w składzie produkcyjnym, i widać, że
ensemble tę lukę w dużej mierze zamyka.

## 7. Finalny ensemble (skład i wagi dobrane wyłącznie na walidacji)

`scripts/build_ensemble.py`: seedy uśredniane w przestrzeni logitów bez dopasowywania,
warianty łączone nieujemnym stackiem logistycznym uczonym **na walidacji**, a wybór
składu — po log-lossie z walidacji krzyżowej **grupowanej po meczach wewnątrz walidacji**,
żeby większy skład nie wygrywał samym dopasowaniem. Test dotknięty raz, na końcu.

Wybrany skład: **D-L z maską + fine-tune L/12 + GBM pozycyjny**, wagi [0.34, 0.79, 0.49].

| Model | AUC [95% CI] | BSS | log-loss | ECE |
|---|---|---|---|---|
| **Ensemble finalny** | **0.758 [0.734, 0.781]** | **11.7%** | **0.340** | 0.020 |
| Poprzedni ensemble C + D-L (zwykła średnia) | 0.746 [0.721, 0.768] | 8.4% | 0.351 | 0.013 |
| G fine-tune L/12 sam | 0.758 [0.733, 0.781] | 9.5% | 0.347 | 0.023 |

| Porównanie | ΔAUC [95% CI] | P | Δlog-loss [95% CI] | P |
|---|---|---|---|---|
| Nowy vs stary ensemble | **+0.0128 [+0.0014, +0.0245]** | 0.987 | **+0.0107 [+0.0058, +0.0157]** | **1.000** |
| Nowy ensemble vs G sam | +0.0000 [−0.0082, +0.0090] | 0.50 | **+0.0072 [+0.0034, +0.0113]** | **1.000** |

Drugi wiersz jest najciekawszy: **ensemble nie poprawia porządkowania względem samego
fine-tuningu (ΔAUC dokładnie 0), całą przewagę zdobywa na jakości prawdopodobieństw.**
To dokładnie to, czego należało oczekiwać po dołożeniu modelu pozycyjnego, który jako
jedyny widzi karne i sięga wysokich wartości xG.

## 8. Istotność różnic — zbiorczo

Nachodzące przedziały ufności **nie oznaczają** braku różnicy: modele są oceniane na
tych samych strzałach, więc właściwą wielkością jest różnica sparowana.

| Porównanie | ΔAUC [95% CI] | P(A lepszy) |
|---|---|---|
| D-L vs GBM pozycyjny | +0.107 [+0.080, +0.138] | **1.000** |
| G fine-tune vs D-L zamrożony | +0.019 [+0.001, +0.037] | **0.977** |
| Nowy vs stary ensemble | +0.013 [+0.001, +0.025] | **0.987** |
| G-L12 vs G-L4 | +0.004 [−0.005, +0.013] | 0.81 |
| D-L vs C | +0.004 [−0.015, +0.022] | 0.64 |
| C + maska vs C | −0.002 [−0.008, +0.005] | 0.32 |
| C + TTA vs C | −0.007 [−0.012, −0.001] | **0.014** (na niekorzyść) |

**Korekta wcześniejszego raportu:** `FINAL_SUMMARY.md` przedstawiał D (0.741) i C
(0.732) jako ranking. Sparowany test mówi, że są **nierozróżnialne** (P = 0.64).

## 9. Rekomendacja produkcyjna po tej rundzie

1. **Ścieżka główna: fine-tuning DINOv2-L z 4 odmrożonymi blokami** (`lr` 3e-4,
   mnożnik backbone'u 0.02), na klatkach z zamaskowaną nakładką. Bije zarówno
   zamrożoną głowę, jak i ConvNeXt.
2. **Model pozycyjny zostaje w składzie** mimo AUC 0.62 — tylko on domyka karne
   i górny ogon rozkładu, a jego wkład jest istotny (P = 1.000 na log-lossie).
3. **Nie wdrażać TTA** — zmierzone pogorszenie.
4. **Maskować nakładkę domyślnie** — darmowe, usuwa ryzyko domenowe.
5. **Monitorować spadek na nieznanych stadionach** przy wdrożeniu na nowe ligi;
   dla fine-tuningu to ~5 pp AUC.
6. Otwarte kierunki bez zmian: pozycje GT z SoccerTrack (odblokują odrzucone cechy
   „światło bramki" i GK), model wideo na klipach 8 s, więcej karnych w treningu.

## 10. Koszt tej rundy

~6 h GPU (RTX 4070 Ti SUPER) na 40 przebiegów treningowych, ~1 h CPU na audyty,
podgrupy i testy istotności. Wszystko wznawialne na poziomie (etap, seed).
