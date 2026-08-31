# Runda przedmerge'owa: wiarygodność i jakość wyników

Data: 2026-08-31. Plan: `../research/pre_merge_plan.md`. Test: 2 229 strzałów /
100 meczów, split po meczach, częstość bazowa 0.133.

Cel rundy nie brzmiał „podnieś AUC", tylko **„upewnij się, że dotychczasowe liczby są
uczciwe, a dopiero potem je poprawiaj"**. Poniżej wyniki w tej kolejności. Trzy z
czterech hipotez poprawy okazały się nieskuteczne i są tu opisane jako zamknięte
negatywy — to też jest wynik i oszczędza czas w produkcji.

---

## 1. Defekt repozytorium wykryty przy okazji (najważniejsza pojedyncza naprawa)

Główny `.gitignore` zawiera reguły `data/` i `models/`. Dopasowują się one na dowolnym
poziomie drzewa, więc **cały `xgpoc/data/` i `xgpoc/models/` był wykluczony z każdego
dotychczasowego commita** — 11 plików źródłowych, w tym wszystkie modele i datasety.
Branch w postaci zapisanej w gicie nie dawał się uruchomić; działał wyłącznie lokalnie.

Naprawa: negacje `!xgpoc/data/` i `!xgpoc/models/` w `.gitignore` eksperymentu.
Przy okazji ujawniło to trzy błędy lintera, których `ruff` wcześniej nie widział,
bo respektuje `.gitignore`.

## 2. Audyt przecieku między splitami — **PASS**

`scripts/audit_leakage.py`, wynik: `results/audit/leakage.json`.

| Kontrola | Wynik |
|---|---|
| Duplikaty `shot_id` | 0 |
| Wspólne mecze train/valid/test | 0 (300 / 100 / 100 meczów) |
| Ten sam mecz pod dwiema nazwami | 0 |
| Klatki nieczytelne / zdegenerowane | 0 / 0 |
| Klatki testowe z „bliźniakiem" w treningu | **0 (0.00%)** |
| Pary z tego samego meczu w różnych splitach | 0 |

**Metodologicznie ważne:** pierwsza wersja używała 64-bitowego dHasha i dała bezużyteczny
wynik — na transmisjach piłkarskich rozkład dystansów dla par *tej samej akcji*
(t0 vs −400 ms, mediana 13) nachodzi na rozkład par losowych (percentyl 1 = 21).
Każdy próg albo gubi prawdziwe bliźniaki, albo flaguje tysiące niezwiązanych ujęć
ogólnych. Werdykt oparto więc na cosinusie osadzeń CLS DINOv2, które separują się
czysto: ta sama akcja p5 = 0.941, pary losowe p99 = 0.921. Obie kalibracje zostają
w raporcie, żeby było widać, dlaczego wybrano to narzędzie.

**Znalezisko poboczne, nie przeciek:** 48% klatek testowych ma w treningu klatkę
„podobnej sceny" (cos ≥ 0.941). Inspekcja wzrokowa (`cross_split_pairs.jpg`) i analiza
par meczów pokazują, że to **ten sam stadion, inna akcja** — 85 ze 100 meczów testowych
ma drużynę gospodarzy występującą też w treningu. To kwestia generalizacji, nie
etykiet, i została zmierzona osobno (sekcja 6).

## 3. Scoreboard: czy model czyta boisko, czy pasek z wynikiem?

Nakładka telewizyjna zlokalizowana z danych (`scripts/find_overlay_region.py`): piksele,
które między t0 a t−1600 ms nie zmieniają się **w stosunku do swojego detalu
przestrzennego** — to odróżnia grafikę od jednolitej murawy, która też się „nie zmienia",
ale tylko dlatego, że nie ma czego zmieniać. Wykryte: scoreboard `[80, 38, 409, 95]`
i logo kanału `[1017, 31, 1210, 116]`.

### Sondy kontrolne (model C uczony wyłącznie na zamaskowanym wycinku)

| Sonda | Co widzi | AUC | BSS |
|---|---|---|---|
| `overlay_only` | grafika **+ cienkie statyczne pasy przy krawędziach kadru** | 0.606 | 1.1% |
| `graphics_only` | **tylko** scoreboard i logo kanału | **0.530** | **−0.2%** |

Pierwsza sonda wyglądała alarmująco: 0.606 AUC to tyle, co cały pipeline pozycyjny.
Druga rozstrzyga sprawę — **sama tablica wyników nie niesie sygnału** (BSS ujemny, czyli
model degeneruje się do stałej). Cały sygnał pierwszej sondy pochodził z pasów przy
krawędziach kadru, które zawierają bandy reklamowe i piksele murawy, czyli pośrednio
kadrowanie kamery. To zupełnie inna diagnoza niż „model czyta stan meczu".

### Retrening na zamaskowanych klatkach

| Model | AUC | BSS | log-loss | ΔAUC vs baza [95% CI] |
|---|---|---|---|---|
| C baza | 0.7361 | 8.0% | 0.3518 | — |
| C + maska | 0.7346 | 7.9% | 0.3520 | −0.0016 [−0.0080, +0.0048] |
| D-L baza | 0.7396 | 8.1% | 0.3533 | — |
| D-L + maska | 0.7353 | 8.2% | 0.3525 | −0.0042 [−0.0118, +0.0030] |

**Werdykt: maskowanie jest darmowe.** Nie ma mierzalnego kosztu, a usuwa całą klasę
ryzyka domenowego (inny realizator, materiał bez nakładki). Rekomendacja produkcyjna:
maskować domyślnie, mimo że w tym zbiorze nic to nie zmienia.

## 4. Fine-tuning DINOv2 zamiast samej głowy (PoC G)

`FinetuneViTXG`: te same wagi DINOv2, odmrożone ostatnie N bloków, gradient
checkpointing, osobny (niższy) LR dla backbone'u. Koszt okazał się dużo niższy niż
zakładałem: pełne odmrożenie ViT-L mieści się w 6.2 GiB i 3.4 min/epokę.

### Sweep głębokości (1 seed, wybór na log-lossie **walidacyjnym**)

| Konfiguracja | valid log-loss | test AUC | test BSS |
|---|---|---|---|
| **DINOv2-L, 4 bloki** | **0.3987** | 0.7465 | 6.5% |
| DINOv2-L, 12 bloków | 0.4221 | 0.678 | −0.2% |
| DINOv2-L, 24 bloki (całość) | 0.4219 | 0.574 | −0.2% |
| DINOv2-B, 12 bloków | 0.4220 | 0.553 | −0.2% |

Głębsze odmrażanie **zapada się do stałej predykcji** (BSS ≈ 0, log-loss równy
bazowemu). Selekcja na walidacji odrzuciła te warianty poprawnie, bez zaglądania
do testu.

### Wynik zwycięskiej konfiguracji (3 seedy)

| Model | AUC | BSS | log-loss | ECE |
|---|---|---|---|---|
| D-L zamrożony (baza) | 0.7396 | 8.1% | 0.3533 | 0.014 |
| **G fine-tune L/4** | **0.7474** | 8.0% | 0.3548 | 0.023 |
| G fine-tune + maska + TTA | 0.7440 | 8.3% | 0.3535 | 0.019 |

Test sparowany po meczach: **ΔAUC +0.0079 [−0.0088, +0.0242], P(G > D) = 0.83,
log-loss nieistotnie gorszy.** Fine-tuning poprawia porządkowanie, ale **nie w sposób
istotny**, a pogarsza kalibrację (ECE 0.014 → 0.023) i kosztuje 20 min/seed zamiast
40 s/seed nad zamrożonymi cechami.

## 5. TTA (odbicie lustrzane) — **hipoteza odrzucona**

| Wariant vs jego baza | ΔAUC [95% CI] | P(lepszy) | Δlog-loss [95% CI] |
|---|---|---|---|
| C + TTA vs C | **−0.0065 [−0.0121, −0.0006]** | **0.014** | −0.0022 [−0.0042, −0.0002] |
| D-L + odbicie (aug + TTA) vs D-L | +0.0014 [−0.0058, +0.0089] | 0.65 | +0.0013 [−0.0008, +0.0034] |
| D-L + maska + TTA vs D-L + maska | −0.0030 [−0.0114, +0.0048] | 0.23 | **−0.0035 [−0.0064, −0.0007]** |
| G + maska + TTA vs G | −0.0032 [−0.0145, +0.0076] | 0.27 | +0.0014 [−0.0016, +0.0041] |

Dla C odbicie **istotnie szkodzi** obu metrykom; dla D z maską istotnie pogarsza
log-loss. Jedyny dodatni przypadek (D bez maski, gdzie odbicie działa też jako
augmentacja treningu głowy) mieści się w szumie seedów.

Interpretacja: C i głowa D uczą się już z losowym odbiciem, więc TTA nie dokłada
informacji — uśrednia natomiast logity dwóch widoków, co spłaszcza pewne predykcje
i psuje kalibrację. **Nie wdrażać.**

## 6. Ocena na podgrupach

Pełna tabela: `results/subgroups/subgroups.csv`, krzywe: `figures_premerge/reliability.png`.

### Pamiętanie stadionów — **nie napędza wyników**

| Model | AUC (gospodarz znany z treningu, n=1880) | AUC (gospodarz nieznany, n=349) |
|---|---|---|
| C | 0.7356 | 0.7371 |
| D-L | 0.7415 | 0.7309 |
| **G fine-tune** | **0.7549** | **0.7033** |
| GBM pozycyjny | 0.6236 | 0.6185 |

C i D generalizują na nieznane stadiony bez spadku. **Fine-tuning traci 5 pp AUC na
nieznanych stadionach** — to dodatkowy, niezależny argument przeciw niemu: jego i tak
nieistotny zysk koncentruje się tam, gdzie model widział już to miejsce.

### Gdzie modele działają, a gdzie nie

| Podgrupa | n | C | D-L | GBM | Komentarz |
|---|---|---|---|---|---|
| 0–6 m | 120 | 0.756 | **0.838** | 0.587 | obraz widzi ustawienie bramkarza |
| 6–11 m | 208 | 0.716 | 0.729 | 0.669 | |
| 11–16 m | 290 | 0.633 | 0.625 | 0.487 | **najtrudniejszy pas dla wszystkich** |
| 16–25 m | 586 | 0.738 | 0.707 | 0.635 | |
| 25+ m | 506 | 0.788 | 0.801 | 0.598 | |
| geometria niedostępna | 519 | 0.708 | 0.721 | **0.558** | 23% zbioru; modele obrazowe degradują się łagodnie tam, gdzie pipeline pozycyjny przestaje działać |

### Karne — najsłabszy punkt modeli obrazowych

| Model | średnie xG na karnych (n=30) | rzeczywista konwersja |
|---|---|---|
| C | 0.294 | 0.767 |
| D-L | 0.275 | 0.767 |
| G | 0.261 | 0.767 |
| **GBM z `is_penalty`** | **0.549** | 0.767 |

Krzywa niezawodności pokazuje to samo systemowo: modele obrazowe **kompresują górny
ogon** — ich predykcje prawie nie przekraczają 0.35, podczas gdy obserwowana częstość
w najwyższym kubełku sięga 0.58. To główny powód, dla którego model pozycyjny — mimo
AUC 0.62 — zostaje w składzie produkcyjnym.

## 7. Istotność różnic (sparowany bootstrap po meczach, 2 000 losowań)

Nachodzące przedziały ufności nie oznaczają braku różnicy: modele oceniane są na tych
samych strzałach, więc właściwą wielkością jest różnica sparowana.

| Porównanie | ΔAUC [95% CI] | P(A lepszy) | Δlog-loss |
|---|---|---|---|
| D-L vs C | +0.0035 [−0.0148, +0.0220] | 0.64 | −0.0015 |
| G vs D-L | +0.0079 [−0.0088, +0.0242] | 0.83 | −0.0016 |
| G vs C | +0.0111 [−0.0079, +0.0299] | 0.88 | −0.0031 |
| C+maska vs C | −0.0016 [−0.0080, +0.0048] | 0.32 | −0.0002 |
| C+TTA vs C | −0.0065 [−0.0121, −0.0006] | **0.014** | −0.0022 |
| D-L vs GBM pozycyjny | +0.107 [+0.080, +0.138] | **1.000** | +0.017 |

**Korekta wcześniejszego raportu:** `FINAL_SUMMARY.md` przedstawiał D (0.741) i C
(0.732) jako ranking. Sparowany test mówi, że są **nierozróżnialne** (P = 0.64).
Jedyna różnica istotna w całym eksperymencie to przewaga modeli obrazowych nad
pozycyjnymi.
