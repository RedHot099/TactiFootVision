# Plan prac przed merge: wiarygodność i jakość wyników

Data: 2026-08-31. Branch `claude-fable/xg-poc`. Stan wyjściowy: rekomendacja C
(ConvNeXt fine-tune) + D (zamrożony DINOv2-L + głowa), ensemble AUC 0.746, BSS 8.4%,
log-loss 0.351 na teście 2 229 strzałów / 100 meczów.

Cel tej rundy: **najpierw upewnić się, że liczby są uczciwe, potem je poprawić,
na końcu opisać rzetelnie.** Kolejność jest celowa — nie ma sensu optymalizować
wyniku, który mógłby pochodzić z przecieku albo z grafiki telewizyjnej.

## W1. Audyt przecieku między splitami (bramka odbioru)

**Problem.** Split jest po meczach (hash `game`), ale nigdy nie zweryfikowaliśmy
tego empirycznie na poziomie pikseli. Ryzyka: (a) ten sam mecz pod dwoma nazwami
w dwóch splitach, (b) powtórki/ujęcia studyjne wspólne dla wielu meczów,
(c) duplikaty `shot_id`, (d) klatki puste/czarne udające sygnał.

**Metoda.** dHash 64-bit każdej klatki t0 (9×8 skala szarości, różnice poziome).
Porównanie każdej klatki testowej z każdą treningową/walidacyjną — 2 229 × 8 801
par, popcount na spakowanych bitach (numpy, sekundy). Kalibracja progu: rozkład
dystansów Hamminga dla par *wewnątrz tego samego strzału* (t0 vs −400 ms) daje
skalę „to samo ujęcie", rozkład par losowych daje skalę „różne ujęcia".

**Kryterium odbioru.** Zero par cross-split z tego samego meczu; udział klatek
testowych z bliźniakiem w treningu < 0.5 %; brak duplikatów `shot_id`; zero klatek
zdegenerowanych (stała jasność). Wynik → `results/audit/leakage.json` + raport.

**Ryzyko.** dHash łapie podobieństwo globalne, nie semantyczne — trafienia z różnych
meczów (np. dwa dalekie ujęcia trybun) będą fałszywie dodatnie i wymagają obejrzenia.
Dlatego raport zapisuje kolaż 20 najbliższych par do inspekcji.

## W2. Scoreboard: czy model czyta boisko, czy pasek z wynikiem?

**Problem.** Każda klatka broadcastu zawiera nakładkę z wynikiem i czasem meczu.
Model może uczyć się stanu meczu (przy 0:3 gra się inaczej niż przy 0:0) — to nie
jest przeciek etykiety, ale **nie generalizuje** na inny realizator ani na materiał
bez nakładki. Nie wiemy, ile z 0.73 AUC to analiza boiska.

**Metoda, trzy kroki.**
1. *Lokalizacja nakładki, sterowana danymi.* Dla ~400 losowych klatek liczymy
   profil wierszowy średniej wariancji międzyklatkowej wewnątrz meczu; nakładka
   jest statyczna, więc daje wyraźne minimum. Wynik: wysokość pasa do maskowania
   (spodziewane ~8–14 % górnej części kadru) + wykres profilu.
2. *Sonda „tylko scoreboard".* Model C uczony **wyłącznie** na wyciętym pasie.
   Jeśli AUC ≈ 0.5 — nakładka nie niesie sygnału i problem znika. Jeśli AUC > 0.55 —
   mamy potwierdzoną zależność od grafiki i punkt odniesienia, ile jej ubyło.
3. *Retrening z maskowaniem.* C (3 seedy) i D-L (ekstrakcja cech z zamaskowanego
   kadru + głowa, 3 seedy) na klatkach z wyzerowanym pasem. Porównanie z bazą.

**Interpretacja z góry** (żeby nie naginać jej do wyniku): spadek AUC po maskowaniu
< 0.005 → nakładka nieistotna, zostawiamy maskowanie jako domyślne, bo jest darmowe;
spadek 0.005–0.02 → część sygnału była w grafice, raportujemy obie liczby, produkcyjną
robi wersja maskowana; spadek > 0.02 → poważne ostrzeżenie, główną liczbą eksperymentu
staje się wersja maskowana.

## W3. Fine-tuning DINOv2 zamiast samej głowy

**Przesłanka.** Skalowanie backbone'u było czyste i monotoniczne (S 0.713 → B 0.722
→ L 0.741 AUC) przy trenowaniu jedynie 0.6 M parametrów głowy nad zamrożonymi cechami.
To najmocniejsza przesłanka, że backbone ma jeszcze zapas, którego głowa nie sięga.

**Metoda.** `FinetuneViTXG` w `d_foundation.py`: DINOv2 ViT-B/L, wejście 448×252,
odmrożone ostatnie N bloków + głowa (attention-pool, ta sama co w D), gradient
checkpointing, warstwowy zanik LR (backbone 10× niższy LR niż głowa), OneCycle, AMP.
Wariant B/last-4 jako tani test poprawności ścieżki, potem L/last-6.
Augmentacja jak w C (flip + jitter fotometryczny), bo teraz uczymy na obrazach.

**Budżet.** ViT-B ≈ 20 min/seed, ViT-L ≈ 45–60 min/seed przy batch 8. 3 seedy dla
zwycięskiego wariantu. Twardy limit `max_minutes` chroni przed przekroczeniem.

**Kryterium sukcesu.** Ensemble 3 seedów bije zamrożone D-L (0.741) o ≥ 0.005 AUC
przy nie gorszym log-loss. Inaczej: raportujemy jako zamkniętą hipotezę („głowa nad
zamrożonym L wystarcza"), co samo w sobie jest wynikiem dla produkcji (tańsza inferencja).

**Ryzyko.** OOM przy L/batch 8 — mitygacja: gradient checkpointing, w razie potrzeby
batch 4 + akumulacja. Przeuczenie przy 6 657 przykładach — mitygacja: mało odmrożonych
bloków, niski LR backbone'u, early stopping na log-lossie walidacyjnym (już jest).

## W4. TTA i ensemble ważony

**TTA.** Lustrzane odbicie jest dla tego zadania poprawną symetrią (bramka atakowana
może być po obu stronach) i jest już używane jako augmentacja w C. Dla C wystarczy
uśrednić logity z kadru i jego odbicia. Dla D cechy są liczone raz i zamrożone, więc
TTA wymaga **drugiego cache'u cech dla odbitych klatek**; wtedy odbicie staje się też
augmentacją treningu głowy (podwojony zbiór), co jest osobnym, sensownym zyskiem.

**Ensemble.** Dotąd używaliśmy prostej średniej. Teraz: predykcje **walidacyjne**
zapisywane obok testowych, wagi dobierane na walidacji (regresja logistyczna na
logitach, wagi nieujemne), ocena wyłącznie na teście. To eliminuje zarzut, że skład
ensemble'u był dobierany na teście.

## W5. Ocena na podgrupach i kalibracja

Metryki w rozbiciu: dystans (kubełki 0–6/6–11/11–16/16–25/25+ m, tylko przy poprawnej
homografii), liga, sezon, `visibility`, karne vs gra otwarta, kwartyl `grass_ratio`
(proxy typu ujęcia). Do tego krzywe niezawodności (reliability diagram) dla finalnych
modeli. Cel: wykryć, czy przewaga modeli obrazowych nie pochodzi z jednej ligi lub
jednego typu kadru, i gdzie kalibracja się psuje.

## W6. Istotność różnic

Obecnie mamy tylko nachodzące przedziały ufności, z których **nie wynika** brak
istotności — modele są oceniane na tych samych strzałach, więc różnica jest sparowana.
Wprowadzam sparowany bootstrap po meczach: losujemy mecze, liczymy Δ metryki dla
pary modeli, raportujemy medianę, CI i udział losowań z Δ > 0. Pary: C vs D-L,
najlepszy pojedynczy vs ensemble, baza vs wersja maskowana.

## W7. Higiena przed merge

Usunięcie martwej ścieżki `dl672` z `run_night.sh` (3× nieudana, wymaga przepisania
ekstraktora na zapis strumieniowy — opisane jako known issue), synchronizacja
`FINAL_SUMMARY.md` i `README.md` z ostatecznymi liczbami, sekcja „jak odtworzyć
wyniki od zera", testy jednostkowe dla nowych modułów, `ruff check`.

## Kolejność wykonania

W1 → W2 (kroki 1–2) → W3 (start treningu w tle) → W2 krok 3 → W4 → W5 → W6 → W7.
Treningi GPU idą sekwencyjnie, analizy CPU wykonują się w przerwach.
