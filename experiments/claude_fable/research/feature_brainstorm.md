# Burza mózgów: jakie cechy dodać do modeli xG

Data: 2026-08-31. Trzy niezależne wejścia: (1) przegląd literatury z efektami
(agent A), (2) adwersarialny audyt wykonalności przy naszej jakości ekstrakcji
(agent B), (3) empiryczna permutation importance obecnych 20 cech (GBM, valid,
neg-log-loss). Kontekst wyników: obrazowe AUC 0.73-0.75, pozycyjne 0.61.

## 1. Empiryka: co dziś naprawdę pracuje w modelu pozycyjnym

```
n_players              0.0109   <- w dużej mierze proxy kadru kamery (zbliżenie/kontra)
angle_to_goal          0.0056
gk_dist_to_ball        0.0034
n_keypoints            0.0012
gk_dist_to_goal_line   0.0009
dist_to_goal           0.0008
nearest_defender_dist  0.0008
...
n_defenders_in_cone   -0.0003   <- ZERO sygnału
n_opponents_within_3m -0.0003
ball_x / ball_y_abs   <0        <- czysty szum
```

Wniosek krytyczny: **cechy obronne już istnieją i nie działają** — nie dlatego, że
są złym pomysłem, tylko dlatego, że toną w szumie wejść (piłka 49-76% pokrycia,
homografia 61-81%, drużyny znane ~50-76%, GK wykrywany w ~20% klatek, ~1 s szumu
czasowego anotacji, przez który obrońcy przemieszczają się o 3-6 m). Wykryty przy
audycie **bug wart naprawy**: gdy drużyna strzelca jest nieznana, wszyscy gracze
z pola liczeni są jako obrońcy (`tabular.py`), co systematycznie zawyża cechy obronne.

## 2. Literatura: ile realnie dają cechy freeze-frame

- Pakiet cech pozycyjnych (GK + obrońcy w stożku + zasłonięcie bramki + presja)
  daje na CZYSTYCH danych trackingowych **+0.03-0.05 AUC** nad model eventowy
  (Anzer & Bauer 2021: 0.772→0.823; KU Leuven: 0.782→0.805).
- Ranking SHAP u A&B: (1-2) dystans+kąt, (3) **pozycja bramkarza**, (4) presja na
  strzelcu, (5) obrońcy w stożku.
- "Światło bramki" nigdy nie występuje solo — zawsze w pakiecie z GK/obrońcami;
  StatsBomb w 2022 przeszedł z dyskretnych "blockerów" na **rozmyte gaussowskie
  (80 cm gracz / 160 cm GK)**, bo wersja binarna psuła kalibrację na krawędziach.
- Duży niedoceniany blok: **typ akcji przed strzałem** (dośrodkowanie / prostopadła /
  cutback / dobitka) — model sekwencyjny 0.826 vs 0.800-0.822; u nas częściowo
  dostępny z klipów 8 s, bez anotacji eventowych.

## 3. Ocena czterech propozycji użytkownika

| Propozycja | Werdykt | Uzasadnienie |
|---|---|---|
| 1. Dystanse do narożników bramki | **odrzucić** | Deterministycznie wyliczalne z (dist, kąt, y) które już mamy; GBM tę informację posiada. Zero nowego sygnału. |
| 2. Światło bramki (zasłonięta powierzchnia) | **budować — najlepszy pomysł z czwórki** | Jedyna propozycja z nowym sygnałem (literatura: część pakietu +0.02-0.03). Kluczowa decyzja implementacyjna: liczyć **w przestrzeni obrazu** (słupki z keypointów boiska + boxy graczy), nie z homografii — omija 19-39% jej awarii i podnosi pokrycie do ~70%+. Wersja rozmyta (Gauss), nie binarna. |
| 3. Liczba zawodników między piłką a bramką | **już mamy** (`n_defenders_in_cone`) | Zmierzona ważność ≈ 0 przy obecnej ekstrakcji; wariant "korytarz" to niemal duplikat stożka. Naprawa = lepsze wejścia, nie nowa definicja. |
| 4. Dystans do najbliższego obrońcy | **już mamy** (`nearest_defender_dist`) | Ważność ~0.0008; próg presji 1.5 m jest poniżej podłogi szumu (sam błąd homografii na dystansie go przekracza). |

## 4. Rekomendowany pakiet (ranking wartość/koszt)

**P0 — wdrożyć od razu:**
1. **`is_penalty` jako wejście** (godzina pracy, pokrycie 100%, zero zależności od
   ekstrakcji). Naprawia jedyny udokumentowany systematyczny błąd naszych modeli:
   karne dostają 0.42-0.46 zamiast ~0.76 (patrz `figures/sytuacja_5.png`). Uwaga
   wdrożeniowa: w produkcji flaga musi być wykrywalna (geometria: piłka na wapnie,
   pusty stożek) albo podawana z zewnątrz — inaczej train/serve mismatch.
2. **Światło bramki w przestrzeni obrazu** (propozycja #2, wersja gaussowska):
   trapez bramki z keypointów linii bramkowej → rzuty boxów graczy z perspektywy
   piłki → % zasłonięcia. Oczekiwanie: +0.01-0.02 AUC solo w GBM; główna wartość
   jako wejście do modeli obrazowych i cecha interpretowalna.
3. **GK-proxy + naprawa bugu drużyn**: bramkarz = najgłębszy gracz drużyny broniącej
   (pokrycie 20%→~55%), do tego lateral offset GK od linii piłka-bramka; plus fix
   "wszyscy = obrońcy przy nieznanej drużynie". Literatura: pozycja GK to #3 cecha
   na świecie; u nas `gk_dist_to_ball` już jest #3 mimo 20% pokrycia — to
   najbardziej niedoinwestowana żyła.

**P1 — warte testu po P0:**
4. **Proxy główki / piłka w powietrzu** (obraz: środek piłki powyżej linii głów
   sąsiadów) — headery to duży czynnik xG (StatsBomb SIH: ±0.25-0.38 xG na
   dobitkach/wolejach); podwójna rola: flaga ważności `ball_xy` (piłka w powietrzu
   psuje homografię płaszczyzny boiska).
5. **Prędkość ataku** (przesunięcie centroidu graczy z klatek −1600..−800 ms;
   NIE piłki i NIE z par zawierających t0 — ryzyko przecieku wyniku przy ~1 s szumu
   anotacji). Proxy kontrataku, ważne u A&B.
6. **Typ akcji przed strzałem z klipu** (dośrodkowanie/dryb/prostopadła) — najtaniej
   nie jako ręczna cecha, lecz przez model klipowy; wracamy do tego przy pracy nad
   klipami.

**P2 — odrzucone (z powodem):** dystanse do słupków (redundantne), korytarz
(duplikat), presja @1.5 m (poniżej szumu), detekcja muru (rzadkość zdarzeń,
obrazowe już to widzą), wysokość piłki w metrach (niewyliczalna z monokularu).

## 5. Najważniejsza decyzja architektoniczna

Nie wlewać tych cech przede wszystkim do modeli pozycyjnych (sufit ~0.61-0.65 przy
obecnej ekstrakcji; fallback piłki +27 pp pokrycia dał <1 pp AUC). Zamiast tego:
**konkatenować mało-wymiarowy, wysokoprecyzyjny wektor cech (P0+P1, z flagami
brakujących wartości) do głowy zwycięskiego modelu D (DINOv2-L)** — retrening głowy
to 40 s, więc każdą cechę ablacjonujemy osobno przeciw bazie 0.741. Przestroga z
własnych wyników: hybryda E (0.707) pokazała, że zaszumione wejścia potrafią
obrazowemu modelowi SZKODZIĆ — stąd tylko cechy o wysokiej precyzji, nie pełny
20-wymiarowy wektor. Realistyczne oczekiwanie: głównie poprawa kalibracji i kieszeni
(karne, główki, sytuacje sam na sam), +0.005-0.015 AUC łącznie.

Uczciwy re-test pełnego potencjału cech pozycyjnych (w tym #2-#4 użytkownika w
czystej formie) wymaga pozycji ground-truth — czeka na kohortę SoccerTrack (faza 2).

## 6. Plan walidacji

1. P0.1 (`is_penalty` do głowy D + do FEATURE_NAMES): oczek. spadek log-loss,
   xG karnych → 0.6-0.75; test kalibracji per podgrupa.
2. P0.2-0.3: każda cecha osobno do głowy D (3 seedy, 2 min każda) + do GBM;
   raport ΔAUC/Δlog-loss z CI; cecha zostaje tylko, gdy nie pogarsza bazy.
3. Rejestr pokrycia każdej cechy (missingness) per liga — kontrola biasu.
