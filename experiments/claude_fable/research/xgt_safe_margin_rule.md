# Sprawdzenie marginesu o_max = −600 ms: reguła zamrożona (xg-temporal 02)

Zapisana i zacommitowana 2026-10-25, przed jakimkolwiek oznaczaniem (pierwsza wersja 999fcec). Poprawka
z 2026-10-25, nadal przed oznaczaniem, decyzją orkiestratora: tabela dowodów bierze tylko kontakty v1.5 z meczów
`training`, a GSR jest opisany jako osobny zbiór. Próbka, definicja naruszenia, reguła decyzji, materiał i
instrukcja dla oznaczającego są bez zmian. sha256 tego pliku jest w
`research/xgt_safe_margin_rule.md.sha256` (ten sam commit), a `scripts/xgt_safe_margin.py` odmawia pracy, gdy plik się
nie zgadza. Spec: `.scratch/xg-temporal/spec.md` („Dowody → Kiedy jest kontakt”, „Kontrola przecieku” 2, decyzje
właściciela z 2026-10-24 D1, D2, D6). Kod: `scripts/xgt_safe_margin.py`, testy `tests/test_xgt_safe_margin.py`.

## Co sprawdzamy

o_max = −600 ms jest ustalone (D1). Ta reguła tylko sprawdza, czy klatka −600 ms pokazuje grę ściśle przed kontaktem
strzału. Wynik to „przechodzi” albo „stop”. Reguła nie wybiera o_max. o_max może się przesunąć tylko na wcześniej i
tylko decyzją właściciela, nigdy na później.

## Próbka

- **Populacja uprawniona.** Strzały roli `training` z `results/xgv_split_plan/plan.csv` (sha256
  `a56054ceb1f89d060352be2fdb8b4649363aae9fd3ff017322f65711bdc5d16a`, czytane kolumny `game`, `role`), bez karnych.
  Strzały i karne z `cache/xgc/offset_frames_v1/shots.csv` (sha256 `57db9d2feed85cc61bc2fdf8926163191a95339166cb9f5a8de05148b63dd367`, zgodny z `complete.json` cache; czytane wyłącznie kolumny `shot_id`,
  `game`, `is_penalty`; kolumn wyniku się nie czyta). Klatki z `cache/xgc/offset_frames_v1/frames.csv` (sha256
  `783df0b2b20740277c96400e93f6108ec84ec9bf867aebb93ca2138e86b7661d`, zgodny z `complete.json` cache; kolumny `shot_id`,
  `offset_ms`, `status`, `cut_between_anchor`). Strzał jest uprawniony, gdy klatki −800, −600, −400 i 0 ms mają status
  `ok`, a klatka −600 ms nie ma flagi cięcia między sobą a kotwicą (czyli w przedziale (−600, 0] nie ma cięcia).
  Uprawnionych jest **6 924 strzały z 328 meczów**. Ograniczenie jest celowe: sprawdzenie dotyczy klatek, które
  wejdą do okna, a strzał bez użytecznej klatki −600 ms takiej klatki do okna nie wnosi.
- **Losowanie.** Identyfikatory posortowane rosnąco, `numpy.random.default_rng(20261025).choice(6924, 120,
  replace=False)`. **Seed 20261025.** Kolejność losowania to kolejność arkuszy `s001`…`s120`; partie po 30 (1: s001–s030,
  2: s031–s060, 3: s061–s090, 4: s091–s120).
- **Lista.** `research/xgt_safe_margin_sample.csv` (kolumny `sheet_id`, `batch`, `shot_id`, `match`), **120 strzałów
  ze 107 meczów**, sha256 `1f9d38cd9b3ebd8cf9f87f9eb75815a9d2ed683db74cc7e548a7761402c20e5e` (stała `SAMPLE_SHA256`
  w skrypcie). Polecenie: `./run.sh --no-sync python scripts/xgt_safe_margin.py sample`.

Próbka, seed i lista są zamrożone. Po rozpoczęciu oznaczania nie wolno losować ponownie, dobierać strzałów, usuwać
arkuszy ani oznaczać ponownie.

## Materiał dla oznaczającego

`./run.sh --no-sync python scripts/xgt_safe_margin.py sheets` zapisuje poza gitem, w `outputs/xgt_safe_margin/`:

- `sheets/<sheet_id>.png`: arkusz 4 × 6 kafelków po 448 × 252 px (rozmiar klatek cache) z podpisem offsetu nad
  każdym kafelkiem. Offsety: −1600, −1200, −1000 oraz co 40 ms od −800 do 0 ms (24 kafelki). Podpisy trzech offsetów
  pytań (−800, −600, −400) są podświetlone i mają dopisek „<- question”. „CUT” (z czerwoną ramką) oznacza flagę
  cięcia między klatką a kotwicą, „MISSING” brak klatki. Nagłówek: `sheet_id` i zdanie, że nie ma klatek po 0 ms.
- `frames/<sheet_id>/m1600.jpg` … `m0040.jpg`, `p0000.jpg`: te same 24 klatki w natywnej rozdzielczości 1280 × 720
  (JPEG, jakość 95). W klatkach 448 × 252 piłka bywa niewidoczna, stąd klatki natywne i lupa.
- Lupa: `scripts/xgt_safe_margin.py zoom` wycina ten sam fragment (domyślnie 320 × 180 px, czyli powiększenie 2×)
  z klatek natywnych przy wybranych offsetach ≤ 0 i zapisuje go jako jeden obrazek.
- `batches.csv`: `sheet_id`, `batch`, `sheet`, `native_frames`. `sheets_manifest.json`: hashe arkuszy, offsety,
  wynik kontroli zgodności z cache.

Klatki pochodzą z klipów 720p, dekodowanych tylko do klatki kotwicy (klatka po 0 ms nigdy nie jest dekodowana).
Wybór klatki i flaga cięcia to reguła `xgpoc.data.offset_frames`. Każda klatka z siatki cache (16 offsetów ≤ 0) jest
sprawdzana z `frames.csv`: indeks klatki, flaga cięcia i skrót pikseli natywnych muszą być równe, inaczej skrypt
staje. Nazwy plików i podpisy zawierają tylko anonimowy `sheet_id` (bez identyfikatora strzału, meczu i wyniku).

Oznaczający nie dostaje: `shot_id`, nazwy meczu, listy próbki, żadnej klatki > 0 ms, wyniku strzału, odpowiedzi
drugiego przebiegu, tabeli istniejących dowodów ani pozostałej treści tej reguły. Dostaje wyłącznie tekst instrukcji
z sekcji „Instrukcja dla oznaczającego”.

Agent przygotowujący (faza A) obejrzał w próbie czytelności wyłącznie arkusz `s001`, jego klatkę natywną −600 ms i
jedną lupę `s001`. Niczego nie oznaczał i nie zapisał żadnej oceny.

## Definicja naruszenia

Strzał jest **uderzony w offsecie o**, gdy kontakt strzałowy (uderzenie piłki STRZAŁEM, tym, którego dotyczy kotwica)
nastąpił w chwili klatki o albo wcześniej. **Naruszenie = strzał uderzony w −600 ms.**

Strzałem jest próba zdobycia bramki: kopnięcie albo uderzenie głową w stronę bramki przeciwnika przez zawodnika
drużyny atakującej. Gdy do 0 ms widać kilka uderzeń, strzałem jest ostatnie uderzenie w stronę bramki wykonane do
0 ms (przy strzale zablokowanym i dobitce: dobitka). Gdy do 0 ms nie widać uderzenia w stronę bramki, strzał pada po
kotwicy i we wszystkich trzech offsetach odpowiedź brzmi „no”. **Nie są strzałem:** podania, dośrodkowania, rzuty
rożne i wolne zagrane do partnera, wrzuty z autu, prowadzenie piłki i przyjęcia, zgrania głową do partnera, odbicia i
rykoszety, interwencje obrońców i bramkarza. Piłka lecąca po takim zagraniu nie jest uderzona strzałem. Doprecyzowanie
jest celowe: reguła `xgpoc/detector/kick.py` w GSR brała pierwsze kopnięcie w oknie i trzy najwcześniejsze jej
„kopnięcia” (−880, −840, −640 ms) to prawdopodobnie podania.

## Odpowiedzi i najgorszy przypadek

- Dwa niezależne przebiegi AI, A i B (D2), bez udziału właściciela. Każdy odpowiada przy każdym arkuszu i każdym z
  offsetów −800, −600, −400 jednym z: `yes` (tak), `no` (nie), `unknown` (nie wiem).
- **Najgorszy przypadek w przebiegu.** Bycie uderzonym jest monotoniczne w czasie. Przebieg ma „uderzony w o”, gdy
  jego odpowiedź w o to `yes` albo `unknown` albo gdy przy którymkolwiek wcześniejszym offsecie pytania odpowiedział
  `yes` (wewnętrzna sprzeczność liczy się jako uderzenie najwcześniejsze). `no` w o przy `unknown` wcześniej nie jest
  uderzeniem: „nie” w −600 rozstrzyga też −800.
- **Najgorszy przypadek między przebiegami.** Strzał narusza margines, gdy którykolwiek przebieg ma „uderzony w
  −600”. Zatem „nie wiem” i niezgodność przebiegów liczą się jako uderzenie najwcześniejsze, czyli naruszenie przy
  −600.
- Każdy z 120 arkuszy musi mieć odpowiedź obu przebiegów; `evaluate` odmawia przy brakach, duplikatach i wartościach
  spoza `yes`/`no`/`unknown`. Arkusz, którego nie da się ocenić, dostaje `unknown`.

## Reguła decyzji

k = liczba naruszeń przy −600 na n = 120. U = jednostronna górna granica 95% Cloppera–Pearsona udziału naruszeń,
U = Beta⁻¹(0,95; k + 1, n − k) (U = 1 przy k = n). **−600 przechodzi, gdy U ≤ 5%.**

| k (z 120) | U |
|---|---|
| 0 | 2,47% |
| 1 | 3,89% |
| 2 | 5,15% |
| 3 | 6,33% |

Zatem **0 albo 1 naruszenie przechodzi, 2 i więcej to stop.**

- **Przechodzi:** ticket 03 używa o_max = −600 bez dyskusji.
- **Stop:** faza staje, ticket 03 jest zablokowany, pytanie wraca do właściciela (w Comments ticketu 02). Bez ponownego
  oznaczania, nowej próbki ani zmiany reguły po obejrzeniu odpowiedzi. o_max może się przesunąć tylko na wcześniej i
  tylko decyzją właściciela.
- **−800 i −400 są opisowe:** ta sama reguła najgorszego przypadku, liczba „uderzonych” i granica U; nie zmieniają
  o_max.

## Zgodność przebiegów

Per offset (−800, −600, −400): udział arkuszy z identyczną odpowiedzią (trzy kategorie), kappa Cohena na kategoriach
`yes`/`no`/`unknown` (nieokreślona, zapisana jako `null`, gdy zgodność przypadkowa wynosi 1), liczba niezgodności,
liczba „unknown” w którymkolwiek przebiegu, liczba „yes” w obu przebiegach i rozkład odpowiedzi każdego przebiegu.
Też liczba arkuszy z wewnętrznie sprzeczną odpowiedzią w każdym przebiegu. Zgodność jest raportowana, nie zmienia
wyniku.

## Niezależność przebiegów

- Każdy przebieg prowadzą świeże agenty AI, bez kontekstu tej fazy przygotowania i bez kontekstu drugiego przebiegu.
  Przebieg może być podzielony na 4 partie oznaczane przez osobne agenty; tekst instrukcji jest dla wszystkich
  identyczny, różnią się tylko podstawienia `{RUN}` (A albo B), `{B}` (1–4) i wynikające z nich ścieżki.
- Odpowiedzi: `outputs/xgt_safe_margin/answers/run_a/batch_<B>.csv` i `.../run_b/batch_<B>.csv`. Przebieg B nie czyta
  katalogu `run_a` (i odwrotnie). Model i data każdego przebiegu trafiają do Comments ticketu 02.
- Lupa zapisuje do `outputs/xgt_safe_margin/zoom/run_<run>/`, osobno dla przebiegów.

## Istniejące dowody (opisowo, poza regułą)

`evaluate` zapisuje jedną tabelę z pochodzeniem (`evidence.csv`):
- 11 kopnięć GSR z `results/pk_ball_evidence/timing.csv` (czytane kolumny `split`, `clip`, `kick_ms`; bez
  `action_class`). Kopnięcia ≤ −600 ms (−880, −840, −640) mają adnotację „może być poprzedzającym podaniem: kick.py bierze
  pierwsze kopnięcie w oknie, niesprawdzone”. GSR to osobny zbiór (SoccerNet Game State Reconstruction 2024,
  `data/SoccerNetGS`, 30-sekundowe sekwencje `SNGS-*` z własną numeracją meczów), a nie mecze SoccerNet z
  `plan.csv`, więc żadne z tych kopnięć nie pochodzi ze 172 meczów wstrzymanych.
- 12 ograniczonych kontaktów AI v1.5 z meczów roli `training`, z `cache/v15/canonical_data_v1/shots.csv` (czytane
  wyłącznie kolumny `shot_id`, `game`, `contact_status`, `contact_quality`, `contact_lower_ms`, `contact_upper_ms`;
  żadnej kolumny wyniku ani `ai_outcome`). Kontrola przecieku 4: żaden wiersz 172 meczów nie wchodzi do odczytu.
  Pozostałe 5 ograniczonych kontaktów v1.5 leży w meczach roli `evaluation`. Są tylko policzone po kolumnie `game`;
  ich linie są pomijane przy parsowaniu, więc ich czasy nie są wczytywane. Tabela i `complete.json` zapisują
  wyłącznie liczbę („5 kontaktów z meczów evaluation pominiętych”).

Te 23 kontakty nie wchodzą do reguły: 0 z 23 dawałoby U = 12,2% (0 z pierwotnych 28: 10,1%).

## Walidacja rozrzutu (D6)

`evaluate` cytuje wyłącznie zagregowaną statystykę z `results/xgv_timing_source/timing_verdict.json`
(`statistics.shot_loo_share_within_200ms` = 0,966 przy `n_shots` = 2 754: udział strzałów SoccerNet w ±200 ms od czasu
dostawcy po zdjęciu offsetu połowy) i jej sha256. Reszt per strzał (`shot_residuals.parquet`) nie czyta. Ogranicza to
rozrzut, nie położenie kontaktu. Walidacja nie zmienia wyniku reguły.

## Wyjścia

```bash
./run.sh --no-sync python scripts/xgt_safe_margin.py evaluate \
  --run-a outputs/xgt_safe_margin/answers/run_a/batch_{1,2,3,4}.csv \
  --run-b outputs/xgt_safe_margin/answers/run_b/batch_{1,2,3,4}.csv
```

zapisuje `results/xgt_safe_margin/`: `answers.csv` (odpowiedzi obu przebiegów per arkusz, najgorszy przypadek per
przebieg i offset, zgodność, naruszenia), `evidence.csv`, `result.json` (werdykt, k, U, statystyki per offset,
dowody, D6) i na końcu `complete.json` (hashe reguły, próbki, plików odpowiedzi, wejść, kodu i wyjść, stan gita).
Wynik („przechodzi” albo „stop”, k, U) trafia do Comments ticketu 02 i do wiersza w
`research/quality_v13_decisions.tsv`.

## Instrukcja dla oznaczającego

Tekst identyczny dla obu przebiegów i wszystkich partii (podstawić tylko `{RUN}` = `a` albo `b` i `{B}` = 1–4).
Katalog roboczy: `/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable`.

````text
Jesteś oznaczającym w sprawdzeniu czasu uderzenia przy strzałach w piłce nożnej. Oznaczasz partię {B}
przebiegu {RUN}. Pracujesz w katalogu
/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable
i nie zmieniasz w nim niczego poza własnym plikiem odpowiedzi i własnymi obrazkami lupy.

MATERIAŁ
Plik outputs/xgt_safe_margin/batches.csv: weź wiersze z batch == {B} (30 arkuszy). Dla każdego sheet_id:
- arkusz outputs/xgt_safe_margin/sheets/<sheet_id>.png: 24 klatki z transmisji telewizyjnej, od -1600 ms
  do 0 ms względem kotwicy strzału. 0 ms to chwila, w której zapis meczu oznacza strzał; kotwica jest
  przybliżona i może się mylić o kilkaset ms w obie strony. Podpis nad każdą klatką podaje jej offset.
  Trzy klatki pytań (-800, -600, -400 ms) mają podświetlony podpis "<- question". "CUT" (czerwona ramka)
  znaczy, że między tą klatką a kotwicą jest cięcie kamery (inne ujęcie albo powtórka), więc ta klatka
  może nie pokazywać tej samej akcji w sposób ciągły aż do 0 ms. "MISSING" znaczy brak klatki.
- klatki natywne 1280x720: outputs/xgt_safe_margin/frames/<sheet_id>/m1600.jpg ... m0040.jpg, p0000.jpg
  (m0600.jpg = -600 ms, p0000.jpg = 0 ms). Są tylko offsety z arkusza.
- lupa (ten sam wycinek klatek natywnych przy kilku offsetach, powiększony):
  ./run.sh --no-sync python scripts/xgt_safe_margin.py zoom <sheet_id> --center X Y \
      --offsets -800 -600 -400 --out outputs/xgt_safe_margin/zoom/run_{RUN}/<sheet_id>_<n>.png
  X, Y to środek wycinka w pikselach klatki 1280x720; domyślny wycinek 320x180 to powiększenie 2x,
  --width 480 daje szerszy wycinek. --offsets przyjmuje dowolne offsety z arkusza.
Klatek po 0 ms nie ma; nie szukaj ich. Nie otwieraj klipów wideo, katalogów cache/, results/ ani research/,
innych plików w outputs/ poza wymienionymi powyżej, ani odpowiedzi innego przebiegu
(outputs/xgt_safe_margin/answers/ poza Twoim plikiem). Nie ustalaj, czym strzał się skończył; to nie jest
częścią zadania.

PYTANIE
Dla każdego z offsetów -800, -600 i -400 ms: czy w chwili tej klatki piłka jest już uderzona STRZAŁEM?

CO JEST STRZAŁEM
Strzał to próba zdobycia bramki: kopnięcie albo uderzenie głową w stronę bramki przeciwnika przez zawodnika
drużyny atakującej. Chodzi o ten strzał, do którego należy kotwica 0 ms. Gdy w klatkach do 0 ms widać kilka
uderzeń piłki, strzałem jest ostatnie uderzenie w stronę bramki wykonane do 0 ms (np. przy strzale
zablokowanym i dobitce: dobitka). Gdy do 0 ms nie widać żadnego uderzenia w stronę bramki, strzał pada
dopiero po kotwicy: wtedy przy wszystkich trzech offsetach odpowiedź to "no".
NIE są strzałem: podania, dośrodkowania, rzuty rożne i wolne zagrane do partnera, wrzuty z autu,
prowadzenie piłki i przyjęcia, zgrania głową do partnera, odbicia i rykoszety od zawodników, interwencje
obrońców i bramkarza. Piłka lecąca po takim zagraniu NIE jest uderzona strzałem. Przykład: w -600 ms piłka
leci po dośrodkowaniu, a strzał głową pada około -150 ms: przy -600 odpowiedź "no".

ODPOWIEDZI
- yes: w klatce o tym offsecie piłka już opuściła nogę albo głowę strzelającego po strzale albo jest
  właśnie w kontakcie przy strzale (kontakt strzałowy w tej klatce lub wcześniej).
- no: w tej klatce strzał jeszcze nie nastąpił (strzelający podbiega, przyjmuje, czeka na piłkę, piłka
  leci do niego po podaniu itp.).
- unknown: nie da się tego rozstrzygnąć (piłka albo strzelający niewidoczni, cięcie, powtórka, nie wiadomo,
  czy wcześniejsze uderzenie było strzałem czy podaniem, nie wiadomo, które uderzenie jest strzałem).
  Nie zgaduj: przy wątpliwości wybierz unknown.
Spójność w czasie: uderzona piłka nie staje się nieuderzona, więc yes przy -800 wymaga yes przy -600 i -400,
a no przy -400 wymaga no przy -600 i -800.

JAK PRACOWAĆ
Dla każdego arkusza: obejrzyj cały arkusz; ustal, kto i między którymi klatkami uderza piłkę strzałem;
potem otwórz co najmniej klatki natywne -800, -600 i -400 ms oraz klatki wokół uderzenia; użyj lupy, gdy
piłka jest słabo widoczna. Każdy arkusz oceniaj niezależnie od pozostałych.

WYNIK
Plik CSV outputs/xgt_safe_margin/answers/run_{RUN}/batch_{B}.csv z nagłówkiem
sheet_id,struck_at_m800,struck_at_m600,struck_at_m400,note
i jednym wierszem na każdy arkusz partii (30 wierszy). Wartości: yes, no albo unknown. note to krótki opis
(np. "dośrodkowanie ok. -700; strzał głową między -200 a -160"), bez przecinków albo w cudzysłowie.
Po zapisaniu partii nie zmieniaj pliku.
````
