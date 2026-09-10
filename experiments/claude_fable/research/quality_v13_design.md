# Wybrany projekt v1.3

Wybrano osobny runner wariantu B. Historyczne runnery mają różne konwencje walidacji i raportują test w trakcie pracy, więc nowa runda nie będzie ich wywoływać. Zachowujemy istniejący trener, modele i preprocessing. Z wariantu A bierzemy temperaturę używaną tylko do wyboru epoki, następnie odrzucaną przed niezależnym Platt scaling, oraz weryfikację paczki w nowym procesie.

Kryteria porównania projektów: niezależność czterech ról, identyczna kohorta, odczyt paczki modelu, mały interfejs, koszt GPU. Oba projekty spełniają zapis paczki i rozdział ról. B lepiej izoluje historyczne skrypty. Odrzucamy z A użycie evaluate do screeningu: screening wyłącznie na select. Recenzja niezależna projektu jest w toku, a jej uwagi będą rozpatrzone przed przyjęciem wyników.

Wywołania:

```bash
./run.sh python scripts/run_v13.py --prepare
./run.sh python scripts/run_v13.py --variants g_base --seeds 0
./run.sh python scripts/run_v13.py --variants full roi control time_mean time_gru geometry --seeds 0 1 2
./run.sh python scripts/evaluate_v13.py
```

Moduły:

- `xgpoc/v13/protocol.py`: deterministyczny podział gospodarzy 60/10/10/20 na fit/select/calibrate/evaluate; historyczny test nie wchodzi do rundy. Niezmienny manifest przypięty do indeksu.
- `xgpoc/v13/inputs.py`: identyczne identyfikatory i kompaktowe cechy ROI/czas/geometria, bez usuwania trudnych rekordów.
- `xgpoc/v13/models.py`: głowa na patchach pełnej klatki, opcjonalne małe wejście dodatkowe; zapisane kwargs odtwarzają konstruktor.
- `scripts/run_v13.py`: seeding przed konstruktorem, trening, odrębna kalibracja i zapis pełnej paczki. Predykcje evaluate bez etykiet.
- `scripts/evaluate_v13.py`: po zamrożeniu listy kandydatów odczyt etykiet evaluate, porównania sparowane. Każde wznowienie sprawdza manifest i konfigurację.

Kohorta rozwojowa jest historycznie znana, więc jej nowy podział nie staje się nietkniętym testem. Nowe dane wymagają oddzielnego protokołu. Model G jest pełnym artefaktem inferencji obrazowej. Małe modele na cache są artefaktami eksperymentalnymi z jawną tożsamością enkodera i wejść.
