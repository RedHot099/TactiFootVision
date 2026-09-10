# Runda v1.3: wykonanie zaleceń audytu

Warunek ukończenia: niezależny podział fit/select/calibrate/evaluate, odtwarzalny artefakt modelu, kontrola danych i czasu, zmierzone ablacje ROI/czas/geometria, audyt końcowy oraz poprawione potwierdzone usterki. Wzrost jakości wymaga redukcji log-loss na identycznej kohorcie z przedziałem sparowanej różnicy. Brak istotnego zysku oznacza INCONCLUSIVE, nie sukces jakościowy.

Lista wykonania:

- [x] Przeczytać Principles poteto-mode i zasady figure-it-out.
- [x] Ground: audyt 2026-09-07 i ścieżka index → datasets/cache → trainer → temperature → CSV → stack.
- [x] Frame: ustalić protokół i dostępność danych. Nie używać historycznego testu do wyboru wariantu.
- [x] Sketch: dwa niezależne projekty walidacji i modelu, ocena i synteza.
- [x] Implement: najpierw odtwarzalność i granice walidacji, potem dane.
- [x] Skontrolować kotwice i pokrycie strzałów, zachować jawne niepewności.
- [ ] Sprawdzić i pozyskać dostępne nowe dane bez obchodzenia dostępu.
- [x] Uruchomić bazę oraz ablację ROI.
- [x] Uruchomić ablację czasu i geometrii.
- [x] Porównać sparowane wyniki; promować tylko uzasadnione warianty.
- [x] Scrap: wycofać nieskuteczne zmiany z konfiguracji rekomendowanej.
- [x] Przeprowadzić niezależny audyt, naprawić zaakceptowane problemy i ponowić odpowiednie sprawdzenia.
- [x] Verify: sprawdzić odczyt modelu w nowym procesie, testy, raport i dziennik.

Stan 2026-09-09: ukończono 31 treningów i ocenę rozwojową. INCONCLUSIVE dla przyrostu jakości; rekomendacja pozostaje g_base. Nowe kompletne dane oraz walidacja człowieka nadal potrzebne. Szczegóły: quality_v13_results_2026-09-09.md.

Zasoby: RTX 4070 Ti SUPER 16 GB, około 264 GB wolnego. Zachować minimum 5 GB wolnego i istniejący cache 108 GB. Pełne wieloseedowe modele G wymagają godzin GPU. Najpierw krótki pomiar przepustowości i poprawności, potem pełne porównanie. Nowy nietknięty test wymaga rzeczywiście nowych meczów; ponowny podział istniejącej kohorty jest tylko oceną rozwojową.

Projekt jest eksperymentem, zmiany pozostają w experiments/claude_fable. Zewnętrzne dane i wcześniejsze wyniki pozostają nietknięte. Dziennik decyzji: quality_v13_decisions.tsv. Dostępne modele recenzujące są tylko z OpenAI; panel nie ma różnorodności dostawców.
