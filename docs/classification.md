# Założenia 
Ten projekt ma za zadanie stworzenie i przeprowadzenie eksperymentu, który pokaże, która konfiguracja pipeline'u daje najlepsze rezultaty w zadaniu przypisania zawodników do zespołów w sposób nienadzorowany. 

Ten projekt używa python UV do uruchamiania kodu i packet managementu.

## Struktura
Nasz pipeline składa się z następujących etapów: 
1. Wyodrębnienie obrazów zawodników z materiału wideo na podstawie plików z konfiguracją - wszytkie informacje na temat zbioru danych są opisane w repozytorium https://github.com/SoccerNet/sn-tracking 
2. Wycięcie i augmentacja obrazów zawodników
3. Przekształcenie obrazów za pomocą embeding processor na wartoścli liczbowe 
4. Zmniejszenie liczby wymiarów zbioru przy pomocy UMAP
5. Klasteryzacja zbioru / podział na drużyny 

Przykładowa konfiguracja takiego pipline'u została opisana w pliku football_ai.py w sekcji """## split players into teams

## Zadania 

### Przygotowanie zbioru danych 
Najpierw musimy pobrać i przygotować zbiór danych, w tym celu należy stworzyć skrypt, który pobierze te dane  ze zbioru soccernet tracking i zapisze je lokalnie 

### Wyodrębnienie obrazów zawodników
Dla gotowych danych trzeba stowrzyć skrypt, który dla każdego nagrania wideo wyodrębni obrazy zawodników i zapisze je w odpowiedniej strukturze folderów: 
> {match_id}/{team_no}/{frame_no_player_no}
Ważne jest dla nas, żebyśmy mieli wszystkich zawodników z danej drużyny w każdym meczu w jednym folderze 

### Klasyfikacja zawodników 
Dla każdego z meczy będziemy wykonywać klasyfikację zawodników i zapisywać jej rezultat do badań 
Najważniejszym celem tego eksperymentu jest zbadanie wpływu różnych komponentów pipeline'u na wyniki klasyfikacji. 

Dla każdego etapu pipeline'u musimy przygotować różne warianty metod, których jakość porównamy między sobą: 
1. Przestrzenie kolorów obrazów: 
    - RGB
    - H (jedynie wartość hue z przestrrzeni HSV)
2. Augmentacja obrazów:
    - Brak augmentacji
    - Wycinanie środkowej części obrazu w różnych proporcjach
    - Kluczowanie tła zawodnika za pomocą metod openCV lub segmentatora SAM2 (porównanie tych metod)
3. Przekształcenie obrazów:
    - CLIP 
    - SIGLIP
    - ResNet16
4. Zmniejszenie liczby wymiarów:
    - UMAP (wybieranie liczby komponentów)
    - brak
5. Klasteryzacja:
    - DBSCAN
    - KMeans
    - CMeans
    

# Final thoughts 
Naszym zadaniem jest stowrzenie kodu python, który umożliwi przeprowadzenie tego eksperymentu. Kod musi być modułowy i pozwalać na wykonywanie całego pipeline przy pomocy jednego skryptu, który będzie sterowany plikiem konfiguracyjnym config.yaml. 

Docelowym przebiegiem eksperymentu jest wykonanie pipeline'u kilka razy dla różnych konfiguracji na każdym etapie, aby sprawdzić jak zmienia się wynik na podstawie różnych parametrów.

Przygotuj kod zgodnie z najnowszymi standardami python i dobrymi praktykami programowania PIP, jeżeli uznasz to za benefitial, to w odpowiednich do tego miejscach zastosuj również OOP. 
Postaraj się, aby kod był jak najlepiej zoptymalizowany i używał równoległych obliczeń dla przyspieszenia wykonania.

# Doprecyzowane założenia operacyjne
- Korzystamy ze wszystkich meczów z SoccerNet Tracking; próbkujemy wideo do 5 FPS i używamy gotowych bounding boxów z repozytorium.
- Przyjmujemy, że bboxy są poprawnie przypisane (brak dodatkowego filtrowania/walidacji).
- Używamy bazowych wersji modeli (CLIP, SigLIP, ResNet) bez fine-tuningu; dostęp do GPU zakładamy przy wszystkich etapach wymagających akceleracji.
- Oceniamy warianty pipeline'u po end-to-end accuracy (wynik klasyfikacji zawodnik→drużyna).
- Zakresy parametrów (UMAP, DBSCAN/KMeans/CMeans, augmentacje itp.) dobieramy na podstawie przeglądu danych/eksperymentów pilotażowych.

# Wyniki 
Wynikiem każdego eksperymentu powinien być plik .csv, który zawiera informacje o wynikach klasyfikacji dla każdego {frame_no}_{player_no} z każdego meczu. W każdym wierszu powinny być również zawarte informacjei o tym, które parametry wykorzystano do wykonania tego eksperymentu, tak, żebyśmy później mogli łatwo przeanalizować wyniki. 

# Wykresy 
Dla gotowego pliku .csv chcemy wykonać wykresy, które pozwolą nam porównać między sobą różne podejścia do każdego etapu pipeline'u.

---

# Podsumowanie wyników eksperymentu (metryki, wnioski, wykresy)

Poniższe podsumowanie bazuje na artefaktach w `results/team_classification/` (w szczególności `results/team_classification/raw/team_classification_report.md` oraz plikach `results/team_classification/numeric/team_classification_metrics*.csv`).

## Zakres i metryki
- Dane: SoccerNet Tracking (wide-view), w raporcie: 7 sekwencji (limit runtime); porównania dla `h` oraz `rgb`.
- Metryki: accuracy oraz metryki pochodne z macierzy pomyłek (TP/FP/TN/FN). W raporcie agregacja jest ważona sumą TP/FP/TN/FN po sekwencjach.

## Najważniejsze wyniki (high-level)
- Baseline: **center crop 0.2** osiąga **accuracy = 0.698** (precision **0.644**, recall **0.833**) dla obu przestrzeni barw.
- Maska **OpenCV (GrabCut)**: w obecnej implementacji wynik **identyczny jak baseline** (brak zysku jakościowego vs center).
- Maska **SAM2 (CPU)**: spadek do **accuracy = 0.583** (ok. **−11 pp** vs baseline) przy wysokim koszcie czasowym (~28 s/sekwencję na CPU).
- **UMAP=3**: niewielki zysk jakości dla `rgb` do **accuracy = 0.722** (+2.4 pp), dla `h` do **0.704** (+0.6 pp).
- Wpływ **przestrzeni barw** jest mniejszy (≤ ~2 pp) niż wpływ kadrowania/maski i redukcji wymiarów.

## Wnioski i rekomendacje
- Trzymać `center` 0.2 jako stabilny baseline do dalszych iteracji.
- `opencv_mask` jest obecnie neutralny — warto sprawdzić tuning GrabCut (większy margines, więcej iteracji, proste czyszczenie maski).
- SAM2 w obecnej formie (CPU) jest nieopłacalny (wolniej i gorzej); sensowny powrót wymaga GPU i stabilnego środowiska zależności.
- UMAP wygląda obiecująco, ale zysk jest mały — warto wrócić do szerszej siatki wymiarów (np. 8–32) i ewentualnego autotuningu per sekwencję.
- Dla weryfikacji stabilności obserwacji docelowo przeliczyć eksperymenty na pełnym zbiorze (bez limitu sekwencji).

## Najważniejsze wykresy
Poniższe wykresy są już wygenerowane i najlepiej oddają obserwowane trendy:

- Baseline i wpływ crop ratio: `../results/team_classification/plots/plots_full_paper_combined/accuracy_vs_crop_ratio.png`
  ![Accuracy vs crop ratio](../results/team_classification/plots/plots_full_paper_combined/accuracy_vs_crop_ratio.png)
- Wpływ UMAP na accuracy: `../results/team_classification/plots/plots_full_paper_combined/accuracy_vs_umap.png`
  ![Accuracy vs UMAP components](../results/team_classification/plots/plots_full_paper_combined/accuracy_vs_umap.png)
- Porównanie center vs SAM2: `../results/team_classification/plots/plots_full_paper_combined/center_vs_sam2.png`
  ![Center vs SAM2](../results/team_classification/plots/plots_full_paper_combined/center_vs_sam2.png)
- Breakdown błędów (confusion): `../results/team_classification/plots/plots_full_paper_combined/confusion_breakdown.png`
  ![Confusion breakdown](../results/team_classification/plots/plots_full_paper_combined/confusion_breakdown.png)
- Zbiorcze porównanie metod/parametrów: `../results/team_classification/plots/plots_full_paper_combined/paper_heatmap_method_ratio.png`
  ![Heatmap: method vs ratio](../results/team_classification/plots/plots_full_paper_combined/paper_heatmap_method_ratio.png)

## Artefakty (źródła)
- Raport: `../results/team_classification/raw/team_classification_report.md`
- Metryki (per sekwencja/konfiguracja): `../results/team_classification/numeric/team_classification_metrics.csv`
- Predykcje (per próbka): `../results/team_classification/numeric/team_classification_predictions.csv`
