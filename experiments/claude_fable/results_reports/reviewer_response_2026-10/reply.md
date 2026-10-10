Temat: Odpowiedź na uwagi z 2 października: złożoność, moment kontaktu, transfer

Dzień dobry,

dziękuję za uwagi. Odpowiadam w kolejności punktów. Szczegóły, wykresy i źródło każdej liczby są w załączonym raporcie REPORT.md. Wszystkie liczby to wyniki rozwojowe: każdy mecz widzieliśmy we wcześniejszych rundach.

1. Ensemble a G-combo. Nie wykazaliśmy przewagi ensemble'u. Redukcja log-loss wynosi +0.0010 [−0.0006; +0.0026], redukcja Briera +0.00022 [−0.00032; +0.00081]. Przedziały pochodzą z bootstrapu po 100 meczach testu. Test równoważności ogranicza możliwy zysk do 0.0024 log-loss i 0.0008 Briera, czyli do mniej niż 3.6% i 4.3% przewagi G-combo nad stałą prognozą. Oba modele mają nachylenie kalibracji ok. 1.2, a u ensemble'u jest ono wyższe o 0.037 [0.007; 0.067]. Zostajemy przy samym G-combo. Wykresy kalibracji i krzywa równoważności są w raporcie.

2. Moment kontaktu i przypisanie goli. Nie mieliśmy nikogo, kto mógłby oglądać wideo, więc ręcznego sprawdzenia nie zrobiliśmy. Zamiast niego przeprowadziliśmy automatyczny test z protokołem zatwierdzonym przed pierwszym wynikiem. Wśród 9 307 strzałów bez gola sprawdziliśmy, od którego offsetu względem kotwicy strzału da się odczytać jego celność. Celność rozstrzyga się przy kontakcie, więc skok czytelności w samej kotwicy znaczyłby, że klatka pokazuje grę po kontakcie. Protokół daje werdykt niekonkluzywny. Zamrożony DINOv2-L, na którym opiera się werdykt, czyta kierunek piłki za słabo, z AUC 0.611 nawet 800 ms po kotwicy. Dlatego kontrola pozytywna i test znanej odpowiedzi nie przeszły. Przecieku po kontakcie nie wykazaliśmy ani nie wykluczyliśmy. Wyniki opisowe nie pokazują przecieku, ale werdyktu nie zastępują. Odczyty z G-combo, które kierunek czytają, nie mają skoku w kotwicy. Udział klatek po kontakcie dla głowy celności wynosi 0.039 [−0.040; 0.112], a model uczony bez głowy celności daje te same krzywe. Przypisanie goli przeliczyliśmy pod kilkoma regułami i granicami dla łańcuchów dobitek. Żaden wniosek się nie zmienia, ale nachylenie kalibracji spada z 1.19 do 1.00, gdy gol należy do pierwszego strzału łańcucha. Punkt 2 uważam więc za otwarty i nie chcę udawać, że jest inaczej.

3. Transfer. Danych z Ekstraklasy nie będzie. Zamiast tego wyłączaliśmy po kolei każdą z pięciu lig krajowych SoccerNet i porównaliśmy model bez tej ligi z modelem uczonym na tej samej liczbie meczów, wśród nich połowie meczów tej ligi. Luka transferu wynosi +0.0041 [+0.0020; +0.0061] log-loss, czyli 1.2%, z tym samym znakiem w obu ziarnach treningu. Rekalibracja na 5–20 meczach nowej ligi jej nie odzyskuje. Pięć lig dzieli realizację telewizyjną i sezony, więc dla Ekstraklasy luka będzie prawdopodobnie większa. Ten protokół również zatwierdziliśmy przed treningami.

Pozdrawiam
Jakub Eichner
