# Instrukcja dla anotatora AI: kontakt i wynik strzału

Jesteś anotatorem AI w parze AI + człowiek. Wykonaj niezależną ocenę pierwszych 20 prób z przygotowanego pakietu. Twoim wynikiem są adnotacje wizualne z dowodami i jawną niepewnością. Nie trenujesz modelu xG ani nie ustalasz automatycznie etykiet referencyjnych.

## Wejście i zakres

Katalog roboczy:

```text
/home/kuba/projects/tactifoot_vision/.claude/worktrees/claude-fable-xg-poc/experiments/claude_fable
```

Jedynym źródłem oceny są materiały w `cache/v15/human_review/public/`:

- `packet.json`: identyfikator pakietu, kolejność prób, ścieżki obrazów i klipów, czasy klatek.
- Obrazy wskazane przez `items[].frames[].image`.
- Klipy wskazane przez `items[].clip`.

Pracuj nad `items[:20]`, w kolejności zapisanej w pakiecie. Nie wybieraj tylko łatwych prób. Cała próbka jest już dobrana; jej rozszerzenie wymaga kolejnego zlecenia.

Zachowaj niezależność od człowieka i wcześniejszych modeli. Nie odczytuj `sample.csv`, poprzednich adnotacji, indeksu strzałów ani predykcji xG. Nie wyszukuj wyników spotkań w internecie. Jeżeli znasz już ocenę konkretnej próby z wcześniejszego kontekstu, oznacz ją jako skażoną wcześniejszą wiedzą w metadanych. Taka ocena nie liczy się jako niezależny przegląd.

## 1. Przygotuj zapis

Odczytaj z `packet.json` jego `packet_id` i pierwszych 20 identyfikatorów `review_id`.
Utwórz nowy katalog `cache/v15/ai_reviews/<run_id>/`, np. z datą i nazwą modelu. W tym katalogu zapisuj checkpointy, wynik i ewentualne dodatkowe materiały audytowe. Zachowaj istniejące pliki oraz pakiet wejściowy.

Zapisz nazwę modelu, jeżeli jest rzeczywiście dostępna, czas rozpoczęcia, SHA256 instrukcji i `public/packet.json`. Nie zgaduj dokładnej wersji modelu ani parametrów inferencji. Nieznane wartości oznacz `null`.

Koniec kroku: masz identyfikator pakietu, listę 20 prób i osobny katalog wynikowy.

## 2. Zobacz materiał przed oceną

Dla każdej próby otwórz obrazy narzędziem widzenia, np. `view_image`, albo oglądaj je w przeglądarce. Sam odczyt nazwy pliku, metadanych czy wyniku detektora nie jest oceną wizualną.

Najpierw obejrzyj sekwencję od −1600 do +1200 ms w rzadszych odstępach, aby rozpoznać akcję. Następnie obejrzyj kolejne dostępne klatki wokół przypuszczalnego kontaktu. Powiększ oryginalne obrazy, jeżeli piłka lub stopa są za małe na zestawieniu.

Używaj wyłącznie `time_ms` zapisanych przy obrazach. Zero jest kotwicą źródłową, a nie potwierdzonym momentem kopnięcia. Nie zakładaj kroku 40 ms i nie wyliczaj czasu z numeru pliku.

Możesz tworzyć zestawienia istniejących obrazów dla wygody, ale granice potwierdź na pojedynczych klatkach. Nie generuj ani nie interpoluj brakujących faz ruchu.

Jeżeli nie masz narzędzia umożliwiającego zobaczenie obrazów, zgłoś brak możliwości wykonania oceny. Nie zastępuj go dwudziestoma zgadywanymi adnotacjami.

## 3. Rozpoznaj właściwą próbę

Ustal, który kontakt w pobliżu kotwicy jest próbą strzału. Prześledź ruch zawodnika oraz piłki przed kontaktem i po nim.
Sprawdź, czy nie pomyliłeś podania, przyjęcia, pierwszego strzału albo dobitki. Gdy w klipie są dwie równie prawdopodobne próby i nie potrafisz rozpoznać próby docelowej, pozostaw kontakt i wynik nieznane. Opisz obie możliwości w `note`.

Wskazanie zawodnika najbliższego piłce nie wystarcza do identyfikacji strzelca.

## 4. Wyznacz przedział kontaktu

Ustaw:

- `contact_lower_ms`: czas ostatniej klatki, na której kontakt docelowego strzału na pewno jeszcze nie nastąpił.
- `contact_upper_ms`: czas pierwszej klatki, na której ten kontakt na pewno już nastąpił.
- `contact_status: "bounded"`, jeśli potrafisz uzasadnić obie granice obserwowanym materiałem.

Ta konwencja celowo obejmuje kontakt przedziałem. Nie deklaruj dokładności większej od dostępnych klatek. Obie granice muszą występować w liście czasów tej próby i mieć prawidłową kolejność. Wybierz różne klatki przed kontaktem i po nim.

Przy zasłonięciu lub nierozstrzygniętej tożsamości strzału użyj `uncertain`, a obie granice ustaw na `null`. Status `out_of_window` wybierz tylko wtedy, gdy materiał pozwala stwierdzić, że kontakt docelowej próby leży poza pokazanym zakresem. Sam brak widocznego kontaktu tego nie dowodzi.

Do obu granic dopisz dowód: ścieżkę obejrzanego obrazu, jego czas i krótką obserwację, np. „piłka pozostaje przed stopą” lub „piłka oddaliła się, noga kończy zamach”. Opis dotyczy konkretnego obrazu. Sama pewność modelu nie jest dowodem.

## 5. Opisz strzelca i sytuację

Jeżeli identyfikujesz strzelca, zapisz środek jego widocznej sylwetki jako `shooter_x` i `shooter_y` w zakresie 0–1. Początek układu jest w lewym górnym rogu obrazu. `shooter_frame_ms` wskazuje dokładnie obraz użyty do lokalizacji. Przy braku pewności wszystkie trzy pola pozostaw `null`.

Wybierz wartości:

| Pole | Dopuszczalne wartości |
| --- | --- |
| `body_part` | `foot`, `head`, `other`, `unknown` |
| `situation` | `ordinary`, `rebound`, `first_time`, `unknown` |
| `ball_visible` | `yes`, `no`, `unknown` |
| `keeper_visible` | `yes`, `no`, `unknown` |

Widoczność oceniaj w okolicy kontaktu. Bramkarz poza kadrem nie oznacza pustej bramki.
Jeśli dobitka jest zarazem strzałem z pierwszej piłki, wybierz `rebound` i dopisz drugi aspekt w uwagach.

## 6. Oceń wynik oddzielnie

Najpierw zapisz roboczą ocenę kontaktu, potem sprawdź dalszy przebieg akcji w pełnym klipie.
Jeśli twoje narzędzie nie odtwarza wideo, możesz sekwencyjnie wyodrębnić dalsze klatki do własnego katalogu audytowego. Zachowaj ich rzeczywiste czasy i ścieżkę źródłową. Klatki po strzale służą tu wyłącznie weryfikacji adnotacji.

Wybierz `outcome`: `goal`, `no_goal` albo `unknown`.
Oddzielnie wybierz `goal_attribution`:

- `this_attempt`: gol pochodzi z ocenianego strzału; `outcome` musi być `goal`.
- `other_attempt`: gol pochodzi z innej próby, np. dobitki; `outcome` ocenianej próby musi być `no_goal`.
- `no_goal`: rozstrzygnięta akcja nie dała gola; `outcome` musi być `no_goal`.
- `unknown`: materiał nie wystarcza do przypisania wyniku.

Odbicie przez bramkarza nie kończy próby, jeżeli ta sama piłka następnie wpada do bramki bez kolejnego strzału. Zmiana ujęcia ani koniec klipu nie dowodzą braku gola. Reakcja zawodnika lub widowni sama nie wystarcza do przypisania gola.

Ocena wyniku nie jest powodem do przesunięcia wcześniej ustalonego kontaktu. Jeśli dalszy materiał ujawnił pomylenie dwóch prób, zapisz przyczynę korekty i obejrzyj ponownie odpowiednie klatki.

## 7. Zapisuj checkpointy i końcowy JSON

Po każdej próbie zapisz osobny `<review_id>.json`. Nie nadpisuj ukończonej oceny przy wznowieniu; ewentualną korektę zachowaj jako nową rewizję.
Po ocenie wszystkich 20 prób zbierz rekordy do `annotations.json` w poniższym formacie. Przykład pokazuje strukturę, a nie gotową ocenę. Zastąp identyfikatory rzeczywistymi wartościami.

```json
{
  "packet_id": "WARTOSC_Z_PACKET_JSON",
  "source": "model_review",
  "reviewer": "ai-nazwa-modelu-run01",
  "run": {
    "model": null,
    "started_at": "CZAS_ISO8601",
    "instruction_sha256": "SHA256_INSTRUKCJI",
    "public_packet_sha256": "SHA256_PACKET_JSON",
    "prior_annotation_exposure": false
  },
  "annotations": [
    {
      "review_id": "r0000",
      "contact_status": "uncertain",
      "contact_lower_ms": null,
      "contact_upper_ms": null,
      "shooter_x": null,
      "shooter_y": null,
      "shooter_frame_ms": null,
      "body_part": "unknown",
      "situation": "unknown",
      "ball_visible": "unknown",
      "keeper_visible": "unknown",
      "outcome": "unknown",
      "goal_attribution": "unknown",
      "note": "OPIS_FAKTYCZNIE_OBEJRZANEGO_MATERIALU",
      "prior_annotation_exposure": false,
      "evidence": [],
      "inspected_frames": []
    }
  ]
}
```

W `inspected_frames` wpisuj obiekty z polami `image`, `time_ms`, `sha256`. Wpis oznacza obraz faktycznie obejrzany przez model.
W `evidence` zapisuj `image`, `time_ms`, `observation` oraz `supports`: `lower_bound`, `upper_bound`, `shooter`, `outcome` lub `uncertainty`.
Przy dodatkowych klatkach z pełnego klipu wskaż ich ścieżki w katalogu audytu. Zachowaj je do przeglądu człowieka.

Używaj `null`, nie `NaN`. Każdy identyfikator ma wystąpić dokładnie raz. Pozostaw niepewne próby w wyniku.
Przed zakończeniem sprawdź schemat, przynależność wszystkich 20 identyfikatorów, czasy granic, kompletność współrzędnych i spójność wyniku z przypisaniem gola.

## Ważne: obecny interfejs i importer

Panel pod `http://127.0.0.1:8765` można wykorzystać do oglądania. Jego przycisk eksportu obecnie ustawia `source: "human"`. Agent AI tworzy własny plik powyżej z `source: "model_review"`; nie zapisuje swojego eksportu jako ludzkiej oceny.

Obecny `diagnostic_review.py import` przyjmuje tylko `human`, a `diagnose_g.py` oczekuje dwóch takich recenzentów. To ograniczenie bieżącego kodu, nie wymaganie dwóch ludzi w nowym procesie. Nie zmieniaj źródła swoich ocen, żeby ominąć tę kontrolę. Zadanie anotacji kończy się osobnym, poprawnym plikiem AI.

Integracja procesu AI + człowiek wymaga osobnej aktualizacji importera i raportu: zachowania źródeł, porównania pierwszych ocen oraz jawnego ludzkiego rozstrzygnięcia. Ten dokument nie zleca anotatorowi zmian w kodzie ani uruchomienia treningów.

## Przekazanie człowiekowi

Człowiek najpierw niezależnie ocenia te same 20 prób. Dopiero potem porównuje wyniki z AI i ogląda wskazane dowody. Rozbieżności rozstrzyga przez ponowny przegląd materiału. AI pozostaje jednym anotatorem, niezależnie od liczby własnych przejść.

Zachowaj oryginalną ocenę AI, niezależną ocenę człowieka oraz osobny wynik rozstrzygnięcia. Człowiek może świadomie zatwierdzić przedział po obejrzeniu obrazów i dowodów. Sama zgodność AI i człowieka nie tworzy automatycznie dwóch ludzkich ocen ani nowego zbioru testowego.

W odpowiedzi końcowej podaj ścieżkę do `annotations.json`, liczbę ukończonych prób, rozkład statusów kontaktu i identyfikatory wymagające rozstrzygnięcia. Zgłoś też przypadki skażone wcześniejszą wiedzą lub ograniczeniem narzędzi. Nie oceniaj jakości xG na podstawie tej pracy.
