# Slajd 3

## Tytuł
Homografia jako rdzeń systemu

## Tekst na slajdzie
Same bboxy w obrazie nie wystarczają do analizy taktycznej.

Homografia pozwala przejść:
- od punktów boiska w obrazie,
- do macierzy odwzorowania,
- do pozycji zawodników i piłki w układzie boiska.

To właśnie ten etap łączy wideo z analityką.

## Układ
- Trzy grafiki obok siebie:
  - `homography_keypoints.png`
  - `homography_vis_pitch_schematic.png`
  - `pitch_projection.png`
- Tekst maksymalnie po lewej górze lub jako krótki podpis pod grafikami.

## Co powiedzieć ustnie
Kluczowe jest to, że model nie kończy pracy na detekcji zawodników. Wykrywamy charakterystyczne punkty boiska, wyznaczamy homografię, a następnie rzutujemy pozycje obiektów z obrazu do wspólnego układu boiska. To jest moment, w którym klasyczne CV staje się narzędziem do analizy taktycznej.
