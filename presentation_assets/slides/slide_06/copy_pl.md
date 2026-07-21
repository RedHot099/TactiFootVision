# Slajd 6

## Tytuł
Trade-off jakości i wydajności

## Tekst na slajdzie
Pipeline jest modułowy i pozwala wybierać punkt pracy.

- YOLO11m: wariant szybki
- RF-DETR Base: wariant jakościowy
- tracking wpływa na ciągłość trajektorii i stabilność ID

Różne komponenty zmieniają koszt obliczeniowy, jakość lokalizacji i użyteczność analityczną.

## Układ
- Główny wykres: `map_vs_fps.png`
- Drugi wykres: `quality_speed_tradeoff.png`
- Opcjonalnie mały trzeci wykres: `inference_fps_boxplot.png`

## Co powiedzieć ustnie
Tutaj pokazujemy, że końcowy system jest modułowy. Możemy przesuwać się w stronę większej szybkości albo większej jakości, w zależności od scenariusza użycia. To ważne, jeśli myślimy zarówno o analizie offline, jak i o potencjalnych zastosowaniach bliższych real-time.
