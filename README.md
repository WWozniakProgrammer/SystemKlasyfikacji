# SystemKlasyfikacji
System klasyfikacji przeżycia pasażerów Titanica z wykorzystaniem miękkiego k-NN z mechanizmem głosowania i automatycznym doborem liczby sąsiadów
<br><br>Cele projektu:
<br>
Czyszczenie i przygotowanie danych (uzupełnianie braków, kodowanie kategorii, normalizacja).
Zaimplementowanie klasyfikatora miękkiego, który:
wykorzystuje zbiór miękki jako mechanizm głosowania w k-NN,
analizuje "miękkie" podobieństwo na podstawie wybranych cech (np. wiek i klasa z tolerancją niepewności).
Automatyczny dobór k:
testowanie różnych wartości k z walidacją krzyżową,
wybór najlepszego k na podstawie dokładności i F1-score.
Porównanie wyników z klasycznymi metodami:
klasyczny k-NN,
SVM,
Random Forest.
Ocena jakości modelu – dokładność, precision, recall, F1-score, wykresy ROC/AUC.
Wizualizacja wyników i podsumowanie.
