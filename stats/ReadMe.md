<h3>ReadMe tymczasowe żeby by się tu nie pogubić</h3>

Statystyki to work in progress - trzeba je mocno zrefaktoryzować.

TODO:

- Zrefaktoryzować kod (Wszystko jako funkcje/klasy - nie skrypty)
- Zredukować liczbę plików

Co tu jest:

- ae_training.py - trenowanie autoencodera potrzebnego do dwóch następnych punktów
- anomaly_detection.py - znajdywanie anomalii na podstawie błędu po rekonstrukcji autoencodera (Można tym znajdywać outliery w bazie i je wywalać)
- clustering.py, clustering_metrics.py - clustrowanie obrazków po encodingach z autoencodera
- image_statistics.py, statistics.py, some_analysis.py - różnego rodzaju statystyki - trzeba przejrzeć, zrefaktoryzować i zrobić z tego jeden plik
- margonem_measures - folder z rzeczami do liczenia FID oraz IS. Metoda liczenia z netu, może da się zrobić prościej, ale na razie zostawiam do wglądu