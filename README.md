# Rozpoznawanie kart do gry przy użyciu sieci neuronowych

## Opis projektu
Opracowanie i porównanie skuteczności trzech różnych architektur sieci neuronowych w zadaniu klasyfikacji obrazów kart do gry. Wykorzystano publicznie dostępny zbiór „Cards Image Dataset-Classification” z platformy [Kaggle](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data), zawierający obrazy 53 różnych kart w rozdzielczości 224×224 pikseli.

## Cel
- Zbadanie wpływu głębokości i złożoności architektury sieci na dokładność klasyfikacji.
- Porównanie modeli:  
  1. **Bazowego** – płytka sieć z jedną warstwą Conv2D.  
  2. **Pośredniego** – dwie warstwy Conv2D z regularyzacją Dropout.  
  3. **Zoptymalizowanego** – głęboka sieć z wieloma blokami Conv2D, BatchNormalization i SpatialDropout2D.
 
## Dane
- Źródło: [Cards Image Dataset-Classification](https://www.kaggle.com/datasets/gpiosenka/cards-image-datasetclassification/data) 
- Liczba obrazów:  
  - Treningowe: 7 094  
  - Walidacyjne:   530  
  - Testowe:       530  
- 53 klasy odpowiadające poszczególnym kartom; obrazy w formacie JPG, rozmiar 224×224×3

## Architektury modeli
### 1. Model bazowy
- **Wejście**: `Input(224,224,3)` + `Rescaling(1./255)`
- **Warstwa ukryta**:  
  - `Conv2D(128, (3×3), activation='relu')`  
  - `MaxPooling2D((2×2))`  
  - `Flatten()`
- **Wyjście**: `Dense(53, activation='softmax')`
- **Parametry**: ok. 2,47 mln (wszystkie trenowalne)  
- Najprostszy, jednowarstwowy model z dużym ryzykiem przeuczenia.

### 2. Model pośredni
- **Wejście**: `Input(224,224,3)` + `Rescaling(1./255)`
- **Bloki konwolucyjne**:  
  1. `Conv2D(32, (3×3), activation='relu')` + `MaxPooling2D((2×2))`  
  2. `Conv2D(64, (3×3), activation='relu')` + `MaxPooling2D((2×2))` + `Dropout(0.3)`
- **Klasyfikacja**: `Flatten()` → `Dense(64, activation='relu')` → `Dense(53, activation='softmax')`
- **Parametry**: ok. 1,32 mln  
- Lepsza generalizacja dzięki dwóm warstwom konwolucyjnym i Dropout.

### 3. Model zoptymalizowany
- **Wejście**: `Input(224,224,3)` + `Rescaling(1./255)`
- **Głębokie bloki konwolucyjne** z inicjalizatorem He Normal, BatchNormalization, SpatialDropout2D, funkcjami ELU/ReLU oraz stopniowym zwiększaniem liczby filtrów (32 → 256).
- **Pooling**: naprzemiennie `MaxPooling2D((4×4))` i `(2×2)`
- **Warstwy gęste**:  
  1. `Dense(256, kernel_initializer='he_normal')` + `BatchNormalization()` + `Activation('elu')` + `Dropout(0.2)`  
  2. `Dense(128, kernel_initializer='he_normal')` + `BatchNormalization()` + `Activation('elu')` + `Dropout(0.2)`  
  3. `Dense(53, activation='softmax')`
- **Parametry**: ok. 5,75 mln (trenowalne ok. 5,75 mln)  
- Najbardziej złożona architektura, minimalizująca przeuczenie i wydobywająca hierarchiczne cechy.

## Wyniki

| Model           | Accuracy | Średni F1-score | Macro F1-score | Weighted F1-score |
|-----------------|----------|-----------------|----------------|-------------------|
| Bazowy          | 0.55     | 0.56            | 0.56           | 0.56              |
| Pośredni        | 0.66     | 0.65            | 0.65           | 0.65              |
| Zoptymalizowany | 0.83     | 0.83            | 0.83           | 0.83              |

### Najlepsze i najgorsze klasy (F1-score, model zoptymalizowany)
- **Najwyższe**:  
  - Three of Hearts (1.00)  
  - Ace of Hearts, Joker, Five of Diamonds, Ten of Hearts (0.95)
- **Najniższe**:  
  - King of Spades, Ten of Clubs (0.70)  
  - King of Hearts (0.67), Four of Hearts, Six of Hearts (0.63)
 
 ## Wnioski
- **Model bazowy**: silne przeuczenie (trenowanie ≈95%, walidacja ≈60%).  
- **Model pośredni**: lepsza równowaga między trenowaniem a walidacją, mniejszy gap strat i wyższa walidacja (≈70%).  
- **Model zoptymalizowany**: stabilne uczenie, najmniejsze przeuczenie, zarówno accuracy jak i F1-score ≈0.83–0.90 w zależności od klasy. Model zoptymalizowany okazał się zdecydowanie najlepszy pod względem generalizacji i stabilności. Dzięki głębokiej architekturze i regularizacji osiągnięto znacząco wyższe metryki w porównaniu z modelem bazowym i pośrednim.
- Ograniczenia: niewielka liczba obrazów walidacyjnych/testowych (10 obrazów na klasę) może wpływać na niestabilność estymacji.
[Wykresy przebiegu trenowania poszczególnych modeli](https://github.com/IamPawel/Playing_Card_Recognition_Using_Neural_Networks/blob/main/Rozpoznawanie%20kart%20do%20gry%20przy%20u%C5%BCyciu%20sieci%20neuronowych.pdf)
