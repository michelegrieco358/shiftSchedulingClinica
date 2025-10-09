# Code Review Findings

## 1. Missing solver/model implementation
- `src/model.py` è presente ma vuoto (dimensione zero). Il progetto non fornisce alcuna logica di modellazione/ottimizzazione, rendendo impossibile costruire o eseguire il modello di scheduling richiesto.
  - Evidenza: `ls -l src/` mostra `model.py` con dimensione 0 byte.

## 2. Uso errato di `or` con DataFrame pandas
- Nella funzione `build_all` (`src/preprocessing.py`) vengono usati operatori booleani `or` per scegliere tra DataFrame alternativi (`df_abs = dfs.get("absences") or ...`).
- I DataFrame pandas non supportano la valutazione di verità; quando l'espressione incontra un DataFrame non vuoto solleva `ValueError: The truth value of a DataFrame is ambiguous`.
- Questo bug impedisce l'esecuzione della funzione appena sono disponibili i DataFrame attesi.

### Impatto
- Il preprocessing fallisce quando esistono i dataset `absences`, `leaves_days_df`, `leaves_df` o `availability_df`, bloccando il flusso di lavoro downstream.
- È sufficiente che uno dei DataFrame sia non vuoto perché si verifichi l'eccezione.

### Suggerimento
- Sostituire gli `or` con una logica esplicita, ad esempio usando `next` su una generator expression oppure iterando sui candidati e scegliendo il primo DataFrame non vuoto.

