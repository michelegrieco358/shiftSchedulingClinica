# Analisi di `loader/leaves.py`

## Controlli preliminari e normalizzazione degli input
- Determina l'elenco dei turni operativi ammessi (`duration_min > 0`) dal catalogo turni; se il catalogo non contiene turni positivi, interrompe il caricamento con `LoaderError`.
- Prepara gli schemi di output (`shift_columns`, `day_columns`) così da restituire DataFrame consistenti anche in assenza del file `leaves.csv`.
- Se `leaves.csv` non esiste, restituisce subito due DataFrame vuoti con le colonne attese.
- Carica il CSV delle assenze tramite `load_absences`, applicando la stessa pulizia/validazione usata in `loader/absences.py` e, subito dopo, incrocia `employee_id` con `employees_df` per recuperare `absence_full_day_hours_effective_h`. Se il valore manca viene usato, come fallback opzionale, il parametro `absence_hours_h`; in mancanza di entrambi viene sollevato un errore.

## Calcolo dei limiti temporali
- Usa `_compute_horizon_window(calendar_df)` per ricavare le date di inizio/fine orizzonte in formato timestamp; da qui deduce il primo giorno del mese (`month_start_date`) per trattenere lo storico minimo richiesto dai report.
- Verifica che tutti gli `employee_id` presenti in `leaves.csv` esistano in `employees_df`; eventuali ID ignoti causano un errore.
- Converte gli intervalli di assenza (`date_from`, `date_to`) in timestamp normalizzati (`start_date_dt`, `end_date_dt`) e prepara l'espansione giorno-per-giorno tramite `explode_absences_by_day`, limitando le date all'intervallo `[min(month_start_date, horizon_start_date), horizon_end_date)` e valorizzando ore/flag di assenza giornalieri.

## Espansione per turno
- Estrae da `shifts_df` gli orari di inizio/fine e il flag `crosses_midnight` dei turni ammessi, costruendo una lista iterabile.
- Per ogni intervallo di assenza:
  - Scorre i giorni dell'assenza (incluso quello precedente per catturare turni che partono la sera prima) e genera timestamp di inizio/fine turno applicando l'eventuale attraversamento della mezzanotte.
  - Se l'intervallo del turno interseca `[absence_start_day, absence_end_day + 1 giorno)`, registra una riga `{employee_id, data, turno, tipo}`.
- Una volta raccolti i record:
  - Converte `data` in timestamp (`data_dt`), elimina duplicati e aggancia il calendario per recuperare settimana, flag weekend/festività e `is_in_horizon`.
  - Join con il catalogo turni per portare durata, orari e flag `crosses_midnight`, calcolando `shift_start_dt`/`shift_end_dt` come timestamp localizzati e spostando l'end di un giorno per i turni notturni.
  - Applica tre filtri combinati: `is_in_horizon`, sovrapposizione dell'intervallo turno con la finestra temporale dell'orizzonte e appartenenza allo storico mensile (`month_start_date ≤ data < horizon_start_date`). Solo i record che soddisfano almeno una condizione vengono mantenuti.

## Aggregazione giornaliera
- `abs_by_day` (ritornato da `explode_absences_by_day`) viene subito riallineato alle ore giornaliere specifiche per dipendente, quindi arricchito con calendario e filtrato sugli stessi criteri di orizzonte/storico.
- Raggruppa per `(employee_id, data)` aggregando:
  - `tipo_set`: unione dei codici di assenza del giorno (join con `|`).
  - Flag giornalieri (`is_absent`, `is_leave_day`) e ore di assenza (`absence_hours_h`).
  - Metadati temporali (settimana, weekend, festività) presi dal primo record disponibile.
- Restituisce il DataFrame giornaliero ordinato con le colonne richieste per l'integrazione nelle fasi successive del loader.

## Output finale
- Ritorna una tupla `(shift_out, day_out)`:
  - `shift_out`: assenze espanse per turno con timestamp precisi, durata e flag di calendario utili a bloccare gli slot durante l'ottimizzazione.
  - `day_out`: aggregazione giornaliera che riassume ore e tipologie di assenza per ogni dipendente/giorno, usata per i vincoli sui conteggi di ferie/permessi.
- Entrambi i DataFrame sono coerenti con gli schemi dichiarati anche quando non ci sono record (liste di colonne vuote), assicurando che gli step successivi non falliscano per colonne mancanti.
