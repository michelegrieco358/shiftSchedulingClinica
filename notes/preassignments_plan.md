# Preassignments Integration Plan

Questo documento descrive la procedura proposta per integrare nel sistema un nuovo CSV di `preassignments` che consenta di rigenerare l'orario su un sottointervallo minimizzando i cambiamenti rispetto a un piano già prodotto.

## 1. Raccolta degli input

1. **History**
   - L'utente esporta le assegnazioni consolidate per l'intervallo già lavorato (es. 1–14 novembre) e le inserisce nel CSV `history` esistente.
   - Questo dataset viene utilizzato dal solver solo per calcolare vincoli cumulativi (riposi, contatori ore, ecc.) ma non viene modificato.
2. **Preassignments**
   - Dal piano originario (es. 1–30 novembre) si estraggono le assegnazioni per il nuovo orizzonte (es. 15–30 novembre).
   - Si salva un CSV `preassignments.csv` con lo stesso schema delle assegnazioni standard; non sono richieste colonne aggiuntive.
3. **Vincoli aggiornati**
   - L'utente modifica i CSV già esistenti per assenze, locks, domanda, ecc.; durante il caricamento verranno considerati solo i record che cadono nel nuovo orizzonte temporale, ignorando silenziosamente quelli precedenti.

## 2. Caricamento e validazione

1. Riutilizzare le pipeline già presenti per leggere i CSV di assenze, locks e domanda senza cambiamenti strutturali.
2. Creare una nuova funzione di caricamento per `preassignments.csv` che:
   - Valid i lo schema (tipi, valori ammessi) in modo identico agli assignments.
   - Verifichi che le date cadano all'interno del nuovo orizzonte indicato dall'utente.
   - Imposti un flag `is_preassignment = True` nel DataFrame risultante.
3. Integrare il DataFrame di preassegnazioni nel modello dati principale, mantenendolo separato dalle decisioni del solver finché non si costruisce la funzione obiettivo.

## 3. Estensione del solver

1. Continuare a generare le variabili binarie `x[dipendente, giorno, turno]` come avviene attualmente.
2. Per ogni combinazione presente in `preassignments`, recuperare il valore precedente `x_prev` (0/1) e definire una penalità di cambiamento:
   - Introdurre una variabile ausiliaria `delta_change` con i vincoli:
     - `delta_change >= x_new - x_prev`
     - `delta_change >= x_prev - x_new`
   - Aggiungere alla funzione obiettivo il termine `penalty_preassignment * delta_change`, dove `penalty_preassignment` è un valore configurabile in `config.yaml`.
3. Garantire che eventuali preassegnazioni incompatibili con i nuovi vincoli rigidi (assenze, locks hard, domanda minima) possano essere violate, ma producano una penalità in funzione obiettivo.
4. Il nuovo termine della funzione obiettivo viene aggiunto solo quando è presente il CSV `preassignments`; in assenza di tale file il solver opera esattamente come nella versione attuale.

## 4. Workflow di esecuzione

1. L'utente indica il nuovo orizzonte temporale (es. 15–30 novembre).
2. Il sistema carica:
   - `history` → contribuisce a vincoli cumulativi.
   - `preassignments` → fornisce i valori `x_prev` da confrontare.
   - Vincoli aggiornati (assenze, locks, domanda).
3. Il solver viene eseguito con l'obiettivo esteso che combina i termini originali con la penalità sui cambiamenti.
4. Dopo l'ottimizzazione, si produce un report che evidenzia:
   - Numero di cambi per dipendente.
   - Turni modificati rispetto alle preassegnazioni.
   - Eventuali preassegnazioni violate e relativo costo.

## 5. Output e storicizzazione

1. Salvare il nuovo piano (es. `assignments_2023-11-15_2023-11-30.csv`) insieme a un report di confronto con le preassegnazioni.
2. Aggiornare lo storico includendo l'intervallo appena pianificato per future riesecuzioni.

## 6. Test suggeriti

1. **Unit test** per la funzione di caricamento/validazione di `preassignments`.
2. **Integrazione** su un caso di esempio (novembre) verificando che la penalità di cambiamento riduca effettivamente il numero di differenze rispetto al piano originario.
3. **Regressione** per assicurarsi che, in assenza di `preassignments`, il solver si comporti come prima.
