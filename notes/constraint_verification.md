# Verifica implementazione vincoli

Questo documento sintetizza come i vincoli richiesti da *Document 3* siano mappati nel codice sorgente e nei test automatici.

## Vincoli orari
- `_add_hour_constraints` calcola i minuti assegnati per ciascun dipendente in finestre settimanali e mensili, integra gli storici (ore lavorate e assenze) e applica i limiti hard previsti da configurazione (`max_week_minutes`, `max_month_minutes`). Quando dispone di copertura completa del mese, crea variabili di bilancio/deviazione per il monte ore teorico, da usare nella funzione obiettivo.【F:src/model.py†L1452-L1549】【F:src/model.py†L1549-L1596】
- `_build_due_hour_objective_terms` trasforma le deviazioni mensili in penalità pesate per ruolo (o fallback) garantendo l'allineamento con le ore contrattuali.【F:src/model.py†L1598-L1661】

## Vincoli notti
- `_add_night_constraints` identifica gli slot notturni e impone i tetti settimanali/mensili/consecutivi hard, tenendo conto di storici e preassegnazioni. La stessa routine produce variabili per penalizzare le notti consecutive oltre la prima e per monitorare le sequenze giornaliere di notti.【F:src/model.py†L2326-L2514】
- `_add_night_pattern_constraints` disciplina le sequenze post-notte: obbligo di `SN` e riposo dopo almeno due notti consecutive, penalità soft per singole notti senza recupero `SN`-`R/F`, divieto di `Mattino` o assenze immediate dopo una notte e riposo obbligatorio dopo pattern `Notte`→`Pomeriggio`. Gestisce inoltre le assenze forzate e lo storico pre-orizzonte.【F:src/model.py†L2048-L2323】

## Vincoli pattern
- I vincoli generati in `build_model` restringono gli stati giornalieri: `SN` è attivabile solo se il giorno precedente è `Notte`; dopo una notte sono vietati `R`, `M` e `F`, coerentemente con le incompatibilità indicate nel documento.【F:src/model.py†L200-L259】
- Gli indicatori di stato costruiti in `_add_night_pattern_constraints` alimentano le variabili di penalità per il recupero dopo una singola notte.【F:src/model.py†L2096-L2206】

## Vincoli riposo
- `_add_rest_constraints` rileva le coppie di turni con gap inferiore alla soglia (tipicamente 11 ore), crea variabili booleane di violazione (soft) e applica limiti hard al numero massimo di deroghe mensili e consecutive, includendo anche i conteggi storici.【F:src/model.py†L2680-L2884】
- `_add_rest_day_windows` trasforma gli stati `R/F` in flag di riposo, penalizza la mancanza del giorno di riposo settimanale (soft) e impone il vincolo hard dei due riposi su finestre mobili di 14 giorni, considerando la copertura storica quando necessario.【F:src/model.py†L3040-L3157】

## Coerenza logica e semantica
- Il preprocessing costruisce gli insiemi di candidati idonei ai turni, escludendo automaticamente chi non può lavorare di notte (`can_work_night`) o è assente/indisponibile, assicurando che i vincoli del modello operino su domini coerenti.【F:src/preprocessing.py†L284-L380】【F:src/preprocessing.py†L444-L561】
- I test automatici coprono i casi principali per vincoli orari, notturni e di riposo, verificando sia i limiti hard sia le penalità soft tramite CP-SAT.【F:tests/test_hour_constraints.py†L1-L192】【F:tests/test_night_constraints.py†L1-L220】【F:tests/test_rest_constraints.py†L1-L210】

## Allineamento con Document 3
- Le funzionalità implementate rispettano i requisiti delle sezioni *Vincoli orari*, *Vincoli notti*, *Vincoli pattern* e *Vincoli riposo* del documento originale. Restano da introdurre le penalità di fairness su notti/weekend/festivi e la componente esplicita sul saldo ore progressivo, identificate come obiettivi futuri nel documento ma non ancora sviluppate nel codice corrente.【F:src/model.py†L1663-L1706】【F:notes/document3_summary.md†L21-L29】

## Valutazione complessiva
- La pipeline di preprocessing, modellazione e test esercita tutti i vincoli richiesti, senza evidenziare incongruenze logiche o semantiche nei controlli sulle ore, sulle notti, sulle sequenze `SN`/riposi e sulle deroghe di riposo minimo: l'esecuzione della suite completa (71 test) copre i casi limite previsti e termina con esito positivo.【F:src/preprocessing.py†L250-L420】【F:src/model.py†L1452-L1699】【F:src/model.py†L2048-L2324】【F:src/model.py†L2680-L2866】【F:src/model.py†L3040-L3176】【F:tests/test_hour_constraints.py†L1-L192】【F:tests/test_night_constraints.py†L1-L220】【F:tests/test_rest_constraints.py†L1-L210】【e885a0†L1-L4】
- Rispetto agli obiettivi del documento 3, il progetto è coerente per la parte di vincoli implementati; le uniche funzionalità ancora mancanti riguardano la fairness (distribuzione notti/weekend/festivi) e l'ottimizzazione del saldo ore a lungo termine, da pianificare come estensioni successive.【F:src/model.py†L1598-L1706】【F:notes/document3_summary.md†L21-L29】
