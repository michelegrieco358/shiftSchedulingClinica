# Verifica implementazione vincoli

## Vincoli orari
- `_add_hour_constraints` calcola le ore assegnate per settimana e mese, gestendo storici e contando le assenze come ore lavorate se previsto. Impone i tetti settimanali/mensili hard e crea variabili di deviazione per il monte ore teorico da penalizzare nell'obiettivo.
- `_build_due_hour_objective_terms` trasforma le deviazioni mensili in penalità ponderate in funzione del ruolo e dei parametri di configurazione.

## Vincoli notti
- `_add_night_constraints` filtra gli slot notturni e applica i limiti settimanali, mensili e consecutivi (hard), costruendo inoltre variabili di penalità per le notti consecutive oltre la prima.
- `_add_night_pattern_constraints` forza la sequenza `SN`-riposo dopo almeno due notti consecutive, impedisce `Mattino` e assenze subito dopo una notte, penalizza le singole notti senza recupero e impone il riposo dopo la sequenza `Notte`-`Pomeriggio`.

## Vincoli pattern
- Gli indicatori di stato e i vincoli su `SN`, `R`, `M` e `F` impediscono combinazioni vietate dopo un turno notturno e collegano le preferenze soft alle variabili di penalità incluse in funzione obiettivo.

## Vincoli riposo
- `_add_rest_constraints` individua le coppie di turni con gap inferiore alla soglia di 11 ore, crea le variabili di violazione (soft) e applica i limiti massimi mensili e consecutivi sulle deroghe (hard).
- `_add_rest_day_windows` calcola i giorni di riposo (R o F) su finestre mobili, applicando un vincolo hard sui 14 giorni e penalizzando (soft) la mancanza del riposo settimanale.

## Note aggiuntive
- I vincoli di copertura (hard) e di idoneità ai turni derivano dal preprocessing che rimuove i candidati non abilitati (es. `can_work_night`).
- Non sono ancora presenti nel modello le penalità di fairness su notti/weekend menzionate fra gli obiettivi futuri del documento.

## Valutazione stato attuale
- L'implementazione dei vincoli orari, notturni, di pattern e di riposo risulta coerente con i requisiti di Document 3 e coperta dai test unitari/di integrazione (`tests/test_hour_constraints.py`, `tests/test_night_constraints.py`, `tests/test_rest_constraints.py`).
- L'intera suite `pytest` passa (70 test) confermando l'assenza di regressioni note; restano da sviluppare le penalità di fairness pianificate (notti/weekend) per completare gli obiettivi di business.
