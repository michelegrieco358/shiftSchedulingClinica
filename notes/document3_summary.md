# Sintesi dei requisiti chiave di Document 3

## Vincoli orari
- Il piano deve rispettare un monte ore mensile teorico per dipendente, con la possibilità di introdurre penalità (vincolo soft) sulle deviazioni dal valore contrattuale.
- Sono previsti limiti massimi inderogabili sulle ore mensili e settimanali (vincoli hard).

## Vincoli notti
- L'idoneità ai turni notturni è definita per ruolo e, se necessario, per singolo dipendente.
- Esistono limiti settimanali, mensili e consecutivi al numero di turni notturni (hard) con eventuali penalità per notti consecutive oltre la prima (soft).

## Vincoli pattern
- Dopo due o più notti consecutive deve comparire la sequenza `SN` (smonto notte) seguita da un giorno di riposo (vincolo hard).
- Dopo una singola notte è raccomandata (vincolo soft) la sequenza `SN` seguita da riposo o assenza.
- La sequenza `Notte` → `Pomeriggio` richiede un giorno di riposo successivo (hard).
- Il turno `Mattino` non può seguire un turno `Notte` e non sono ammessi turni notturni immediatamente prima di un'assenza pianificata (hard).

## Vincoli riposo
- Il riposo minimo di 11 ore fra due turni è un vincolo soft; tuttavia sono fissati limiti massimi alle deroghe mensili e consecutive (hard).
- Ogni lavoratore deve avere almeno un giorno di riposo a settimana (vincolo soft) e almeno due giorni di riposo nell'arco di due settimane consecutive (vincolo hard).

## Obiettivi aggiuntivi
- La funzione obiettivo include penalità per migliorare la fairness nella distribuzione di notti, weekend e festivi oltre al contenimento del saldo ore.
