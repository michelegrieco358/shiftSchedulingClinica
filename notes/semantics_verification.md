# Verifica della semantica del loader

Questo documento riassume le verifiche manuali svolte sui moduli del loader che gestiscono la logica di turni e coperture.

## Turni di domanda di default
- `TURNI_DOMANDA` è definito come `{M, P, N}` in `loader/utils.py`.
- Durante la costruzione degli slot, il loader richiede un opt-in esplicito da `reparto_shift_map.csv` per ogni turno con durata positiva che **non** appartiene a `TURNI_DOMANDA`. Questo conferma che M/P/N sono abilitati implicitamente, mentre gli altri turni di domanda devono essere abilitati in modo esplicito.

## Idoneità ruolo-turno
- `load_shift_role_eligibility` garantisce che ogni turno di domanda presente nel catalogo abbia almeno una riga `(shift_code, role)` marcata come allowed. Non esiste alcuna compilazione automatica di idoneità “di default”.

## Logica di copertura
- `load_coverage_roles` vieta duplicati solo per la stessa quintupla `(coverage_code, shift_code, reparto_id, gruppo, role)`. Di conseguenza, lo stesso ruolo può comparire sotto più codici di copertura per lo stesso reparto e turno.
- `expand_requirements` esegue un merge many-to-many tra month plan e gruppi di copertura, quindi il month plan può attivare più codici di copertura per lo stesso reparto e turno nello stesso giorno.
- `load_coverage_groups` garantisce l’unicità di `(coverage_code, shift_code, reparto_id, gruppo)` e conserva, per ogni gruppo, il `total_staff` dichiarato, l’eventuale `overstaff_cap` e l’elenco dei ruoli che devono contribuire al minimo del gruppo.

## Vincoli di gruppo
- In fase di espansione, ogni riga del month plan viene combinata con tutti i propri gruppi, mantenendo `total_staff`, l’eventuale cap di overstaff e la lista dei ruoli come `ruoli_totale_set`. Ogni combinazione `(slot_id, coverage_code, gruppo)` rimane quindi distinta e arriva intatta alla fase di modellazione.【F:loader/coverage.py†L120-L168】【F:loader/coverage.py†L181-L205】
- Quando il modello CP-SAT costruisce i vincoli di gruppo, itera ciascuna di queste combinazioni, interpreta l’insieme dei ruoli e crea un vincolo `somma(assegnazioni per i ruoli dell’insieme) ≥ total_staff` (e `≤ cap` se previsto). Il vincolo conta tutte le variabili `x[(dipendente, slot_id)]` i cui dipendenti appartengono a quei ruoli, senza escludere assegnazioni già usate da altri gruppi.【F:src/model.py†L3418-L3479】【F:src/model.py†L3501-L3546】
- Se lo stesso ruolo compare in più gruppi, le assegnazioni di quel ruolo soddisfano più vincoli contemporaneamente. Esempio: per un turno attivo con due gruppi `A` (minimo 2 infermieri) e `B` (minimo 4 tra infermieri e OSS), assegnare 2 infermieri e 2 OSS comporta:
  - il vincolo di ruolo “infermieri” viene soddisfatto perché la domanda minima (2) deriva dal gruppo `A`;
  - il vincolo di gruppo `A` è soddisfatto dai 2 infermieri assegnati;
  - il vincolo di gruppo `B` vede 4 assegnazioni in ruoli ammessi (`2 infermieri + 2 OSS`) e quindi è anch’esso soddisfatto.
  Le stesse variabili di decisione concorrono quindi a più disuguaglianze, motivo per cui i minimi di gruppo non si sommano automaticamente: per renderli additivi bisognerebbe creare slot separati (o vincoli disgiunti) per ciascun gruppo, come discusso nella review.

## Ore di assenza configurate
- Il loader delle ferie utilizza direttamente `absence_full_day_hours_effective_h` calcolato in `load_employees`; qualora il valore manchi, è possibile passare un fallback esplicito tramite il parametro opzionale `absence_hours_h` di `load_leaves`. In assenza di entrambi i valori il caricamento fallisce, così da evitare assenze con durata indefinita.【F:loader/employees.py†L217-L318】【F:loader/leaves.py†L58-L135】
- È possibile definire ore giornaliere specifiche per ruolo attraverso `defaults.absences.full_day_hours_by_role_h`; ad esempio, la configurazione campione assegna 7.5 h a `INFERMIERE` e `OSS`, e 7.0 h a `CAPOSALA`. Questi valori sono impiegati da `load_employees` per popolare `absence_full_day_hours_effective_h`, che resta disponibile nel DataFrame dei dipendenti come riferimento puntuale per persona o ruolo.【F:config.yaml†L21-L27】【F:loader/employees.py†L217-L318】
