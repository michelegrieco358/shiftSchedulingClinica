# Revisione tecnica del progetto Shift Scheduling

## 1. Stato generale
- Il caricatore dei dati (`loader`) costruisce correttamente il contesto con tutte le tabelle previste, incluse le informazioni su lock, storia turni e calendari, in linea con la specifica funzionale.【F:loader/__init__.py†L200-L259】
- Il modello CP-SAT instanzia le variabili di assegnazione e di stato, applicando vincoli su ore, notti, pattern post-notte, riposo settimanale e penalizzazioni per straordinari e lavoro cross reparto.【F:src/model.py†L260-L337】【F:src/model.py†L770-L835】
- Le informazioni di calendario includono flag weekend/festività utili per le metriche di fairness richieste dal documento di analisi.【F:loader/calendar.py†L30-L109】

## 2. Criticità rispetto alla specifica

### 2.1 Lock di pre-assegnazione non applicati
- La specifica prevede la possibilità di forzare o vietare assegnazioni turno-dipendente.【F:docspec_parsed.txt†L8-L14】
- Il loader costruisce `locks_must_df` e `locks_forbid_df`, e il `ModelContext` li espone.【F:loader/__init__.py†L218-L259】【F:src/model.py†L39-L64】
- Tuttavia `build_model` e i relativi vincoli non referenziano mai `context.locks_must` o `context.locks_forbid`; nessuna variabile o constraint usa queste tabelle, per cui i lock configurati vengono ignorati in fase di ottimizzazione.【F:src/model.py†L260-L337】【F:src/model.py†L770-L835】
  - Impatto: vincoli hard richiesti dal cliente possono essere infranti senza che il modello se ne accorga.

### 2.2 Vincolo hard sul saldo progressivo mensile mancante
- Il documento definisce un vincolo hard sul delta di saldo ore tra inizio e fine mese.【F:docspec_parsed.txt†L67-L68】
- Il loader calcola `max_balance_delta_month_h` per default e override dipendente, rendendolo disponibile nel dataset.【F:loader/employees.py†L184-L423】
- Nel modello, i parametri orari estratti per dipendente includono solo ore dovute e tetti settimanali/mensili; il delta massimo non viene mai letto né trasformato in un vincolo.【F:src/model.py†L770-L835】
  - Impatto: il solver può produrre piani che superano la variazione massima di saldo imposta dal cliente, violando un vincolo classificato come HARD.

### 2.3 Penalità di fairness su notti/weekend/festivi assenti
- La specifica richiede una componente d’obiettivo che bilanci la distribuzione di notti, weekend e festività lavorate tramite penalizzazioni di varianza.【F:docspec_parsed.txt†L85-L107】
- Sebbene i dati di calendario forniscano i flag necessari (weekend, festività), la funzione obiettivo del modello include solo penalità su ore dovute, notti consecutive, recupero singola notte, riposi e cross-reparto.【F:src/model.py†L284-L303】
- Non sono presenti variabili né termini che misurino la dispersione di notti/weekend/festivi assegnati; la fairness resta completamente non implementata.
  - Impatto: il piano risultante può sbilanciare sistematicamente i turni gravosi tra gli operatori, contravvenendo a un obiettivo organizzativo esplicito.

### 2.4 Penalità di stabilità del piano non implementata
- L’analisi funzionale prevede una penalità per le variazioni rispetto al piano storico.【F:docspec_parsed.txt†L101-L104】
- Il modello usa la storia solo per ricostruire vincoli (pattern notti, riposi, limiti già consumati), ma non aggiunge alcun termine d’obiettivo che penalizzi deviazioni rispetto agli assegnamenti precedenti.【F:src/model.py†L213-L335】
  - Impatto: il solver può rivoluzionare i turni rispetto allo storico senza nessun costo, riducendo la stabilità operativa richiesta.

## 3. Test esistenti
- La suite `pytest` copre ampiamente il loader e i vincoli principali, ma non ci sono test che verifichino l’applicazione dei lock, né l’esistenza dei termini di fairness/stabilità o del vincolo sul saldo progressivo. L’assenza di tali controlli rende questi regressioni invisibili.【F:src/model.py†L260-L337】

## 4. Raccomandazioni
1. **Integrare i lock nel modello**: creare vincoli che forzino `x[e,s]=1` per i lock “must” e `x[e,s]=0` per i lock “forbid”, rispettando anche le incompatibilità con assenze e idoneità.
2. **Implementare il vincolo sul delta di saldo**: calcolare il saldo finale per mese usando ore assegnate + storico e imporre `|saldo_finale - saldo_iniziale| ≤ max_balance_delta_month_h` per ogni dipendente.
3. **Aggiungere penalità di fairness**: derivare per dipendente il numero di notti, weekend e festività nell’orizzonte (inclusa eventuale storia) e introdurre in funzione obiettivo termini quadrativi/assoluti che riproducano `Z_fair` descritto nel documento.
4. **Introdurre la penalità di stabilità**: confrontare le assegnazioni con la pianificazione precedente (history) e sommare un termine ponderato che misuri le differenze, come previsto da `Z_stabilità`.
5. **Estendere i test**: aggiungere casi che falliscono se i lock non sono rispettati, se il saldo varia oltre i limiti, o se i termini di fairness/stabilità mancano.

Implementare questi punti riallineerà il modello ai requisiti HARD/SOFT definiti nel documento di analisi e ridurrà il rischio di soluzioni operative non accettabili.
