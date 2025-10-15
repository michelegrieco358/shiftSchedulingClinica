# Shift Scheduling Clinica

Repository di supporto al progetto di pianificazione turni per una clinica
ospedaliera. Il codice fornisce tre blocchi principali:

* **Loader (`loader/`)** – normalizza i file CSV di input e costruisce i
  DataFrame necessari al solver.
* **Preprocessing (`src/preprocessing/`)** – calcola i bilanci progressivi e i
  coefficienti di penalità dinamici richiesti dal modello.
* **Modello (`src/model/`)** – definisce il problema CP-SAT con vincoli su orari,
  notti, riposi e fairness, come descritto nella documentazione di progetto.

La cartella `data/` contiene esempi di input, mentre `scripts/` raccoglie gli
strumenti di validazione per i dati e le configurazioni.

## Requisiti

Prima di eseguire il loader assicurarsi di installare le dipendenze Python:

```bash
pip install -r requirements.txt
```

## Utilizzo

Per eseguire l'intero caricamento dati usando i file CSV forniti nella cartella
`data/`:

```bash
python -m loader --config config.yaml --data-dir data
```

Opzionalmente è possibile esportare i DataFrame intermedi in CSV di debug:

```bash
python -m loader --config config.yaml --data-dir data --export-csv
```

I file verranno salvati nella cartella `_expanded` all'interno della directory
dati. Per scegliere una destinazione alternativa è possibile indicare
`--export-dir` (percorso assoluto o relativo alla directory dati).

Il loader può essere integrato nel solver principale tramite la funzione
`load_context` che restituisce un `ModelContext` pronto per la risoluzione.

## Limiti orari caricati per dipendente

Il loader normalizza tutti i valori orari in minuti e mette a disposizione tre
grandezze da usare nei vincoli del solver:

* **`dovuto_min`** – le ore teoriche mensili previste dal contratto. Il valore è
  letto da `employees.csv` (colonna `ore_dovute_mese_h`) oppure dal default
  `defaults.contract_hours_by_role_h` indicato in `config.yaml`.
* **`max_month_min`** – il limite mensile inderogabile. Quando non è presente
  l'override `max_month_hours_h` nel CSV, viene calcolato come `1.25 × ore
  contrattuali mensili`, permettendo una tolleranza del 25% rispetto al dovuto.
* **`max_week_min`** – il limite settimanale inderogabile. Se il CSV non fornisce
  un override (`max_week_hours_h`), il loader parte dalle ore contrattuali
  mensili e le ripartisce su una settimana "media" del mese usando la formula
  `ore_mese / giorni_orizzonte × 7`. Il cap finale è `1.5 × quota settimanale` e
  viene applicato anche alle settimane parziali (iniziali/finali), così da
  impedire concentrazioni eccessive di straordinario in una singola settimana
  senza imporre limiti artificiali sui singoli giorni.

Gli stessi controlli sono replicati nello script `scripts/check_data.py`, in
modo da intercettare eventuali override errati prima dell'esecuzione del loader.

## Pesi delle penalit��

Il file `config.yaml` espone i pesi di tutte le penalit�� soft del modello, cos�� da
permettere il tuning senza modifiche al codice:

- `fairness.night_weight` e `fairness.weekend_weight` controllano
  rispettivamente la distribuzione equa dei turni notturni e dei weekend/festivi.
- `night.single_night_recovery_penalty_weight` regola le penalit�� sui pattern post-notte.
- Restano invariati i pesi gi�� presenti
  (`rest_rules.rest11_penalty_weight`, `rest_rules.weekly_rest_penalty_weight`,
  `night.extra_consecutive_penalty_weight`, `cross.penalty_weight`,
  `preassignments.change_penalty_weight`,
  `defaults.balance.due_hours_penalty_weight`,
  `defaults.balance.final_balance_penalty_weight`).

Tutti i valori devono essere numeri non negativi; impostando un peso a `0` la
relativa penalit�� viene disattivata.
