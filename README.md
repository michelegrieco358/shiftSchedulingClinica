# Shift Scheduling Clinica Loader

Questo repository contiene il loader dei dati per il progetto di pianificazione turni della clinica.

## Requisiti

Prima di eseguire il loader assicurarsi di installare le dipendenze Python:

```bash
pip install -r requirements.txt
```

## Utilizzo

Per eseguire il caricamento completo dei dati usando i file CSV forniti nella directory corrente:

```bash
python -m loader --config config.yaml --data-dir .
```

Opzionalmente è possibile esportare i DataFrame intermedi in CSV di debug:

```bash
python -m loader --config config.yaml --data-dir . --export-csv
```

I file verranno salvati nella cartella `_expanded` all'interno della directory dati.
Per scegliere una destinazione alternativa è possibile indicare `--export-dir` (percorso
assoluto o relativo alla directory dati).

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
