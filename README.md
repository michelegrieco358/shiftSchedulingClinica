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
