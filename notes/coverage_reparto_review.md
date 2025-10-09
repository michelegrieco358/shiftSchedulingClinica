# Sintesi coperture reparto-specifiche

## Stato del loader (loader/coverage.py)

- load_coverage_groups e load_coverage_roles normalizzano i campi chiave (coverage_code, shift_code, 
eparto_id, gruppo, 
ole) con strip() e upper(), sollevando LoaderError quando valori o duplicati di reparto risultano errati.
- alidate_groups_roles ed expand_requirements propagano sempre 
eparto_id nei DataFrame derivati e nei messaggi, basandosi sul join (coverage_code, shift_code, 
eparto_id).
- uild_slot_requirements lavora su copie e restituisce slot_id, 
eparto_id, 
ole, demand mantenendo l'ordinamento richiesto.

## Stato dei controlli (scripts/check_data.py)

- I check richiedono 
eparto_id, deduplicano per reparto e normalizzano i campi in uppercase.
- Il reader CSV usa ora utf-8-sig, quindi i file con BOM (es. employees.csv) vengono letti senza interventi manuali.

## Prossimi miglioramenti suggeriti

- Ampliare i test di integrazione sulle combinazioni overstaffing e ruoli cross reparto.
- Aggiornare la documentazione operativa con esempi di CSV allineati alla normalizzazione automatica del loader.
