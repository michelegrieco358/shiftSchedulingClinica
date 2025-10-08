# Valutazione richieste coperture reparto-specifiche

## Stato attuale del codice (`loader/coverage.py`)

- Le funzioni `load_coverage_groups` e `load_coverage_roles` richiedono già la colonna `reparto_id` attraverso `_ensure_cols`, puliscono il campo con `str.strip()` e alzano `LoaderError` se manca il valore; inoltre gestiscono i duplicati usando la chiave estesa che include `reparto_id` e mantengono la colonna nei DataFrame restituiti.【F:loader/coverage.py†L35-L109】
- `validate_groups_roles` unisce `roles` e `groups` anche su `reparto_id`, include il reparto nei messaggi d'errore e nelle aggregazioni, e mantiene l'informazione in tutte le righe sintetiche generate.【F:loader/coverage.py†L111-L196】
- `expand_requirements` effettua merge su `coverage_code`, `shift_code` e `reparto_id`, riporta errori con il reparto e conserva la colonna nei DataFrame finali.【F:loader/coverage.py†L198-L252】
- `build_slot_requirements` aggrega su (`coverage_code`, `shift_code`, `reparto_id`, `ruolo`) e fa il merge con gli slot includendo `reparto_id`, restituendo però un DataFrame finale privo della colonna reparto.【F:loader/coverage.py†L254-L292】

### Gap rispetto alle modifiche richieste

1. **Normalizzazione in uppercase**: attualmente i campi stringa vengono solo `strip`-pati. Non viene applicato `str.upper()` su `coverage_code`, `shift_code`, `gruppo`, `ruolo`, `reparto_id` come richiesto.
2. **Errori sul `month_plan`**: il DataFrame risultante da `build_slot_requirements` perde `reparto_id`, mentre il requisito finale chiede di mantenerlo nell'output principale.
3. **Eccezioni specifiche**: il codice solleva `LoaderError` (non `ValueError`), ma i messaggi soddisfano già la richiesta di includere il reparto. Occorre decidere se allinearsi al tipo d'eccezione richiesto oppure mantenere la gerarchia attuale.
4. **Immutabilità dei DataFrame**: alcune funzioni modificano DataFrame di input (ad esempio `build_slot_requirements` fa `dropna` in place sul merge). Per rispettare l'indicazione "Non mutare DF in-place" andrebbero creati sempre DataFrame derivati senza side-effect.

## Stato degli script di controllo (`scripts/check_data.py`)

- Le funzioni `_check_coverage_groups` e `_check_coverage_roles` gestiscono già la presenza di `reparto_id`, verificano duplicati sulle chiavi estese e propagano i valori normalizzati. Anche la validazione incrociata considera il reparto.【F:scripts/check_data.py†L516-L676】

### Gap residui

1. **Uppercase/normalizzazione**: analogamente al loader, gli script di check non applicano `upper()` sui campi chiave; se è un vincolo forte, andrebbe aggiunto.
2. **Gestione colonne obbligatorie**: la richiesta chiede di fallire se `reparto_id` manca. Oggi lo script accetta ancora file senza reparto (usa `dept_col = None`), quindi andrebbe irrigidito per richiedere sempre `reparto_id`.

## Conclusioni

Gran parte delle modifiche richieste risultano già presenti nel codice, in particolare le chiavi di join e la propagazione di `reparto_id`. Restano da coprire: normalizzazione in uppercase, inclusione esplicita di `reparto_id` nell'output di `build_slot_requirements`, evitare mutazioni in-place e rendere obbligatoria la colonna `reparto_id` nei check script, oltre ad eventuali allineamenti sul tipo d'eccezione richiesto.
