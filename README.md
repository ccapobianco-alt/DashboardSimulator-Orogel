# Dashboard Produzione - OROGEL

Dashboard interattiva (Plotly Dash) per analizzare produzione agricola simulata con meteo e pH del suolo su diverse basi di analisi (giornaliera, mensile, bimestrale, semestrale, annuale).

## Funzionalità
- 3 colture analizzate: **Pisello, Spinacio, Fagiolino** (+ *Tutte le colture*).
- Generazione **deterministica** dei dati per anno+coltura → i totali tornano sempre.
- Meteo e pH suolo giornalieri/mensili, aggregazioni mensili/bimestrali/semestrali, correlazioni (Pearson).
- **Storico (2020–2025)** per quantità annue con resoconto.
- Export CSV di tutti i riepiloghi.
- UI con loghi, stile personalizzato in `assets/style.css`.

## Requisiti
- Python 3.9+ consigliato
- Librerie: vedi `requirements.txt`

## Installazione
```bash
git clone https://github.com/<tuo-utente>/orogel-dashboard.git
cd orogel-dashboard
python -m venv .venv
# per Windows:
.venv\Scripts\activate
# per macOS/Linux:
source .venv/bin/activate
```
- installa tutte le dipendenze necessarie:
```bash
pip install -r requirements.txt
```
## Avvio
```bash
python DashboardOrogel-Sim.py
```
Automaticamente si aprirà la dashboard all'indirizzo: http://127.0.0.1:/8050/

