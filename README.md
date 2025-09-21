# DashboardSimulator-Orogel
Questo progetto simula dati ambientali e produttivi e li visualizza attraverso una dashboard interattiva.

Struttura del Progetto:  
DashboardOrogel-Simulator.py: Punto di ingresso per avviare la simulazione, contenente la logica di simulazione di generazione dati.  
riepilogo_2020_2025_Tutte_le_colture.csv: File di output generato dal simulatore dall'anno default all'anno selezionato.  
requirements.txt: Elenco delle dipendenze Python.  
Prerequisiti  
Testato e funzionante su Python 3.13.0
Installazione
Clona il repository o scarica i file in una cartella locale.
Crea e attiva un ambiente virtuale (consigliato):  
python -m venv venv  
# Su Windows  
.\venv\Scripts\activate  
# Su macOS/Linux  
source venv/bin/activate  
Installa tutte le dipendenze necessarie:  
pip install -r requirements.txt  
Utilizzo  
Per utilizzare l'applicazione, segui questi due passaggi:  

Esegui il simulatore per generare il file di dati:

python main.py
Questo creerà il file simulated_vineyard_data.csv nella stessa cartella.

Avvia la dashboard per visualizzare i dati:

python dashboard.py
Apri il browser e vai all'indirizzo http://127.0.0.1:8050/ per vedere la dashboard.
