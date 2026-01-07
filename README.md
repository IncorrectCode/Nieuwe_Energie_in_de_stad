🎯 Energy Label Prediction & Priority Scoring Pipeline

Dit project bevat twee Python-scripts voor het analyseren van woningdata uit PostgreSQL, het voorspellen van energielabels en het berekenen van een prioriteitsscore ten behoeve van besluitvorming.

📂 Bestanden
🔹 1. random_forest_classifier.py

Script dat:

data uit PostgreSQL inlaadt

een Random Forest Classifier traint

ontbrekende energielabels voorspelt

de voorspelde labels terugschrijft naar de database

🔹 2. prioriteitsscore_definitief.py

Script dat:

data uit PostgreSQL inlaadt

features encodeert & schaalt

PCA toepast

clustering uitvoert (K-Means)

een gecombineerde prioriteitsscore (PC1 + clustergewicht) berekent

de score terugschrijft naar PostgreSQL


Functionaliteit samengevat
| Functionaliteit            | Bestandsnaam                     |
| -------------------------- | -------------------------------- |
| Voorspellen energielabels  | `random_forest_classifier.py`    |
| Berekenen prioriteitsscore | `prioriteitsscore_definitief.py` |
| Database ophalen & updaten | beide                            |
| Machine learning model     | Random Forest                    |
| Dimensionality reduction   | PCA                              |
| Clustering                 | K-Means                          |


Installatie
1️⃣ Clone de repository
git clone <repository-url>
cd <repo-map>

2️⃣ Installeer vereisten
pip install -r requirements.txt

📦 Benodigde libraries

pandas

numpy

scikit-learn

psycopg2

scipy

Tip: werk in een virtual environment

🔐 Belangrijk: database-gegevens

De scripts bevatten database-connecties.

👉 Plaats GEEN echte wachtwoorden op GitHub.

Aanbevolen aanpak:

✔ gebruik .env file
✔ laad met python-dotenv
✔ voeg .env toe aan .gitignore

Voorbeeld:

DB_HOST=...
DB_NAME=...
DB_USER=...
DB_PASS=...

▶️ Uitvoeren
Energielabels voorspellen
python random_forest_classifier.py

Prioriteitsscore berekenen
python prioriteitsscore_definitief.py

📊 Metrics & validatie

Het prioriteitsscore-script berekent o.a.:

Spearman rank correlation

Silhouette score

Deze worden in de console geprint voor kwaliteitscontrole.

🧠 Modeldetails
Random Forest (energielabels)

500 trees

bootstrap sampling

log2 feature selection

train/test split 80/20

Prioriteitsscore

PCA (1 component → PC1 score)

schaaltransformatie (StandardScaler)

K-Means clustering (k = 4)

combinatie:

70% PC1 + 30% clustergewicht

🛡️ Disclaimer

Dit project verwerkt:

woningdata

databaseverbindingen

🔸 Let op privacy
🔸 Versleutel wachtwoorden
🔸 Publiceer geen gevoelige data

🧭 Toekomstige uitbreidingen

API-endpoint toevoegen

Docker-container

Model opslaan naar .pkl

Automatische ML-pipeline (MLflow)

Dash/Streamlit dashboard
