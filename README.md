# 🎯 Energy Label Prediction & Priority Scoring Pipeline

Dit project bevat twee Python-scripts voor het analyseren van woningdata uit PostgreSQL, het voorspellen van energielabels en het berekenen van een prioriteitsscore ten behoeve van besluitvorming.

---

## 📂 Bestanden

### 1️⃣ random_forest_classifier.py
- Laadt data uit PostgreSQL  
- Traindt een Random Forest Classifier  
- Voorspelt ontbrekende energielabels  
- Schrijft voorspellingen terug naar de database  

### 2️⃣ prioriteitsscore_definitief.py
- Laadt data uit PostgreSQL  
- Encodeert en schaalt features  
- Voert PCA uit  
- Clustert met K-Means  
- Berekent gecombineerde prioriteitsscore  
- Schrijft resultaten terug naar de database  

---

## 🚀 Functionaliteitsoverzicht

| Functionaliteit                 | Script                          |
|---------------------------------|----------------------------------|
| Voorspellen van energielabels   | random_forest_classifier.py      |
| Berekenen van prioriteitsscore  | prioriteitsscore_definitief.py   |
| Database ophalen & updaten      | Beide                            |
| Dimensionality reduction (PCA)  | prioriteitsscore_definitief.py   |
| Clustering (K-Means)            | prioriteitsscore_definitief.py   |

---

## ⚙️ Installatie

### 1️⃣ Repository clonen
```bash
git clone <repository-url>
cd <repository-map>
```

### 2️⃣ (Aanbevolen) Virtual environment
```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Dependencies installeren
```bash
pip install -r requirements.txt
```

---

## 📦 Vereiste libraries

- pandas  
- numpy  
- scikit-learn  
- psycopg2  
- scipy  

---

## 🔐 Database-configuratie

⚠️ Gebruik **geen hardcoded wachtwoorden** in GitHub.

Gebruik een `.env` bestand:

```
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASS=
```

👉 Voeg `.env` toe aan `.gitignore`.

---

## ▶️ Gebruik

### Energielabels voorspellen
```bash
python random_forest_classifier.py
```

### Prioriteitsscore berekenen
```bash
python prioriteitsscore_definitief.py
```

---

## 🧠 Modelinformatie

### Random Forest energielabelmodel
- n_estimators = 500  
- bootstrap = True  
- max_features = log2  
- train/test split = 80/20  

### Prioriteitsscoremodel
- StandardScaler  
- PCA (1 component → PC1)  
- K-Means (k = 4)  
- Eindscore:
```
0.7 * PC1 + 0.3 * clustergewicht
```

---

## 📊 Validatiemetrics

Worden automatisch weergegeven in de console:

- Spearman rank correlation  
- Silhouette score  

---

## 🛡️ Disclaimer

Dit project werkt met:

- woningdata  
- databaseverbindingen  

Let op:

- privacy
- databeveiliging
- credentialbeheer

---

## 🧭 Mogelijke uitbreidingen

- Docker container  
- Model persistentie (.pkl)  
- API-endpoint  
- Dashboard (Streamlit/Dash)  
- MLflow integratie
