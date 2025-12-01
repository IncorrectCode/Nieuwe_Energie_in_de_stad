import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# === PostgreSQL Connectie ===
conn = psycopg2.connect(
    host="145.38.184.32",
    port="5432",
    dbname="RetrofitEuropeGroup",
    user="DATABIM2526AB",
    password="pv6QGtNJ#rs\\#<xoc#'n"
)
print("✔ Verbonden met de PostgreSQL database!")

# === Dataset inladen ===
query = "SELECT * FROM futurefactory.utrecht_model;"
df = pd.read_sql_query(query, conn)
conn.close()

# === Target ===
target_column = "energielabel"
df = df.copy()
df = df.dropna(subset=[target_column])

kolommen_weg = [
    "huisnummer", "oppervlakte_scheidingmuur", "identificatie", "geom",
    "huisletter", "straatnaam", "woonplaats", "verspringt_met_buren", "prioriteitsscore"
]

X = df.drop(columns=[target_column] + [c for c in kolommen_weg if c in df.columns])
X = pd.get_dummies(X, drop_first=True)
y = df[target_column]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Beste Model instellen ===
best_params = {
    'n_estimators': 500,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'log2',
    'max_depth': None,
    'bootstrap': True
}

best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

print("✔ Beste RandomForest model getraind en klaar voor gebruik!")

# ===============================================
# === Energielabel Voorspellingen Wegschrijven ===
# ===============================================

# Rijen zoeken met onbekend label
conn = psycopg2.connect(
    host="145.38.184.32",
    port="5432",
    dbname="RetrofitEuropeGroup",
    user="DATABIM2526AB",
    password="pv6QGtNJ#rs\\#<xoc#'n"
)

query_missing = """
SELECT *
FROM futurefactory.utrecht_model
WHERE energielabel IS NULL;
"""

df_missing = pd.read_sql_query(query_missing, conn)

if df_missing.empty:
    print("\n🚫 Geen ontbrekende energielabels gevonden!")
else:
    print(f"\n🔍 Aantal ontbrekende energielabels: {len(df_missing)}")

    X_missing = df_missing.drop(columns=[c for c in kolommen_weg if c in df_missing.columns], errors="ignore")

    # Zelfde encoding als training
    X_missing = pd.get_dummies(X_missing, drop_first=True)

    # Kolommen gelijkmaken aan trainingsdata
    X_missing = X_missing.reindex(columns=X.columns, fill_value=0)

    predictions = best_rf.predict(X_missing)

    # Updates terugschrijven naar DB
    cur = conn.cursor()
    for idx, pred in enumerate(predictions):
        identificatie = df_missing.iloc[idx]["identificatie"]

        update_sql = """
        UPDATE futurefactory.utrecht_model
        SET energielabel = %s
        WHERE identificatie = %s;
        """

        cur.execute(update_sql, (pred, identificatie))

    conn.commit()
    cur.close()
    conn.close()

    print("✔ Alle ontbrekende energielabels succesvol aangevuld en opgeslagen!")
