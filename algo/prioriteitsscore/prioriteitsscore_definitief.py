import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Database configuratie (pas dit aan met nieuwe credentials!) ===
DB_HOST = "145.38.184.32"
DB_PORT = "5432"
DB_NAME = "RetrofitEuropeGroup"
DB_USER = "DATABIM2526AB"
DB_PASS = "pv6QGtNJ#rs\\#<xoc#'n"

# === Connectie maken met PostgreSQL ===
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    print("✔ Verbonden met de PostgreSQL database!")
except Exception as e:
    print("❌ Kon geen verbinding maken met de PostgreSQL database!")
    raise SystemExit(str(e))

# === Data ophalen ===
query = """SELECT * FROM futurefactory.utrecht_model;"""
df = pd.read_sql_query(query, conn)
print(f"✔ Data succesvol ingeladen! Aantal rijen: {len(df)}")

# === Kolommen opschonen (identificatie behouden!) ===
kolommen_weg = [
    "huisnummer", "oppervlakte_scheidingmuur", "geom", "huisletter",
    "straatnaam", "woonplaats", "verspringt_met_buren", "prioriteitsscore"
]
df_ml = df.drop(columns=[c for c in kolommen_weg if c in df.columns], errors="ignore")

# === Categorical encoding ===
df_encoded = df_ml.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# === Missende waarden oplossen ===
df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

# === Schalen ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# === PCA naar 1 score ===
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).flatten()
pc1_score = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min())

# === Clustering voor weging ===
kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)
df_priority = pd.DataFrame({"cluster": clusters, "pc1": pc1_score})

cluster_means = df_priority.groupby("cluster")["pc1"].mean().sort_values(ascending=False)
cluster_rank_map = {cluster: rank for rank, cluster in enumerate(cluster_means.index)}
cluster_priority = df_priority["cluster"].map(cluster_rank_map)
cluster_priority_norm = 100 * (cluster_priority - cluster_priority.min()) / (
        cluster_priority.max() - cluster_priority.min())

# === Eindscore berekenen ===
final_score = 0.7 * df_priority["pc1"] + 0.3 * cluster_priority_norm

# === Toevoegen aan originele data ===
df["prioriteitsscore"] = final_score

# === Nieuwe connectie voor UPDATE ===
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)
cursor = conn.cursor()

# === BULK UPDATE voorbereiden ===
update_data = [(round(float(row.prioriteitsscore), 2), str(row.identificatie))
               for _, row in df.iterrows()]

update_query = """
    UPDATE futurefactory.utrecht_model AS t
    SET prioriteitsscore = v.score
    FROM (VALUES %s) AS v(score, identificatie)
    WHERE t.identificatie = v.identificatie;
"""

# Gebruik psycopg2 mogrify
from psycopg2.extras import execute_values
execute_values(cursor, update_query, update_data)

conn.commit()
cursor.close()
conn.close()

print("🚀 Prioriteitsscores succesvol geüpdatet in PostgreSQL!")
