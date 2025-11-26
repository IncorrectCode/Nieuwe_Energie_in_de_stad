import pandas as pd
import numpy as np
import psycopg2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === PostgreSQL connectie ===
try:
    conn = psycopg2.connect(
        host="145.38.184.32",
        port="5432",
        dbname="RetrofitEuropeGroup",
        user="DATABIM2526AB",
        password="pv6QGtNJ#rs\\#<xoc#'n"
    )
    print("✔ Verbonden met de PostgreSQL database!")
except Exception as e:
    print("❌ Kon geen verbinding maken met de PostgreSQL database!")
    print("Foutmelding:", e)
    raise SystemExit("Script gestopt vanwege database verbindingsfout!")

# === SQL-query naar tabel ===
query = """
SELECT *
FROM futurefactory.utrecht_model;
"""

try:
    df = pd.read_sql_query(query, conn)
    print(f"✔ Data succesvol ingeladen! Aantal rijen: {len(df)}")
except Exception as e:
    print("❌ Fout bij inladen van database tabel!")
    print("Foutmelding:", e)
    conn.close()
    raise SystemExit("Script gestopt vanwege fout in SQL-query of dataverwerking!")

conn.close()

# === Kolommen die we niet kunnen gebruiken ===
kolommen_weg = [
    "huisnummer", "oppervlakte_scheidingmuur", "identificatie", "geom",
    "huisletter", "straatnaam", "woonplaats", "verspringt_met_buren",
    "prioriteitsscore"
]

df = df.drop(columns=[c for c in kolommen_weg if c in df.columns], errors="ignore")

# === Categorical encoding ===
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# === Missings oplossen ===
df_encoded = df_encoded.fillna(df_encoded.median())

# === Schalen ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

# === PCA ===
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).flatten()

# === Normaliseren ===
pc1_score = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min())

# === Clustering ===
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_scaled)

df_priority = pd.DataFrame({"cluster": clusters, "pc1": pc1_score})
cluster_means = df_priority.groupby("cluster")["pc1"].mean().sort_values(ascending=False)

cluster_rank_map = {cluster: rank for rank, cluster in enumerate(cluster_means.index)}
cluster_priority = df_priority["cluster"].map(cluster_rank_map)

cluster_priority_norm = 100 * (cluster_priority - cluster_priority.min()) / (
    cluster_priority.max() - cluster_priority.min()
)

# === Final score ===
final_score = 0.7 * df_priority["pc1"] + 0.3 * cluster_priority_norm

df_result = df.copy()
df_result["prioriteitsscore_voorspeld"] = final_score

# === Opslaan ===
df_result.to_csv("prioriteitsscore_voorspeld.csv", index=False)

print("🎯 Klaar! prioriteitsscore_voorspeld.csv is opgeslagen.")
