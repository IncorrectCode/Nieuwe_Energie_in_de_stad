import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# === Dataset inladen ===
df = pd.read_csv("dataset_utrecht_opgemaakt.csv")

# === Kolommen die we niet kunnen gebruiken ===
kolommen_weg = [
    "huisnummer", "oppervlakte_scheidingmuur", "identificatie", "geom",
    "huisletter", "straatnaam", "woonplaats", "verspringt_met_buren",
    "prioriteitsscore"   # bestaat al maar gebruiken we niet
]

df = df.drop(columns=[c for c in kolommen_weg if c in df.columns], errors="ignore")

# === Categorische kolommen encoden ===
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

# === PCA uitvoeren ===
pca = PCA(n_components=1)
pc1 = pca.fit_transform(X_scaled).flatten()

# === PC1 normaliseren naar 0–100% ===
pc1_score = 100 * (pc1 - pc1.min()) / (pc1.max() - pc1.min())

# === KMeans Clustering ===
k = 4  # aantal clusters (kan je aanpassen)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# === Clusterprioriteit bepalen ===
df_priority = pd.DataFrame({
    "cluster": clusters,
    "pc1": pc1_score
})

cluster_means = df_priority.groupby("cluster")["pc1"].mean().sort_values(ascending=False)

cluster_rank_map = {
    cluster: rank for rank, cluster in enumerate(cluster_means.index)
}

# === Per woning cluster-prioriteit 0–100 op basis van cluster ranking ===
cluster_priority = df_priority["cluster"].map(cluster_rank_map)

cluster_priority_norm = 100 * (cluster_priority - cluster_priority.min()) / (
        cluster_priority.max() - cluster_priority.min()
)

# === Eindscores combineren ===
# 70% PC1 + 30% cluster score (weegt continue + discrete structuur)
final_score = 0.7 * df_priority["pc1"] + 0.3 * cluster_priority_norm

# === Final score toevoegen aan dataset ===
df_result = df.copy()
df_result["prioriteitsscore_voorspeld"] = final_score

# === Opslaan ===
df_result.to_csv("prioriteitsscore_voorspeld.csv", index=False)

print("Klaar! Prioriteitsscore opgeslagen in prioriteitsscore_voorspeld.csv")
