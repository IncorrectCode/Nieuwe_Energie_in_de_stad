import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier

# === Dataset inladen ===
df = pd.read_csv("dataset_utrecht_opgemaakt.csv", index_col=0, header=0)

# === Targetkolom controleren ===
target_column = "energielabel"
if target_column not in df.columns:
    raise ValueError(f"Kolom '{target_column}' bestaat niet in je dataset!")

df = df.dropna(subset=[target_column])

# === Alleen numerieke features gebruiken ===
X = df.select_dtypes(include=["number"]).copy()
y = df[target_column]

if X.empty:
    raise ValueError("Geen numerieke features gevonden! Controleer de dataset.")

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === LazyPredict Classificatie ===
print("Probleemtype: Classificatie")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# === Resultaten bekijken ===
print("\n=== Model ranking ===")
print(models.sort_values(by="Accuracy", ascending=False).head(10))

# === Feature importances met RandomForest ===
def show_feature_importances(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)
    print("\n=== Feature Importances (RandomForest) ===")
    print(importances)

# Roep de functie aan
show_feature_importances(X_train, y_train)
