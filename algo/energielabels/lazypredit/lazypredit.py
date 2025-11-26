import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# === Dataset inladen ===
df = pd.read_csv("dataset_utrecht_opgemaakt.csv", header=0)

# === Targetkolom controleren ===
target_column = "energielabel"
if target_column not in df.columns:
    raise ValueError(f"Kolom '{target_column}' bestaat niet in je dataset!")

df = df.dropna(subset=[target_column])

# === Kolommen uitsluiten ===
kolommen_weg = [
    "huisnummer", "oppervlakte_scheidingmuur", "identificatie", "geom",
    "huisletter", "straatnaam", "woonplaats", "verspringt_met_buren"
]

# === Features ===
X = df.drop(columns=[target_column] + [c for c in kolommen_weg if c in df.columns], errors="ignore")

# === One-hot encoding voor categorische kolommen ===
X = pd.get_dummies(X, drop_first=True)

# === Targetvariabele ===
y = df[target_column]

# Controle
if X.empty:
    raise ValueError("Geen features gevonden! Controleer de dataset.")

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === LazyPredict Classificatie ===
print("Probleemtype: Classificatie")
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print("\n=== Model ranking ===")
print(models.sort_values(by="Accuracy", ascending=False).head(10))


# === Feature importances met RandomForest ===
def show_feature_importances(X_train, y_train, feature_names):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\n=== Feature Importances (RandomForest) ===")
    print(importances)

show_feature_importances(X_train, y_train, X.columns)


# === Hyperparameter tuning RandomForest ===
print("\n=== Hyperparameter tuning RandomForest ===")

param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nBeste hyperparameters gevonden:")
print(random_search.best_params_)

print("\nBeste CV-score:")
print(random_search.best_score_)

best_rf = random_search.best_estimator_

print("\nAccuracy van het beste model op de test set:")
print(best_rf.score(X_test, y_test))
