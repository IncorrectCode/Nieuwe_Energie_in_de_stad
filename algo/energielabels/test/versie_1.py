import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

# ==========================================================
# CONFIG
# ==========================================================
RANDOM_STATE = 42
INPUT_FILE = "databim ruwe data.xlsx"
MODEL_FILE = "energielabel_model.pkl"

# ==========================================================
# 1. DATA INLEZEN
# ==========================================================
print("📥 Excel inlezen...")
xls = pd.ExcelFile(INPUT_FILE, engine="openpyxl")

dfs = []
for sheet in xls.sheet_names:
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet, engine="openpyxl")
    print(f"Loaded sheet '{sheet}' with shape {df.shape}")
    df["_sheet"] = sheet
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True, sort=False)
print(f"Totale rijen na concat: {len(data)}")

# ==========================================================
# 2. KOLOMMEN DETECTEREN
# ==========================================================
cols = data.columns

def find_col(keyword):
    for c in cols:
        if keyword in c.lower():
            return c
    return None

col_bouwjaar = find_col("bouw")
col_daktype = find_col("dak")
col_m2 = find_col("m2") or find_col("m²") or find_col("oppervlakte")
col_label = find_col("energ") or find_col("label")

print("Gevonden kolommen:")
print(" bouwjaar:", col_bouwjaar)
print(" daktype:", col_daktype)
print(" m2:", col_m2)
print(" energielabel:", col_label)

if not all([col_bouwjaar, col_daktype, col_m2, col_label]):
    raise ValueError("❌ Niet alle benodigde kolommen gevonden!")

# ==========================================================
# 3. DATA OPSCHONEN
# ==========================================================
df = data[[col_bouwjaar, col_daktype, col_m2, col_label]].copy()

# Splits bekende/onbekende labels
df_known = df[df[col_label].notna()].copy()
df_unknown = df[df[col_label].isna()].copy()

print(f"Rijen met label (train): {len(df_known)}")
print(f"Rijen zonder label (te voorspellen): {len(df_unknown)}")

# ==========================================================
# 4. LABEL VERWERKING
# ==========================================================
if np.issubdtype(df_known[col_label].dtype, np.number):
    print("📊 Numeriek label gevonden – converteren naar A–G klassen...")

    # Inspecteer distributie
    min_val, max_val = df_known[col_label].min(), df_known[col_label].max()
    print(f"Label-range: {min_val} → {max_val}")

    # Automatisch dynamische bins aanmaken (7 intervallen = A–G)
    bins = np.linspace(min_val, max_val, 8)
    labels = ["G", "F", "E", "D", "C", "B", "A"]

    df_known["label_class"] = pd.cut(df_known[col_label], bins=bins, labels=labels)
    y = df_known["label_class"]
else:
    y = df_known[col_label].astype(str).str.upper().str.strip()

X = df_known[[col_bouwjaar, col_daktype, col_m2]]

# Controle labelverdeling
print("📈 Labelverdeling:\n", y.value_counts())

# Verwijder labels met te weinig voorbeelden
label_counts = y.value_counts()
valid_labels = label_counts[label_counts >= 2].index
X = X[y.isin(valid_labels)]
y = y[y.isin(valid_labels)]

# Controleer op lege kolommen
empty_cols = [c for c in X.columns if X[c].isna().all()]
if empty_cols:
    print(f"⚠️ Volledig lege kolommen verwijderd: {empty_cols}")
    X = X.drop(columns=empty_cols)

# Feature-typen bepalen
num_features = [c for c in X.columns if X[c].dtype in [np.number, "int64", "float64"]]
cat_features = [c for c in X.columns if c not in num_features]

print("Num. features:", num_features)
print("Cat. features:", cat_features)

# ==========================================================
# 5. PIPELINE
# ==========================================================
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced"  # ✅ Belangrijk bij scheve labelverdeling
    ))
])

# ==========================================================
# 6. TRAINEN & TESTEN
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training met X shape: {X_train.shape} y shape: {y_train.shape}")

try:
    print("Doing 3-fold CV (accuracy)...")
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy", error_score="raise")
    print(f"CV scores: {scores} mean: {scores.mean():.3f}")
except Exception as e:
    print("⚠️ Cross-validation overgeslagen door fout:", e)

# Train eindmodel
model.fit(X_train, y_train)
print("✅ Model getraind.")

# ==========================================================
# 7. EVALUATIE
# ==========================================================
y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ==========================================================
# 8. OPSLAAN & VOORSPELLEN
# ==========================================================
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model opgeslagen als '{MODEL_FILE}'")

if len(df_unknown) > 0:
    X_missing = df_unknown[X.columns.intersection(df_unknown.columns)]
    preds = model.predict(X_missing)
    df_unknown["voorspeld_energielabel"] = preds

    df_result = pd.concat([df_known, df_unknown], ignore_index=True)
    df_result.to_excel("voorspelde_energielabels.xlsx", index=False)
    print("📊 Bestand met voorspellingen opgeslagen als 'voorspelde_energielabels.xlsx'")

print("✅ Klaar!")
