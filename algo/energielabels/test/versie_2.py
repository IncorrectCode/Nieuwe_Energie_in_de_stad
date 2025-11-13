import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)

# ==========================================================
# CONFIG
# ==========================================================
RANDOM_STATE = 42
INPUT_FILE = "databim ruwe data.xlsx"
MODEL_FILE = "energielabel_model.pkl"
MANUAL_LABEL_COL = "pand_energieklasse"  # De energielabel kolom

# ==========================================================
# 1. DATA INLEZEN
# ==========================================================
print("📥 Excel inlezen...")
xls = pd.ExcelFile(INPUT_FILE, engine="openpyxl")

dfs = []
for sheet in xls.sheet_names:
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet, engine="openpyxl")
    print(f"  Sheet '{sheet}': {df.shape}")
    df["_sheet"] = sheet
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True, sort=False)
print(f"✅ Totale rijen: {len(data):,}")
print(f"✅ Totale kolommen: {len(data.columns)}\n")

# ==========================================================
# 2. ENERGIELABEL KOLOM VINDEN
# ==========================================================
print("="*80)
print("🔍 ZOEK ENERGIELABEL KOLOM")
print("="*80)

col_label = None

if MANUAL_LABEL_COL in data.columns:
    col_label = MANUAL_LABEL_COL
    print(f"✅ Handmatig geconfigureerde kolom gevonden: '{col_label}'")
else:
    label_keywords = ['energieklasse', 'energielabel', 'energie_klasse', 'energie_label']
    for col in data.columns:
        if any(kw in col.lower() for kw in label_keywords):
            col_label = col
            print(f"✅ Automatisch gevonden: '{col_label}'")
            break

if not col_label:
    raise ValueError("❌ Geen energielabel kolom gevonden.")

# Inspectie
non_null = data[col_label].notna().sum()
print(f"\nKolom '{col_label}' bevat {non_null:,} niet-lege waarden.")
print(f"Voorbeelden unieke waarden: {data[col_label].dropna().unique()[:10]}")

# ==========================================================
# 3. LABEL NORMALISATIE (TEKST OF NUMERIEK)
# ==========================================================
df_known = data[data[col_label].notna()].copy()
df_unknown = data[data[col_label].isna()].copy()

if pd.api.types.is_numeric_dtype(df_known[col_label]):
    print("\n📊 Numeriek label gevonden – converteren naar A–G klassen...")
    
    percentiles = np.percentile(df_known[col_label].dropna(), [0, 15, 30, 45, 60, 75, 90, 100])
    percentiles = np.unique(percentiles)
    labels = ["G", "F", "E", "D", "C", "B", "A"]

    if len(percentiles) - 1 < len(labels):
        print(f"⚠️ Minder unieke waarden gevonden ({len(percentiles)-1} bins) — labels worden aangepast.")
        labels = labels[-(len(percentiles)-1):]

    df_known["label_class"] = pd.cut(
        df_known[col_label],
        bins=percentiles,
        labels=labels,
        include_lowest=True,
        duplicates="drop"
    )
    y = df_known["label_class"]

else:
    print("\n🔤 Tekstuele labels gevonden – normaliseren naar A–G...")
    df_known[col_label] = df_known[col_label].astype(str).str.upper().str.strip()
    df_known[col_label] = df_known[col_label].str.replace(r"A\++", "A", regex=True)
    valid_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df_known = df_known[df_known[col_label].isin(valid_labels)].copy()
    y = df_known[col_label]

print(f"✅ Label distributie:\n{y.value_counts()}")

# ==========================================================
# 4. FEATURE SELECTIE (vereenvoudigd)
# ==========================================================
possible_features = [
    "pand_bouwjaar", "pand_gebruiksoppervlakte_thermische_zone",
    "pand_gebouwtype", "pand_aantal_kamers", "pand_postcode",
    "pand_daktype", "pand_verwarmingstype", "pand_isolatie", "pand_glas"
]

feature_cols = [c for c in possible_features if c in data.columns]
print(f"\n✅ Geselecteerde features: {feature_cols}")

X = df_known[feature_cols].copy()

# ==========================================================
# 5. FEATURE ENGINEERING
# ==========================================================
bouwjaar_col = next((c for c in feature_cols if "bouwjaar" in c.lower()), None)
oppervlakte_col = next((c for c in feature_cols if "oppervlakte" in c.lower()), None)

if bouwjaar_col and bouwjaar_col in X.columns:
    X[bouwjaar_col] = pd.to_numeric(X[bouwjaar_col], errors="coerce")
    X.loc[X[bouwjaar_col] < 1500, bouwjaar_col] = np.nan
    X.loc[X[bouwjaar_col] > 2025, bouwjaar_col] = np.nan
    X["gebouw_leeftijd"] = 2025 - X[bouwjaar_col]

if oppervlakte_col and oppervlakte_col in X.columns:
    X[oppervlakte_col] = pd.to_numeric(X[oppervlakte_col], errors="coerce")
    X.loc[X[oppervlakte_col] < 10, oppervlakte_col] = np.nan
    q99 = X[oppervlakte_col].quantile(0.99)
    X.loc[X[oppervlakte_col] > q99 * 2, oppervlakte_col] = np.nan

# ==========================================================
# 6. MODEL PIPELINE
# ==========================================================
num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
cat_features = [c for c in X.columns if c not in num_features]

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="onbekend")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_features),
    ("cat", cat_transformer, cat_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        min_samples_split=15,
        min_samples_leaf=8,
        subsample=0.8,
        max_features="sqrt",
        random_state=RANDOM_STATE
    ))
])

# ==========================================================
# 7. TRAINING
# ==========================================================
print("\n🏋️  MODEL TRAINING...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

model.fit(X_train, y_train)
print("✅ Model getraind!")

# ==========================================================
# 8. EVALUATIE
# ==========================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n📊 Test Accuracy: {acc:.2%}")
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================================
# 9. MODEL OPSLAAN
# ==========================================================
model_data = {
    "model": model,
    "features": feature_cols,
    "label_column": col_label
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model_data, f)
print(f"💾 Model opgeslagen als '{MODEL_FILE}'")

# ==========================================================
# 10. CONCLUSIE
# ==========================================================
print("\n✅ KLAAR!")
print(f"Test accuracy: {acc:.1%}")
if acc < 0.55:
    print("⚠️ Model presteert matig – controleer labelverdeling of features.")
elif acc < 0.70:
    print("⚠️ Redelijk model – uitbreiden met meer kenmerken kan helpen.")
else:
    print("✅ Goed model – nauwkeurige voorspellingen verwacht.")
