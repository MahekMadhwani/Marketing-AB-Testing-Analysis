"""
STEP 1: Load & Clean
Marketing A/B Testing Dataset
"""

import pandas as pd

# ---- Load ----
df = pd.read_csv("marketing_AB.csv")   # replace with your actual filename once uploaded

# Kaggle's version usually ships with an extra unnamed index column - drop it if present
unnamed_cols = [c for c in df.columns if "Unnamed" in c]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

# Standardize column names (lowercase, underscores) so code is easier to write later
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)

# ---- Basic sanity checks ----
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ---- Check group balance (this matters a LOT for this dataset) ----
print("\nTest group distribution:")
print(df["test_group"].value_counts())
print(df["test_group"].value_counts(normalize=True).round(4) * 100, "%")

# ---- Check target variable ----
print("\nConverted distribution:")
print(df["converted"].value_counts())
print("Overall conversion rate:", round(df["converted"].mean() * 100, 3), "%")

# ---- Conversion rate by group (the headline number) ----
conv_by_group = df.groupby("test_group")["converted"].mean().sort_values(ascending=False)
print("\nConversion rate by group:\n", (conv_by_group * 100).round(3))

# ---- Quick look at total_ads distribution ----
print("\nTotal ads seen - summary stats:\n", df["total_ads"].describe())

# ---- Save cleaned version ----
df.to_csv("marketing_AB_cleaned.csv", index=False)
print("\nSaved cleaned file as marketing_AB_cleaned.csv")
