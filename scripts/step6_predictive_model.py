"""
STEP 6: Predictive Model
Can we predict whether a user converts, using test_group, total_ads, day, and hour?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                              precision_recall_curve, classification_report, confusion_matrix)
from xgboost import XGBClassifier

df = pd.read_csv("marketing_AB_cleaned.csv")

# =========================================================
# 1. Feature engineering
# =========================================================
df["is_ad"] = (df["test_group"] == "ad").astype(int)
df["log_total_ads"] = np.log1p(df["total_ads"])

# cyclical encoding for hour (captures that hour 23 is close to hour 0)
df["hour_sin"] = np.sin(2 * np.pi * df["most_ads_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["most_ads_hour"] / 24)

# one-hot encode day
day_dummies = pd.get_dummies(df["most_ads_day"], prefix="day", drop_first=True)
df = pd.concat([df, day_dummies], axis=1)

feature_cols = ["is_ad", "log_total_ads", "hour_sin", "hour_cos"] + list(day_dummies.columns)
X = df[feature_cols]
y = df["converted"].astype(int)

print("Features used:", feature_cols)
print("Class balance:\n", y.value_counts(normalize=True))

# =========================================================
# 2. Train/test split (stratified because of class imbalance)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# =========================================================
# 3. Logistic Regression (baseline, interpretable)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

y_pred_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
pr_auc_lr = average_precision_score(y_test, y_pred_proba_lr)

print(f"\n--- Logistic Regression ---")
print(f"ROC-AUC: {roc_auc_lr:.4f}")
print(f"PR-AUC (average precision): {pr_auc_lr:.4f}")

coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": log_reg.coef_[0]}).sort_values("coefficient", ascending=False)
print("\nLogistic Regression coefficients (standardized):")
print(coef_df)

# =========================================================
# 4. XGBoost (captures non-linearity like the saturation curve)
# =========================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss", random_state=42
)
xgb.fit(X_train, y_train)

y_pred_proba_xgb = xgb.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
pr_auc_xgb = average_precision_score(y_test, y_pred_proba_xgb)

print(f"\n--- XGBoost ---")
print(f"ROC-AUC: {roc_auc_xgb:.4f}")
print(f"PR-AUC (average precision): {pr_auc_xgb:.4f}")

feat_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb.feature_importances_
}).sort_values("importance", ascending=False)
print("\nXGBoost feature importances:")
print(feat_importance)

# =========================================================
# 5. ROC curve comparison plot
# =========================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_lr:.3f})")
ax.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_xgb:.3f})")
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("plot10_roc_comparison.png", dpi=120)
plt.close()

# =========================================================
# 6. Precision-Recall curve (more informative given 2.5% positive rate)
# =========================================================
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_pred_proba_xgb)
baseline = y_test.mean()

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(rec_lr, prec_lr, label=f"Logistic Regression (AP={pr_auc_lr:.3f})")
ax.plot(rec_xgb, prec_xgb, label=f"XGBoost (AP={pr_auc_xgb:.3f})")
ax.axhline(baseline, color="k", linestyle="--", label=f"Baseline (prevalence={baseline:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("plot11_pr_curve_comparison.png", dpi=120)
plt.close()

print("\nAll plots saved. Model objects and test data will be used for SHAP in the next cell.")

# Save for SHAP step
import joblib
joblib.dump(xgb, "xgb_model.pkl")
X_test.to_csv("X_test.csv", index=False)
