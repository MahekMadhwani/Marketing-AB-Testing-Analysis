"""
STEP 6b: SHAP Interpretability
Explain the XGBoost model's predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

xgb = joblib.load("xgb_model.pkl")
X_test = pd.read_csv("X_test.csv")

# Sample for speed (SHAP on 100k+ rows is slow; 5000 is plenty for a stable summary)
X_sample = X_test.sample(5000, random_state=42)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

# =========================================================
# 1. Global feature importance (SHAP summary bar)
# =========================================================
fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("plot12_shap_importance.png", dpi=120, bbox_inches="tight")
plt.close()

# =========================================================
# 2. SHAP summary (beeswarm) - shows direction of effect
# =========================================================
fig = plt.figure(figsize=(8, 5))
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("plot13_shap_beeswarm.png", dpi=120, bbox_inches="tight")
plt.close()

# =========================================================
# 3. SHAP dependence plot for log_total_ads (confirm the saturation curve)
# =========================================================
fig = plt.figure(figsize=(7, 5))
shap.dependence_plot("log_total_ads", shap_values, X_sample, interaction_index="is_ad", show=False)
plt.tight_layout()
plt.savefig("plot14_shap_dependence_total_ads.png", dpi=120, bbox_inches="tight")
plt.close()

print("SHAP plots saved.")
print("\nMean |SHAP value| per feature (global importance):")
mean_abs_shap = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)
print(mean_abs_shap)
