"""
STEP 5: Segment Analysis
- Dose-response: does more ad exposure mean more conversion? Where's the saturation point?
- Does the ad-vs-psa effect hold consistently across days and hours (Simpson's Paradox check)?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
df = pd.read_csv("marketing_AB_cleaned.csv")
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# =========================================================
# 1. Dose-response curve: total_ads bucket vs conversion rate
# =========================================================
bins = [0, 5, 10, 20, 40, 80, 160, 300, 2100]
labels = ["1-5", "6-10", "11-20", "21-40", "41-80", "81-160", "161-300", "300+"]
df["ads_bucket"] = pd.cut(df["total_ads"], bins=bins, labels=labels)

bucket_conv = df.groupby("ads_bucket", observed=True)["converted"].agg(["mean", "count"])
bucket_conv["mean"] = bucket_conv["mean"] * 100
print("=== Conversion rate by total_ads bucket (all users) ===")
print(bucket_conv)

fig, ax = plt.subplots(figsize=(9, 4.5))
sns.barplot(x=bucket_conv.index, y=bucket_conv["mean"], ax=ax, color="#4C72B0")
ax.set_ylabel("Conversion Rate (%)")
ax.set_xlabel("Total ads seen (bucket)")
ax.set_title("Dose-Response: Conversion Rate vs Ad Exposure")
for i, v in enumerate(bucket_conv["mean"]):
    ax.text(i, v + 0.1, f"{v:.2f}%", ha="center")
plt.tight_layout()
plt.savefig("plot7_dose_response.png", dpi=120)
plt.close()

# =========================================================
# 2. Conversion by day, split by test group (interaction check)
# =========================================================
day_group = df.groupby(["most_ads_day", "test_group"])["converted"].mean().unstack() * 100
day_group = day_group.reindex(day_order)
print("\n=== Conversion rate by day, split by group ===")
print(day_group.round(3))

fig, ax = plt.subplots(figsize=(9, 4.5))
day_group.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate by Day, Ad vs PSA")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot8_day_by_group.png", dpi=120)
plt.close()

# Formal check: chi-square test PER DAY, and flag any reversal
print("\n=== Per-day significance check (ad vs psa) ===")
day_results = []
for day in day_order:
    sub = df[df["most_ads_day"] == day]
    ct = pd.crosstab(sub["test_group"], sub["converted"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    ad_rate = sub[sub["test_group"]=="ad"]["converted"].mean() * 100
    psa_rate = sub[sub["test_group"]=="psa"]["converted"].mean() * 100
    reversal = ad_rate < psa_rate
    day_results.append((day, ad_rate, psa_rate, p, reversal))
    flag = "  <-- REVERSAL" if reversal else ""
    sig = "significant" if p < 0.05 else "not significant"
    print(f"{day:10s} | ad={ad_rate:.3f}% psa={psa_rate:.3f}% | p={p:.4f} ({sig}){flag}")

# =========================================================
# 3. Conversion by hour, split by test group (interaction check)
# =========================================================
hour_group = df.groupby(["most_ads_hour", "test_group"])["converted"].mean().unstack() * 100
print("\n=== Conversion rate by hour, split by group (head) ===")
print(hour_group.round(3).head(24))

fig, ax = plt.subplots(figsize=(12, 4.5))
hour_group.plot(ax=ax, marker="o", color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Conversion Rate (%)")
ax.set_xlabel("Hour of day")
ax.set_title("Conversion Rate by Hour, Ad vs PSA")
ax.set_xticks(range(0,24))
plt.tight_layout()
plt.savefig("plot9_hour_by_group.png", dpi=120)
plt.close()

# Formal check: chi-square test PER HOUR, and flag any reversal
print("\n=== Per-hour significance check (ad vs psa) ===")
hour_results = []
for hour in range(24):
    sub = df[df["most_ads_hour"] == hour]
    ct = pd.crosstab(sub["test_group"], sub["converted"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        continue
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    ad_rate = sub[sub["test_group"]=="ad"]["converted"].mean() * 100
    psa_rate = sub[sub["test_group"]=="psa"]["converted"].mean() * 100
    reversal = ad_rate < psa_rate
    hour_results.append((hour, ad_rate, psa_rate, p, reversal))
    flag = "  <-- REVERSAL" if reversal else ""
    sig = "significant" if p < 0.05 else "not significant"
    print(f"hour {hour:2d} | ad={ad_rate:.3f}% psa={psa_rate:.3f}% | p={p:.4f} ({sig}){flag}")

# =========================================================
# 4. Simpson's Paradox summary
# =========================================================
day_reversals = [d for d in day_results if d[4]]
hour_reversals = [h for h in hour_results if h[4]]
print(f"\n=== Simpson's Paradox check summary ===")
print(f"Days where psa outperformed ad: {len(day_reversals)} / {len(day_results)}")
print(f"Hours where psa outperformed ad: {len(hour_reversals)} / {len(hour_results)}")
if day_reversals:
    print("Reversal days:", [d[0] for d in day_reversals])
if hour_reversals:
    print("Reversal hours:", [h[0] for h in hour_reversals])

print("\nAll plots saved.")
