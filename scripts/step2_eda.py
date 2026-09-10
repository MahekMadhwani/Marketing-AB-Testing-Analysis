"""
STEP 2: Exploratory Data Analysis
Marketing A/B Testing Dataset
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
df = pd.read_csv("marketing_AB_cleaned.csv")

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# 1. Conversion rate by test group
conv_by_group = df.groupby("test_group")["converted"].mean() * 100

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(conv_by_group.index, conv_by_group.values, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate: Ad vs PSA")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}%", ha="center")
plt.tight_layout()
plt.savefig("plot1_conversion_by_group.png", dpi=120)
plt.close()

# 2. total_ads distribution + outlier check
print("=== total_ads outlier check ===")
print(df["total_ads"].describe(percentiles=[.5, .9, .95, .99, .999]))
print("\nTop 10 highest total_ads values:")
print(df.sort_values("total_ads", ascending=False)[["user_id", "test_group", "converted", "total_ads"]].head(10))

p99 = df["total_ads"].quantile(0.99)
print(f"\n99th percentile of total_ads: {p99}")
print(f"Number of users above 99th percentile: {(df['total_ads'] > p99).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["total_ads"], bins=60, ax=axes[0])
axes[0].set_title("Total Ads Seen - Distribution (raw)")
axes[0].set_xlabel("Total ads")

sns.histplot(np.log1p(df["total_ads"]), bins=60, ax=axes[1], color="#55A868")
axes[1].set_title("Total Ads Seen - Distribution (log scale)")
axes[1].set_xlabel("log(1 + total ads)")
plt.tight_layout()
plt.savefig("plot2_total_ads_distribution.png", dpi=120)
plt.close()

# 3. Conversion rate by day of week
conv_by_day = df.groupby("most_ads_day")["converted"].mean().reindex(day_order) * 100

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=conv_by_day.index, y=conv_by_day.values, ax=ax, color="#4C72B0")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate by Day of Week (most_ads_day)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("plot3_conversion_by_day.png", dpi=120)
plt.close()

# 4. Conversion rate by hour of day
conv_by_hour = df.groupby("most_ads_hour")["converted"].mean() * 100

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(x=conv_by_hour.index, y=conv_by_hour.values, marker="o", ax=ax)
ax.set_xlabel("Hour of day (0-23)")
ax.set_ylabel("Conversion Rate (%)")
ax.set_title("Conversion Rate by Hour of Day (most_ads_hour)")
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig("plot4_conversion_by_hour.png", dpi=120)
plt.close()

# 5. Heatmap: Day x Hour conversion rate
pivot = df.pivot_table(index="most_ads_day", columns="most_ads_hour", values="converted", aggfunc="mean")
pivot = pivot.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot * 100, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Conversion Rate (%)"})
ax.set_title("Conversion Rate Heatmap: Day x Hour")
plt.tight_layout()
plt.savefig("plot5_heatmap_day_hour.png", dpi=120)
plt.close()

# 6. User volume by day/hour (context for the heatmap above)
pivot_counts = df.pivot_table(index="most_ads_day", columns="most_ads_hour", values="user_id", aggfunc="count")
pivot_counts = pivot_counts.reindex(day_order)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot_counts, cmap="Blues", ax=ax, cbar_kws={"label": "Number of Users"})
ax.set_title("User Volume Heatmap: Day x Hour")
plt.tight_layout()
plt.savefig("plot6_heatmap_volume.png", dpi=120)
plt.close()

# Print summary numbers
print("\n=== Conversion rate by group ===")
print(conv_by_group.round(3))

print("\n=== Conversion rate by day ===")
print(conv_by_day.round(3))

print("\n=== Conversion rate by hour ===")
print(conv_by_hour.round(3))

print("\n=== Best day ===", conv_by_day.idxmax(), f"({conv_by_day.max():.3f}%)")
print("=== Worst day ===", conv_by_day.idxmin(), f"({conv_by_day.min():.3f}%)")
print("=== Best hour ===", conv_by_hour.idxmax(), f"({conv_by_hour.max():.3f}%)")
print("=== Worst hour ===", conv_by_hour.idxmin(), f"({conv_by_hour.min():.3f}%)")

print("\nAll 6 plots saved.")
