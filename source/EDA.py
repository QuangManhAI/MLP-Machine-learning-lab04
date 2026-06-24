import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pathlib import Path

SAVE_DIR = Path("reports/eda")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name, dpi=150):
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"{name}.png", dpi=dpi)

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)


df = pd.read_csv("data/raw/housing.csv")

print("=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print(df.head())
print("\nInfo:")
print(df.info())
print("\nOcean Proximity value counts:")
print(df["ocean_proximity"].value_counts())


print("\n" + "=" * 60)
print("2. STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("3. MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"Missing": missing, "Percent (%)": missing_pct}))

plt.figure(figsize=(8, 4))
missing[missing > 0].plot(kind="bar", color="tomato", edgecolor="white")
plt.title("Missing Values per Column")
plt.ylabel("Count")
save_fig("eda_01_missing_values")
plt.show()

df.hist(bins=50, figsize=(12, 8))
plt.suptitle("Feature Distributions", fontsize=14)
save_fig("eda_02_feature_distributions")
plt.show()


plt.figure(figsize=(8, 5))
plt.hist(df["median_house_value"], bins=50, color="steelblue", edgecolor="white")
plt.axvline(df["median_house_value"].median(), color="red",
            linestyle="--", label=f'Median: ${df["median_house_value"].median():,.0f}')
plt.title("Distribution of Median House Value")
plt.xlabel("House Value ($)")
plt.ylabel("Count")
plt.legend()
save_fig("eda_03_target_distribution")
plt.show()



# 6a. Basic
df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
        alpha=0.2, figsize=(10, 7))
plt.title("Geographic Distribution (Basic)")
save_fig("eda_04_geo_basic")
plt.show()

# 6b. Nâng cao – size=population, color=price (theo notebook mẫu)
df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
        s=df["population"] / 100, label="Population",
        c="median_house_value", cmap="jet", colorbar=True,
        legend=True, sharex=False, figsize=(10, 7))
plt.title("Geographic Distribution of House Prices – California")
save_fig("eda_05_geo_price_population")
plt.show()

corr = df.corr(numeric_only=True)


print("\n" + "=" * 60)
print("7. CORRELATION WITH TARGET")
print("=" * 60)
print(corr["median_house_value"].sort_values(ascending=False))


plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Heatmap (numeric features)")
save_fig("eda_06_correlation_heatmap")
plt.show()

target_corr = corr["median_house_value"].drop("median_house_value").sort_values()
colors = ["tomato" if v < 0 else "steelblue" for v in target_corr]
plt.figure(figsize=(8, 5))
target_corr.plot(kind="barh", color=colors, edgecolor="white")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Median House Value")
plt.xlabel("Pearson Correlation")
save_fig("eda_07_target_correlation_bar")
plt.show()


attributes = ["median_house_value", "median_income",
              "total_rooms", "housing_median_age"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.suptitle("Scatter Matrix – Top Features", fontsize=13)
save_fig("eda_08_scatter_matrix")
plt.show()


df.plot(kind="scatter", x="median_income", y="median_house_value",
        alpha=0.1, grid=True, figsize=(8, 6))
plt.title("Median Income vs Median House Value")
save_fig("eda_09_income_vs_price")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df["ocean_proximity"].value_counts().plot(
    kind="bar", ax=axes[0], color="steelblue", edgecolor="white", rot=15)
axes[0].set_title("Count by Ocean Proximity")
axes[0].set_ylabel("Count")

df.groupby("ocean_proximity")["median_house_value"].mean().sort_values().plot(
    kind="barh", ax=axes[1], color="coral", edgecolor="white")
axes[1].set_title("Avg House Value by Ocean Proximity")
axes[1].set_xlabel("Avg Median House Value ($)")

save_fig("eda_10_ocean_proximity_bar")
plt.show()

# Boxplot
plt.figure(figsize=(10, 5))
order = df.groupby("ocean_proximity")["median_house_value"].median().sort_values().index
sns.boxplot(x="ocean_proximity", y="median_house_value",
            data=df, order=order, palette="Set2")
plt.title("House Price Distribution by Ocean Proximity")
plt.xlabel("Ocean Proximity")
plt.ylabel("Median House Value ($)")
plt.xticks(rotation=15)
save_fig("eda_11_ocean_proximity_boxplot")
plt.show()


df["rooms_per_house"]      = df["total_rooms"]    / df["households"]
df["bedrooms_ratio"]       = df["total_bedrooms"] / df["total_rooms"]
df["people_per_house"]     = df["population"]     / df["households"]

corr2 = df.corr(numeric_only=True)
new_corr = corr2["median_house_value"][
    ["rooms_per_house", "bedrooms_ratio", "people_per_house"]
].sort_values(ascending=False)

print("\n" + "=" * 60)
print("11. NEW FEATURE CORRELATION WITH TARGET")
print("=" * 60)
print(new_corr)


full_corr = corr2["median_house_value"].drop("median_house_value").sort_values()
colors2 = ["tomato" if v < 0 else "steelblue" for v in full_corr]
plt.figure(figsize=(9, 6))
full_corr.plot(kind="barh", color=colors2, edgecolor="white")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Feature Correlation with Target (incl. engineered features)")
plt.xlabel("Pearson Correlation")
save_fig("eda_12_correlation_engineered")
plt.show()

print("\n" + "=" * 60)
print(f"EDA COMPLETE – 12 charts saved to '{SAVE_DIR}/'")
print("=" * 60)