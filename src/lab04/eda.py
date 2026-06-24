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

# Configure default plotting styles
plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)


def plot_missing_values(df):
    """Plots missing values bar chart."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    plt.figure(figsize=(8, 4))
    missing[missing > 0].plot(kind="bar", color="tomato", edgecolor="white")
    plt.title("Missing Values per Column")
    plt.ylabel("Count")
    save_fig("eda_01_missing_values")
    plt.show()
    return pd.DataFrame({"Missing": missing, "Percent (%)": missing_pct})


def plot_feature_distributions(df):
    """Plots histogram distributions of all numerical features."""
    df.hist(bins=50, figsize=(12, 8))
    plt.suptitle("Feature Distributions", fontsize=14)
    save_fig("eda_02_feature_distributions")
    plt.show()


def plot_target_distribution(df):
    """Plots distribution of target median_house_value."""
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


def plot_geographic_distribution(df, advanced=False):
    """Plots geographical distribution based on longitude/latitude."""
    if not advanced:
        df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                alpha=0.2, figsize=(10, 7))
        plt.title("Geographic Distribution (Basic)")
        save_fig("eda_04_geo_basic")
    else:
        df.plot(kind="scatter", x="longitude", y="latitude", grid=True,
                s=df["population"] / 100, label="Population",
                c="median_house_value", cmap="jet", colorbar=True,
                legend=True, sharex=False, figsize=(10, 7))
        plt.title("Geographic Distribution of House Prices – California")
        save_fig("eda_05_geo_price_population")
    plt.show()


def plot_correlation_matrix(df):
    """Plots correlation matrix list and heatmap."""
    corr = df.corr(numeric_only=True)
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm",
                fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap (numeric features)")
    save_fig("eda_06_correlation_heatmap")
    plt.show()
    
    # Bar plot
    target_corr = corr["median_house_value"].drop("median_house_value").sort_values()
    colors = ["tomato" if v < 0 else "steelblue" for v in target_corr]
    plt.figure(figsize=(8, 5))
    target_corr.plot(kind="barh", color=colors, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Correlation with Median House Value")
    plt.xlabel("Pearson Correlation")
    save_fig("eda_07_target_correlation_bar")
    plt.show()
    
    return corr["median_house_value"].sort_values(ascending=False)


def plot_scatter_matrix(df):
    """Plots scatter matrix of top features."""
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(df[attributes], figsize=(12, 8))
    plt.suptitle("Scatter Matrix – Top Features", fontsize=13)
    save_fig("eda_08_scatter_matrix")
    plt.show()


def plot_income_vs_price(df):
    """Plots income vs house price scatter plot."""
    df.plot(kind="scatter", x="median_income", y="median_house_value",
            alpha=0.1, grid=True, figsize=(8, 6))
    plt.title("Median Income vs Median House Value")
    save_fig("eda_09_income_vs_price")
    plt.show()


def plot_ocean_proximity_analysis(df):
    """Plots bar & boxplots for ocean proximity analysis."""
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


def plot_engineered_features_correlation(df):
    """Tries adding new features and plots their target correlations."""
    df_new = df.copy()
    df_new["rooms_per_house"] = df_new["total_rooms"] / df_new["households"]
    df_new["bedrooms_ratio"] = df_new["total_bedrooms"] / df_new["total_rooms"]
    df_new["people_per_house"] = df_new["population"] / df_new["households"]
    
    corr2 = df_new.corr(numeric_only=True)
    
    # Bar plot for all including new
    full_corr = corr2["median_house_value"].drop("median_house_value").sort_values()
    colors2 = ["tomato" if v < 0 else "steelblue" for v in full_corr]
    plt.figure(figsize=(9, 6))
    full_corr.plot(kind="barh", color=colors2, edgecolor="white")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Feature Correlation with Target (incl. engineered features)")
    plt.xlabel("Pearson Correlation")
    save_fig("eda_12_correlation_engineered")
    plt.show()
    
    return corr2["median_house_value"][
        ["rooms_per_house", "bedrooms_ratio", "people_per_house"]
    ].sort_values(ascending=False)
