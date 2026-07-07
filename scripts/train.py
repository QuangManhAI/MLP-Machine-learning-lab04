import sys, os, json, time, logging, shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lab04.process import CombinedAttributesAdder
from lab04.model import MultiLayerPercepTron, LinearRegressionScratch, HistGradientBoostingRegressorScratch

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = PROJECT_ROOT / "log" / TIMESTAMP
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = PROJECT_ROOT / "reports" / "EDA"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

SKEWED_FEATS = ["total_rooms", "total_bedrooms", "population", "households"]


def load_data():
    df = pd.read_csv(PROJECT_ROOT / "data/raw/housing.csv")
    log.info("Loaded housing.csv — shape: %s", df.shape)
    return df


def preprocess(df, log_transform=False):
    df = df.copy()
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(df, df["income_cat"]):
        train_set = df.loc[train_idx]
        test_set = df.loc[test_idx]
    for s in (train_set, test_set):
        s.drop("income_cat", axis=1, inplace=True)

    housing = train_set.drop("median_house_value", axis=1)
    housing_labels = train_set["median_house_value"].copy()
    housing_test = test_set.drop("median_house_value", axis=1)
    housing_labels_test = test_set["median_house_value"].copy()

    if log_transform:
        skewed_cols = ["total_rooms", "total_bedrooms", "population", "households"]
        for col in skewed_cols:
            housing[col] = np.log1p(housing[col])
            housing_test[col] = np.log1p(housing_test[col])
        log.info("Log1p applied at raw level to %s", skewed_cols)

    num_attribs = housing.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
    ])

    X_train = full_pipeline.fit_transform(housing)
    X_test = full_pipeline.transform(housing_test)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(housing_labels.values.reshape(-1, 1))
    y_test_raw = housing_labels_test.values.reshape(-1, 1)

    log.info("X_train shape: %s, X_test shape: %s", X_train.shape, X_test.shape)
    log.info("Preprocessing complete (%s) — %d features", "log" if log_transform else "original", X_train.shape[1])

    return X_train, X_test, y_train, y_test_raw, scaler_y


def compute_metrics(y_true, y_pred):
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2), "MAPE": float(mape)}


def train_linear(X_train, y_train):
    t0 = time.time()
    model = LinearRegressionScratch(lr=0.05, epochs=200)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    return model, elapsed


def train_mlp(X_train, y_train):
    t0 = time.time()
    model = MultiLayerPercepTron(input_dim=X_train.shape[1], epochs=500, lr=0.1)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    return model, elapsed


def train_hgb(X_train, y_train):
    t0 = time.time()
    model = HistGradientBoostingRegressorScratch(
        n_estimators=50, learning_rate=0.1, max_depth=4, min_samples_leaf=20
    )
    model.fit(X_train, y_train.ravel())
    elapsed = time.time() - t0
    return model, elapsed


def train_and_eval(X_train, X_test, y_train, y_test_raw, scaler_y, tag=""):
    log.info("--- Training on %s features ---", tag if tag else "original")
    models = {}
    times = {}

    m, t = train_linear(X_train, y_train)
    models["Linear"] = m; times["Linear"] = t
    log.info("  Linear — %.2fs | final loss: %.6f", t, m.loss_history[-1])

    m, t = train_mlp(X_train, y_train)
    models["MLP"] = m; times["MLP"] = t
    log.info("  MLP — %.2fs | final loss: %.6f", t, m.loss_history[-1])

    m, t = train_hgb(X_train, y_train)
    models["HGB"] = m; times["HGB"] = t
    log.info("  HGB — %.2fs | final loss: %.6f", t, m.loss_history[-1])

    metrics = {}
    for name, model in models.items():
        y_pred_scaled = model.predict(X_test)
        if y_pred_scaled.ndim == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        metrics[name] = compute_metrics(y_test_raw, y_pred)

    return models, metrics, times


def plot_comparison_dual(metrics_orig, metrics_log, save_path):
    models = ["Linear", "MLP", "HGB"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {"Linear": "#3498db", "MLP": "#e74c3c", "HGB": "#e67e22"}

    for ax, metric_key, title, fmt in zip(
        axes,
        ["RMSE", "MAE", "R2"],
        ["RMSE (lower is better)", "MAE (lower is better)", "R² (higher is better)"],
        ["${:,.0f}", "${:,.0f}", "{:.3f}"],
    ):
        x = np.arange(len(models))
        w = 0.3
        orig_vals = [metrics_orig[m][metric_key] for m in models]
        log_vals = [metrics_log[m][metric_key] for m in models]

        bars1 = ax.bar(x - w / 2, orig_vals, w, label="Original", color=[colors[m] for m in models], alpha=0.5, edgecolor="white")
        bars2 = ax.bar(x + w / 2, log_vals, w, label="Log", color=[colors[m] for m in models], alpha=0.9, edgecolor="white")

        for bars, vals in [(bars1, orig_vals), (bars2, log_vals)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                        fmt.format(v), ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Dual comparison saved to %s", save_path)


def plot_loss_comparison(models_orig, models_log, save_path):
    names = ["Linear", "MLP", "HGB"]
    colors = {"Linear": "#3498db", "MLP": "#e74c3c", "HGB": "#e67e22"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, name in zip(axes, names):
        orig = models_orig[name].loss_history
        log_ = models_log[name].loss_history
        ax.plot(orig, color=colors[name], linewidth=1.5, alpha=0.6, label="Original")
        ax.plot(log_, color=colors[name], linewidth=1.5, linestyle="--", label="Log")
        ax.set_title(f"{name} — Loss Comparison")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Training Loss: Original vs Log-Transformed Features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info("Loss comparison saved to %s", save_path)


def main():
    log.info("=" * 60)
    log.info("Training run started at %s", TIMESTAMP)
    log.info("=" * 60)

    df = load_data()

    # Preprocess both versions from the same raw data
    X_train, X_test, y_train, y_test_raw, scaler_y = preprocess(df, log_transform=False)
    X_train_log, X_test_log, _, _, _ = preprocess(df, log_transform=True)

    # Train on original features
    models_orig, metrics_orig, times_orig = train_and_eval(
        X_train, X_test, y_train, y_test_raw, scaler_y, tag="original"
    )

    # Train on log-transformed features
    models_log, metrics_log, times_log = train_and_eval(
        X_train_log, X_test_log, y_train, y_test_raw, scaler_y, tag="log-transformed"
    )

    # Comparison table
    log.info("=" * 70)
    log.info("COMPARISON: Original vs Log-Transformed")
    log.info("=" * 70)
    for name in ["Linear", "MLP", "HGB"]:
        lo = metrics_orig[name]
        lm = metrics_log[name]
        dr2 = (lm["R2"] - lo["R2"]) * 100
        drmse = (lo["RMSE"] - lm["RMSE"]) / lo["RMSE"] * 100
        log.info(
            "  %-20s | Orig R²: %.4f  Log R²: %.4f  ΔR²: %+.2fpp  ΔRMSE: %+.2f%%",
            name, lo["R2"], lm["R2"], dr2, drmse,
        )

    # Build summary JSON
    summary = {
        "timestamp": TIMESTAMP,
        "dataset": "California Housing",
        "skewed_features": SKEWED_FEATS,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "models": {},
    }
    for name in ["Linear", "MLP", "HGB"]:
        summary["models"][name] = {
            "original": {"metrics": metrics_orig[name], "train_time_s": round(times_orig[name], 2)},
            "log": {"metrics": metrics_log[name], "train_time_s": round(times_log[name], 2)},
        }

    with open(LOG_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Metrics saved to %s", LOG_DIR / "metrics.json")

    # Save comparison plot
    plot_comparison_dual(metrics_orig, metrics_log, LOG_DIR / "comparison_dual.png")
    shutil.copy(LOG_DIR / "comparison_dual.png", REPORT_DIR / "comparison_dual.png")

    # Save loss comparison
    plot_loss_comparison(models_orig, models_log, LOG_DIR / "loss_comparison.png")
    shutil.copy(LOG_DIR / "loss_comparison.png", REPORT_DIR / "loss_comparison.png")

    # Also save original single-version plots
    plot_comparison_dual(metrics_orig, metrics_orig, LOG_DIR / "comparison_bar.png")
    shutil.copy(LOG_DIR / "comparison_bar.png", REPORT_DIR / "comparison_bar.png")

    log.info("All outputs saved to %s", LOG_DIR)
    print(f"\nResults saved to: {LOG_DIR}")
    print("Comparison: Original vs Log-Transformed features (skewed count features)")


if __name__ == "__main__":
    main()
