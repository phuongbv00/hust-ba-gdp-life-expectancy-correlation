"""
Life expectancy analysis pipeline
Sources: Kaggle - Life Expectancy (WHO) dataset

This script implements the four analysis sections required:
III.1. Correlation between GDP and life expectancy by country
III.2. Life expectancy & GDP for top/bottom countries
III.3. Potential factors affecting low-life-expectancy countries
III.4. Global trends of life expectancy and GDP

Additionally, it trains and compares four prediction models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

Outputs (saved under ./output): cleaned/normalized datasets, reports (CSV/JSON) and figures (PNG).

Usage: place the Kaggle CSV in data/Life Expectancy Data.csv and run:
    python life_expectancy_analysis.py

"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------- Configuration & Artifacts --------------------
OUTPUT_DIR = Path("output")
DATA_PATH = Path("data/Life Expectancy Data.csv")


@dataclass
class PipelineArtifacts:
    raw_path: Path
    cleaned_path: Path
    normalized_path: Path
    reports_dir: Path
    figures_dir: Path
    models_dir: Path


def _ensure_output_dirs() -> PipelineArtifacts:
    OUTPUT_DIR.mkdir(exist_ok=True)
    reports_dir = OUTPUT_DIR / "reports"
    figures_dir = OUTPUT_DIR / "figures"
    models_dir = OUTPUT_DIR / "models"
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    return PipelineArtifacts(
        raw_path=DATA_PATH,
        cleaned_path=OUTPUT_DIR / "cleaned_dataset.csv",
        normalized_path=OUTPUT_DIR / "normalized_dataset.csv",
        reports_dir=reports_dir,
        figures_dir=figures_dir,
        models_dir=models_dir,
    )


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# -------------------- Preprocessing / Cleaning (adapted from your pipeline) --------------------
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("/", "_", regex=False)
    )
    df = df.copy()
    df.columns = cols
    rename_map = {
        "life_expectancy": "life_expectancy",
        "life_expectancy_": "life_expectancy",
        "measles_": "measles",
        "under_five_deaths_": "under_five_deaths",
        "hiv_aids": "hiv_aids",
        "percentage_expenditure": "percentage_expenditure",
    }
    df = df.rename(columns=rename_map)
    return df


def _clean_data(art: PipelineArtifacts) -> pd.DataFrame:
    df_raw = pd.read_csv(art.raw_path)
    df = _standardize_columns(df_raw)

    expected_cols = set(df.columns.tolist())

    # Coerce numerics
    numeric_cols = df.select_dtypes(exclude=[object]).columns.tolist()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trim string cols
    for c in [c for c in ["country", "status"] if c in df.columns]:
        df[c] = df[c].astype(str).str.strip()

    before = len(df)
    df = df.drop_duplicates()

    # Remove duplicates by (country, year) keeping most complete
    if {"country", "year"}.issubset(df.columns):
        df = (
            df.sort_values(["country", "year"])
            .assign(missing_count=lambda x: x.isna().sum(axis=1))
            .sort_values(["country", "year", "missing_count"])
            .drop_duplicates(["country", "year"], keep="first")
            .drop(columns=["missing_count"])
        )

    # Keep rows with essential targets
    if "life_expectancy" in df.columns and "gdp" in df.columns:
        df = df[df["life_expectancy"].notna() & df["gdp"].notna()]

    # Impute some important columns by country median then global median
    key_impute = [
        "alcohol", "bmi", "polio", "hepatitis_b", "diphtheria",
        "total_expenditure", "income_composition_of_resources", "schooling",
        "adult_mortality",
    ]
    if "country" in df.columns:
        for col in key_impute:
            if col in df.columns:
                df[col] = df.groupby("country")[col].transform(lambda s: s.fillna(s.median()))
    for col in key_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Clip extreme outliers
    for col in [c for c in ["gdp", "measles", "percentage_expenditure"] if c in df.columns]:
        upper = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper, upper, df[col])

    df.to_csv(art.cleaned_path, index=False)

    report = {
        "rows_before": int(before),
        "rows_after": int(len(df)),
        "columns": list(df.columns),
        "missing_after": df.isna().sum().sort_values(ascending=False).to_dict(),
    }
    _save_json(report, art.reports_dir / "1_cleaning_report.json")
    return df


# -------------------- Analysis sections (III.1 - III.4) --------------------

def analyze_gdp_life_corr(df: pd.DataFrame, art: PipelineArtifacts) -> pd.DataFrame:
    """III.1 Correlation between GDP and life expectancy (global & per-country)
    Returns per-country correlation table and saves figures/reports.
    """
    results = []
    if not {"country", "gdp", "life_expectancy"}.issubset(df.columns):
        return pd.DataFrame(results)

    # Global Pearson & Spearman (use log gdp for stability)
    df_valid = df.replace({"gdp": {0: np.nan}}).dropna(subset=["gdp", "life_expectancy"])
    log_gdp = np.log10(df_valid["gdp"]).replace([-np.inf, np.inf], np.nan).dropna()
    common_idx = log_gdp.index.intersection(df_valid.index)
    global_pearson = pearsonr(log_gdp.loc[common_idx], df_valid.loc[common_idx, "life_expectancy"])
    global_spearman = spearmanr(df_valid.loc[common_idx, "gdp"], df_valid.loc[common_idx, "life_expectancy"],
                                nan_policy='omit')

    # Per-country correlations
    for country, g in df.groupby("country"):
        g = g.dropna(subset=["gdp", "life_expectancy"])
        if len(g) >= 3:
            x = np.log10(g["gdp"].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).dropna()
            y = g.loc[x.index, "life_expectancy"]
            if len(x) >= 3:
                try:
                    r_p, p_p = pearsonr(x, y)
                    r_s, p_s = spearmanr(g.loc[x.index, "gdp"], y, nan_policy='omit')
                except Exception:
                    r_p, p_p, r_s, p_s = np.nan, np.nan, np.nan, np.nan
                results.append({
                    "country": country,
                    "n": int(len(g)),
                    "pearson_r_log10gdp_life": float(r_p) if pd.notna(r_p) else None,
                    "pearson_pvalue": float(p_p) if pd.notna(p_p) else None,
                    "spearman_r_gdp_life": float(r_s) if pd.notna(r_s) else None,
                    "spearman_pvalue": float(p_s) if pd.notna(p_s) else None,
                })

    corr_df = pd.DataFrame(results)
    corr_df.to_csv(art.reports_dir / "III1_corr_by_country.csv", index=False)

    # Global summary
    global_summary = {
        "global_pearson_log10gdp_life": float(global_pearson[0]),
        "global_pearson_pvalue": float(global_pearson[1]),
        "global_spearman_gdp_life": float(global_spearman.correlation),
        "global_spearman_pvalue": float(global_spearman.pvalue),
    }
    _save_json(global_summary, art.reports_dir / "III1_global_corr_summary.json")

    # Visual: scatter (log gdp vs life expectancy)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_valid, x=np.log10(df_valid["gdp"]), y="life_expectancy", hue=df_valid.get("status"))
    plt.xlabel("log10(GDP)")
    plt.title("Global: log10(GDP) vs Life Expectancy")
    plt.tight_layout()
    plt.savefig(art.figures_dir / "III1_scatter_loggdp_life.png", dpi=150)
    plt.close()

    return corr_df


def analyze_top_bottom(df: pd.DataFrame, art: PipelineArtifacts, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """III.2 Analyze top and bottom countries by average life expectancy"""
    if not {"country", "life_expectancy"}.issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    agg = df.groupby("country", as_index=False).agg(
        avg_life_expectancy=("life_expectancy", "mean"),
        avg_gdp=("gdp", "mean") if "gdp" in df.columns else ("life_expectancy", "mean")
    ).sort_values("avg_life_expectancy", ascending=False)

    top = agg.head(top_n)
    bottom = agg.tail(top_n)

    top.to_csv(art.reports_dir / "III2_top_countries.csv", index=False)
    bottom.to_csv(art.reports_dir / "III2_bottom_countries.csv", index=False)

    # Visuals
    if not top.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=top.sort_values("avg_life_expectancy"), x="avg_life_expectancy", y="country")
        plt.title(f"Top {top_n} Countries by Average Life Expectancy")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_top_countries.png", dpi=150)
        plt.close()

    if not bottom.empty:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=bottom.sort_values("avg_life_expectancy"), x="avg_life_expectancy", y="country")
        plt.title(f"Bottom {top_n} Countries by Average Life Expectancy")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_bottom_countries.png", dpi=150)
        plt.close()

    # Scatter GDP vs life for top/bottom
    for name, subset in [("top", top), ("bottom", bottom)]:
        countries = subset["country"].tolist()
        subdf = df[df["country"].isin(countries)]
        if not subdf.empty and "gdp" in subdf.columns:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(data=subdf, x="gdp", y="life_expectancy", hue="country")
            plt.xscale("log")
            plt.title(f"GDP vs Life Expectancy: {name} countries")
            plt.tight_layout()
            plt.savefig(art.figures_dir / f"III2_scatter_gdp_life_{name}.png", dpi=150)
            plt.close()

    return top, bottom


def analyze_low_life_factors(df: pd.DataFrame, art: PipelineArtifacts) -> pd.DataFrame:
    """III.3 Analyze potential factors for countries with low average life expectancy."""
    if not {"country", "life_expectancy"}.issubset(df.columns):
        return pd.DataFrame()

    country_stats = df.groupby("country").agg(avg_life=("life_expectancy", "mean"))
    q25 = country_stats["avg_life"].quantile(0.25)
    low_countries = country_stats[country_stats["avg_life"] <= q25].index

    df_low = df[df["country"].isin(low_countries)].copy()

    candidate_features = [
        "adult_mortality", "hiv_aids", "measles", "bmi", "alcohol", "polio",
        "hepatitis_b", "diphtheria", "schooling", "income_composition_of_resources",
        "percentage_expenditure", "total_expenditure", "gdp"
    ]
    features = [f for f in candidate_features if f in df_low.columns]

    corr_results = []
    for f in features:
        x = df_low[f]
        y = df_low["life_expectancy"]
        if x.notna().sum() >= 5 and y.notna().sum() >= 5:
            try:
                r_p, p_p = pearsonr(x.dropna(), y.loc[x.dropna().index].dropna())
            except Exception:
                r_p, p_p = np.nan, np.nan
            corr_results.append({
                "feature": f,
                "pearson_r": float(r_p) if pd.notna(r_p) else None,
                "pearson_pvalue": float(p_p) if pd.notna(p_p) else None,
            })

    corr_df = pd.DataFrame(corr_results).sort_values("pearson_r", ascending=False)
    corr_df.to_csv(art.reports_dir / "III3_low_life_factors.csv", index=False)

    # Bar plot of correlations
    if not corr_df.empty:
        plt.figure(figsize=(8, max(3, 0.4 * len(corr_df))))
        sns.barplot(data=corr_df, x="pearson_r", y="feature", palette="vlag")
        plt.axvline(0, color="k", linewidth=1)
        plt.title("Pearson r between features and life expectancy (low-life countries)")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III3_low_life_factors.png", dpi=150)
        plt.close()

    # Additional: compare mean feature values between low and high groups
    comp = _potential_issues_low_life(df)
    comp.to_csv(art.reports_dir / "III3_low_vs_high_feature_means.csv", index=False)

    return corr_df


def _potential_issues_low_life(df: pd.DataFrame) -> pd.DataFrame:
    if not set(["country", "life_expectancy"]).issubset(df.columns):
        return pd.DataFrame()
    country_stats = df.groupby("country").agg(avg_life=("life_expectancy", "mean"))
    q25 = country_stats["avg_life"].quantile(0.25)
    q75 = country_stats["avg_life"].quantile(0.75)
    low_countries = country_stats[country_stats["avg_life"] <= q25].index
    high_countries = country_stats[country_stats["avg_life"] >= q75].index

    features = [c for c in [
        "adult_mortality", "hiv_aids", "measles", "bmi", "alcohol", "polio",
        "hepatitis_b", "diphtheria", "schooling", "income_composition_of_resources",
        "percentage_expenditure", "total_expenditure"
    ] if c in df.columns]

    def summarize(group_countries):
        subset = df[df["country"].isin(group_countries)]
        return subset[features].mean(numeric_only=True)

    low_means = summarize(low_countries)
    high_means = summarize(high_countries)
    comp = pd.DataFrame({
        "low_life_mean": low_means,
        "high_life_mean": high_means,
        "difference_low_minus_high": low_means - high_means,
    }).sort_values("difference_low_minus_high", ascending=False)
    comp.index.name = "feature"
    return comp.reset_index()


def analyze_global_trends(df: pd.DataFrame, art: PipelineArtifacts) -> pd.DataFrame:
    """III.4 Global trends of life expectancy and GDP over time"""
    if "year" not in df.columns:
        return pd.DataFrame()

    trend = df.groupby("year").agg(
        mean_life_expectancy=("life_expectancy", "mean") if "life_expectancy" in df.columns else ("year", "size"),
        mean_gdp=("gdp", "mean") if "gdp" in df.columns else ("year", "size"),
    ).reset_index()
    trend = trend.sort_values("year")
    for col in [c for c in ["mean_life_expectancy", "mean_gdp"] if c in trend.columns]:
        trend[f"{col}_pct_change"] = trend[col].pct_change() * 100

    # Plot trends
    plt.figure(figsize=(9, 4))
    if "mean_life_expectancy" in trend.columns:
        sns.lineplot(data=trend, x="year", y="mean_life_expectancy", label="Life expectancy")
    if "mean_gdp" in trend.columns:
        ax2 = plt.twinx()
        sns.lineplot(data=trend, x="year", y="mean_gdp", color="orange", label="GDP (mean)", ax=ax2)
        ax2.set_ylabel("Mean GDP")
    plt.title("Global trends: Life expectancy and GDP")
    plt.tight_layout()
    plt.savefig(art.figures_dir / "III4_global_trends.png", dpi=150)
    plt.close()

    trend.to_csv(art.reports_dir / "III4_global_trends.csv", index=False)
    return trend


# -------------------- Modeling: train & compare 4 models --------------------

def train_and_evaluate_models(df: pd.DataFrame, art: PipelineArtifacts,
                              target: str = "life_expectancy") -> pd.DataFrame:
    """Train Linear, Ridge, Lasso, RandomForest to predict target.
    Saves model comparison metrics and feature importances for tree model.
    """
    # Prepare feature matrix X and label y
    # Choose candidate features (exclude identifiers and target)
    exclude = {"country", "year", "status", target}
    features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if len(features) == 0:
        print("No numeric features available for modeling.")
        return pd.DataFrame()

    data = df[features + [target]].dropna()
    X = data[features]
    y = data[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipelines
    pipelines = {
        "Linear": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1))]),
        "RandomForest": Pipeline(
            [("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=200, random_state=42))])
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    for name, pipe in pipelines.items():
        # Cross-validation RMSE (negative MSE -> convert)
        neg_mse = cross_val_score(pipe, X_train, y_train, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1)
        rmse_cv = np.sqrt(-neg_mse).mean()

        # Fit on train and evaluate on test
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        result = {
            "model": name,
            "rmse_cv": float(rmse_cv),
            "rmse_test": float(rmse_test),
            "r2_test": float(r2)
        }
        results.append(result)

        # Save simple residual plot
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title(f"{name} Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(art.figures_dir / f"model_{name}_actual_vs_predicted.png", dpi=150)
        plt.close()

        # If RandomForest, save feature importances
        if name == "RandomForest":
            rf = pipe.named_steps["model"]
            importances = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values(
                "importance", ascending=False)
            importances.to_csv(art.reports_dir / "model_randomforest_feature_importances.csv", index=False)
            plt.figure(figsize=(8, max(3, 0.25 * len(importances))))
            sns.barplot(data=importances, x="importance", y="feature")
            plt.title("Random Forest Feature Importances")
            plt.tight_layout()
            plt.savefig(art.figures_dir / "model_randomforest_feature_importances.png", dpi=150)
            plt.close()

    metrics_df = pd.DataFrame(results).sort_values("rmse_test")
    metrics_df.to_csv(art.reports_dir / "model_comparison_metrics.csv", index=False)
    _save_json(metrics_df.to_dict(orient="records"), art.reports_dir / "model_comparison_metrics.json")

    return metrics_df


# -------------------- Main --------------------
if __name__ == "__main__":
    art = _ensure_output_dirs()

    print("Loading and cleaning data...")
    df = _clean_data(art)

    print("III.1 - GDP vs Life Expectancy correlation analysis")
    corr_by_country = analyze_gdp_life_corr(df, art)

    print("III.2 - Top and Bottom countries analysis")
    top, bottom = analyze_top_bottom(df, art, top_n=10)

    print("III.3 - Factors for low-life countries")
    low_factors = analyze_low_life_factors(df, art)

    print("III.4 - Global trends")
    trends = analyze_global_trends(df, art)

    print("Training & comparing predictive models")
    model_metrics = train_and_evaluate_models(df, art)

    print("All done. Outputs saved to ./output (reports + figures).")
