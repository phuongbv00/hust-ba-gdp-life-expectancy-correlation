from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = Path("output")
DATA_PATH = Path("data/Life Expectancy Data.csv")


@dataclass
class PipelineArtifacts:
    raw_path: Path
    cleaned_path: Path
    normalized_path: Path
    reports_dir: Path
    figures_dir: Path


def _ensure_output_dirs() -> PipelineArtifacts:
    OUTPUT_DIR.mkdir(exist_ok=True)
    reports_dir = OUTPUT_DIR / "reports"
    figures_dir = OUTPUT_DIR / "figures"
    reports_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    return PipelineArtifacts(
        raw_path=DATA_PATH,
        cleaned_path=OUTPUT_DIR / "cleaned_dataset.csv",
        normalized_path=OUTPUT_DIR / "normalized_dataset.csv",
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


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
    # Some known aliases from docs
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

    # Expected important columns
    expected_cols = {
        "country", "year", "status", "life_expectancy", "gdp", "population",
        "adult_mortality", "infant_deaths", "alcohol", "percentage_expenditure",
        "hepatitis_b", "measles", "bmi", "under_five_deaths", "polio",
        "total_expenditure", "diphtheria", "hiv_aids", "thinness_1_19_years",
        "thinness_5_9_years", "income_composition_of_resources", "schooling"
    }

    # Coerce numerics
    numeric_like = list(expected_cols - {"country", "status"})
    for c in numeric_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Trim strings
    for c in ["country", "status"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()

    # Remove duplicates by (country, year), keep the most complete row
    if {"country", "year"}.issubset(df.columns):
        df = (
            df.sort_values(["country", "year"])
            .assign(missing_count=lambda x: x.isna().sum(axis=1))
            .sort_values(["country", "year", "missing_count"])
            .drop_duplicates(["country", "year"], keep="first")
            .drop(columns=["missing_count"])
        )

    # Minimal requirement: life_expectancy and gdp must exist
    df = df[df["life_expectancy"].notna() & df["gdp"].notna()]

    # Simple imputations: within-country median for a few key indicators
    key_impute = [
        "alcohol", "bmi", "polio", "hepatitis_b", "diphtheria",
        "total_expenditure", "income_composition_of_resources", "schooling",
        "adult_mortality",
    ]
    if "country" in df.columns:
        for col in key_impute:
            if col in df.columns:
                df[col] = df.groupby("country")[col].transform(
                    lambda s: s.fillna(s.median())
                )
    # Global median fallback
    for col in key_impute:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Handle extreme outliers for gdp/measles/percentage_expenditure with clipping (99th percentile)
    for col in ["gdp", "measles", "percentage_expenditure"]:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper, upper, df[col])

    df.to_csv(art.cleaned_path, index=False)

    # Save cleaning report
    report = {
        "rows_before": int(before),
        "rows_after": int(len(df)),
        "columns": list(df.columns),
        "missing_after": df.isna().sum().sort_values(ascending=False).to_dict(),
    }
    _save_json(report, art.reports_dir / "1_cleaning_report.json")
    return df


def _evaluate_quality(df: pd.DataFrame, art: PipelineArtifacts) -> None:
    # 2a. Statistics
    summary = {
        "shape": df.shape,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing": df.isna().sum().to_dict(),
        "duplicates_country_year": int(
            df.duplicated(["country", "year"]).sum() if {"country", "year"}.issubset(df.columns) else 0
        ),
        "describe_numeric": df.select_dtypes(include=[np.number]).describe().to_dict(),
        "unique_countries": int(df["country"].nunique()) if "country" in df.columns else None,
        "years_range": [
            int(df["year"].min()) if "year" in df.columns else None,
            int(df["year"].max()) if "year" in df.columns else None,
        ],
    }
    _save_json(summary, art.reports_dir / "2a_quality_summary.json")

    # 2b. Visualizations
    sns.set(style="whitegrid", context="notebook")

    # Histograms for key variables
    key_vars = ["life_expectancy", "gdp", "bmi", "alcohol", "adult_mortality"]
    for col in key_vars:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), bins=30, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(art.figures_dir / f"2a_hist_{col}.png", dpi=150)
            plt.close()

    # Boxplot life expectancy by status
    if set(["life_expectancy", "status"]).issubset(df.columns):
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x="status", y="life_expectancy")
        plt.title("Life Expectancy by Status")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "2b_box_life_expectancy_by_status.png", dpi=150)
        plt.close()

    # Correlation heatmap for selected variables
    corr_cols = [c for c in [
        "life_expectancy", "gdp", "bmi", "alcohol", "adult_mortality", "polio",
        "hepatitis_b", "diphtheria", "hiv_aids", "schooling",
    ] if c in df.columns]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(method="pearson").round(2)
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation heatmap")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "2b_corr_heatmap.png", dpi=150)
        plt.close()

    # Scatter GDP vs Life Expectancy (log x)
    if set(["gdp", "life_expectancy"]).issubset(df.columns):
        plt.figure(figsize=(6, 4))
        ax = sns.scatterplot(
            data=df, x="gdp", y="life_expectancy",
            hue=df["status"] if "status" in df.columns else None,
            alpha=0.6, edgecolor=None
        )
        ax.set_xscale("log")
        plt.title("GDP per capita vs Life Expectancy")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "2b_scatter_gdp_life_expectancy.png", dpi=150)
        plt.close()


def _normalize_for_analysis(df: pd.DataFrame, art: PipelineArtifacts) -> pd.DataFrame:
    """
    Normalize numeric features for analysis with per-column rules.
    - Keep 'year' unchanged (no normalization).
    - Apply StandardScaler to other numeric analytical features.
    """
    df_norm = df.copy()

    # Identify numeric columns and exclude special identifiers
    numeric_cols_all = df_norm.select_dtypes(include=[np.number]).columns.tolist()

    # Per-field rules: 'year' must be preserved
    exclude_from_scaling = [c for c in ["year"] if c in numeric_cols_all]

    # Columns to scale: numeric minus excluded
    scale_cols = [c for c in numeric_cols_all if c not in exclude_from_scaling]

    scaler = StandardScaler()
    if scale_cols:
        df_norm[scale_cols] = scaler.fit_transform(df_norm[scale_cols])
    else:
        # If nothing to scale, create a dummy fitted scaler to keep meta consistent
        class _DummyScaler:
            mean_ = np.array([])
            scale_ = np.array([])

        scaler = _DummyScaler()

    # Persist normalized dataset
    df_norm.to_csv(art.normalized_path, index=False)

    # Save metadata with clarity on scaled vs excluded columns
    scaler_info = {
        "scaled_columns": scale_cols,
        "excluded_columns": exclude_from_scaling,
        "rules": {
            "year": "kept as original value, not normalized"
        },
        "mean_": scaler.mean_.tolist() if hasattr(scaler, "mean_") else [],
        "scale_": scaler.scale_.tolist() if hasattr(scaler, "scale_") else [],
    }
    _save_json(scaler_info, art.reports_dir / "3_normalization_meta.json")

    return df_norm


def _corr_by_geounit(df: pd.DataFrame) -> pd.DataFrame:
    # 4a. Correlation per geographic unit (country)
    results = []
    if not set(["country", "gdp", "life_expectancy"]).issubset(df.columns):
        return pd.DataFrame(results)

    for country, g in df.groupby("country"):
        g = g.dropna(subset=["gdp", "life_expectancy"])
        if len(g) >= 3:
            # Use log GDP for stability
            x = np.log10(g["gdp"].replace(0, np.nan)).replace([-np.inf, np.inf], np.nan).dropna()
            y = g.loc[x.index, "life_expectancy"]
            if len(x) >= 3 and len(y) >= 3:
                try:
                    r_p, p_p = pearsonr(x, y)
                    r_s, p_s = spearmanr(g["gdp"], g["life_expectancy"], nan_policy='omit')
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
    return pd.DataFrame(results)


def _life_expectancy_extremes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 4b. Highest and lowest life expectancy units (by average over years)
    if not set(["country", "life_expectancy"]).issubset(df.columns):
        return pd.DataFrame(), pd.DataFrame()

    agg = df.groupby("country", as_index=False).agg(
        avg_life_expectancy=("life_expectancy", "mean"),
        avg_gdp=("gdp", "mean") if "gdp" in df.columns else ("life_expectancy", "mean")
    ).sort_values("avg_life_expectancy", ascending=False)

    top = agg.head(10)
    bottom = agg.tail(10)
    return top, bottom


def _potential_issues_low_life(df: pd.DataFrame) -> pd.DataFrame:
    # 4c. Compare feature means between bottom 25% and top 25% countries by avg life expectancy
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


def _global_trends(df: pd.DataFrame) -> pd.DataFrame:
    # 4d. Are life expectancy and GDP growing globally?
    if "year" not in df.columns:
        return pd.DataFrame()

    trend = df.groupby("year").agg(
        mean_life_expectancy=("life_expectancy", "mean") if "life_expectancy" in df.columns else ("year", "size"),
        mean_gdp=("gdp", "mean") if "gdp" in df.columns else ("year", "size"),
    ).reset_index()

    # Simple growth rates
    trend = trend.sort_values("year")
    for col in [c for c in ["mean_life_expectancy", "mean_gdp"] if c in trend.columns]:
        trend[f"{col}_pct_change"] = trend[col].pct_change() * 100

    # Plot trends
    plt.figure(figsize=(8, 4))
    if "mean_life_expectancy" in trend.columns:
        sns.lineplot(data=trend, x="year", y="mean_life_expectancy", label="Life expectancy")
    if "mean_gdp" in trend.columns:
        ax2 = plt.twinx()
        sns.lineplot(data=trend, x="year", y="mean_gdp", color="orange", label="GDP (mean)", ax=ax2)
        ax2.set_ylabel("Mean GDP")
    plt.title("Global trends: Life expectancy and GDP")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "4d_global_trends.png", dpi=150)
    plt.close()
    return trend


def _analyze(df: pd.DataFrame, art: PipelineArtifacts) -> None:
    # 4a
    corr_tbl = _corr_by_geounit(df)
    corr_tbl.to_csv(art.reports_dir / "4a_corr_by_country.csv", index=False)
    # Figure: top 20 absolute Pearson correlations by country
    if not corr_tbl.empty and "pearson_r_log10gdp_life" in corr_tbl.columns:
        plot_df = (
            corr_tbl.dropna(subset=["pearson_r_log10gdp_life"]) \
                .assign(abs_r=lambda x: x["pearson_r_log10gdp_life"].abs()) \
                .sort_values("abs_r", ascending=False).head(20)
        )
        plt.figure(figsize=(8, 6))
        sns.barplot(data=plot_df, x="pearson_r_log10gdp_life", y="country", palette="vlag")
        plt.axvline(0, color="k", linewidth=1)
        plt.title("Top 20 |Pearson r| (log10 GDP vs Life Expectancy) by Country")
        plt.xlabel("Pearson r")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "4a_corr_by_country_top20.png", dpi=150)
        plt.close()

    # 4b
    top, bottom = _life_expectancy_extremes(df)
    top.to_csv(art.reports_dir / "4b_top_life_expectancy_countries.csv", index=False)
    bottom.to_csv(art.reports_dir / "4b_bottom_life_expectancy_countries.csv", index=False)
    # Figures: top and bottom countries by average life expectancy
    if not top.empty and {"country", "avg_life_expectancy"}.issubset(top.columns):
        plt.figure(figsize=(8, 4))
        sns.barplot(data=top.sort_values("avg_life_expectancy"), x="avg_life_expectancy", y="country", color="#4c72b0")
        plt.title("Top 10 Countries by Average Life Expectancy")
        plt.xlabel("Average Life Expectancy")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "4b_top_life_expectancy_countries.png", dpi=150)
        plt.close()
    if not bottom.empty and {"country", "avg_life_expectancy"}.issubset(bottom.columns):
        plt.figure(figsize=(8, 4))
        sns.barplot(data=bottom.sort_values("avg_life_expectancy"), x="avg_life_expectancy", y="country",
                    color="#dd8452")
        plt.title("Bottom 10 Countries by Average Life Expectancy")
        plt.xlabel("Average Life Expectancy")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "4b_bottom_life_expectancy_countries.png", dpi=150)
        plt.close()

    # 4b-extra: Two-group comparisons (Low vs High life expectancy groups)
    # Define groups by country-level average life expectancy (bottom 25% vs top 25%)
    if set(["country", "life_expectancy"]).issubset(df.columns):
        country_avg = df.groupby("country").agg(avg_life=("life_expectancy", "mean"))
        if not country_avg.empty:
            q25 = country_avg["avg_life"].quantile(0.25)
            q75 = country_avg["avg_life"].quantile(0.75)
            low_countries = country_avg[country_avg["avg_life"] <= q25].index
            high_countries = country_avg[country_avg["avg_life"] >= q75].index

            # Label each row with a group (ensure object dtype to avoid numpy promotion issues)
            df_grp = df.copy()
            df_grp["group"] = pd.Series(index=df_grp.index, dtype=object)
            df_grp.loc[df_grp["country"].isin(low_countries), "group"] = "Low life exp (Q1)"
            df_grp.loc[df_grp["country"].isin(high_countries), "group"] = "High life exp (Q4)"
            df_grp = df_grp.dropna(subset=["group"])  # keep only two groups

            # 4b-extra-1: GDP trends over time for the two groups
            if set(["year", "gdp"]).issubset(df_grp.columns) and not df_grp.empty:
                trends_grp = df_grp.groupby(["year", "group"], as_index=False).agg(
                    mean_gdp=("gdp", "mean"),
                    mean_life_expectancy=("life_expectancy", "mean")
                ).sort_values(["year", "group"])
                trends_grp.to_csv(art.reports_dir / "4b_group_trends_by_year.csv", index=False)

                plt.figure(figsize=(8, 4))
                ax = sns.lineplot(data=trends_grp, x="year", y="mean_gdp", hue="group")
                ax.set_ylabel("Mean GDP")
                plt.title("GDP trends by group (Low vs High life expectancy)")
                plt.tight_layout()
                plt.savefig(art.figures_dir / "4b_group_gdp_trends.png", dpi=150)
                plt.close()

                # 4b-extra-3: Life expectancy trends over time for the two groups
                plt.figure(figsize=(8, 4))
                ax = sns.lineplot(data=trends_grp, x="year", y="mean_life_expectancy", hue="group")
                ax.set_ylabel("Mean life expectancy")
                plt.title("Life expectancy trends by group (Low vs High)")
                plt.tight_layout()
                plt.savefig(art.figures_dir / "4b_group_life_expectancy_trends.png", dpi=150)
                plt.close()

            # 4b-extra-2: Correlation comparison between groups (per-country Pearson r)
            # Reuse per-country correlation and map to groups; then draw boxplot of r by group
            if not corr_tbl.empty and "pearson_r_log10gdp_life" in corr_tbl.columns:
                corr_grp = corr_tbl.merge(country_avg.reset_index(), on="country", how="left")
                corr_grp["group"] = pd.Series(index=corr_grp.index, dtype=object)
                corr_grp.loc[corr_grp["country"].isin(low_countries), "group"] = "Low life exp (Q1)"
                corr_grp.loc[corr_grp["country"].isin(high_countries), "group"] = "High life exp (Q4)"
                corr_grp = corr_grp.dropna(subset=["group", "pearson_r_log10gdp_life"])
                if not corr_grp.empty:
                    corr_grp[["country", "group", "pearson_r_log10gdp_life"]].to_csv(
                        art.reports_dir / "4b_group_corr_by_country.csv", index=False
                    )
                    plt.figure(figsize=(6, 4))
                    # Use a stable color palette list to avoid seaborn FutureWarning
                    palette = {"Low life exp (Q1)": "#dd8452", "High life exp (Q4)": "#4c72b0"}
                    sns.boxplot(data=corr_grp, x="group", y="pearson_r_log10gdp_life", palette=palette)
                    sns.stripplot(data=corr_grp, x="group", y="pearson_r_log10gdp_life", color="k", alpha=0.4,
                                  jitter=0.2)
                    plt.axhline(0, color="k", linewidth=1)
                    plt.title("Per-country Pearson r (log10 GDP vs Life) by group")
                    plt.xlabel("")
                    plt.ylabel("Pearson r")
                    plt.tight_layout()
                    plt.savefig(art.figures_dir / "4b_group_correlation_comparison.png", dpi=150)
                    plt.close()

    # 4c
    comp = _potential_issues_low_life(df)
    comp.to_csv(art.reports_dir / "4c_potential_issues_low_life.csv", index=False)
    # Figure: feature differences low minus high life countries
    if not comp.empty and "difference_low_minus_high" in comp.columns and "feature" in comp.columns:
        plot_df = comp.sort_values("difference_low_minus_high", ascending=False)
        plt.figure(figsize=(9, max(4, 0.3 * len(plot_df))))
        sns.barplot(data=plot_df, x="difference_low_minus_high", y="feature", palette="coolwarm")
        plt.axvline(0, color="k", linewidth=1)
        plt.title("Feature Means: Low-life minus High-life Countries")
        plt.xlabel("Difference (low - high)")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "4c_potential_issues_low_life.png", dpi=150)
        plt.close()

    # 4d
    trends = _global_trends(df)
    trends.to_csv(art.reports_dir / "4d_global_trends.csv", index=False)


if __name__ == "__main__":
    art = _ensure_output_dirs()

    df = _clean_data(art)

    _evaluate_quality(df, art)

    # _ = _normalize_for_analysis(df, art)

    _analyze(df, art)
