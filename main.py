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

# Import analysis modules
from analyze_part1 import analyze_gdp_life_corr

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


def evaluate_dataset_statistics(df: pd.DataFrame, art: PipelineArtifacts):
    """
    Evaluate basic statistical characteristics of the cleaned dataset using descriptive statistics
    and simple distribution checks. Saves a JSON report and a few diagnostic plots.

    Additionally, save three separate table figures instead of a single combined one:
    - II_dataset_eval_overview_table.png
    - II_dataset_eval_missingness_table.png
    - II_dataset_eval_numeric_summary_table.png

    Returns a dictionary of summary statistics.
    """
    stats: Dict[str, Any] = {}

    # Overall info
    stats["n_rows"] = int(len(df))
    stats["n_columns"] = int(df.shape[1])

    # Column-wise missingness
    missing = df.isna().sum().sort_values(ascending=False)
    stats["missing_counts"] = {k: int(v) for k, v in missing.to_dict().items()}
    stats["missing_pct"] = {k: (int(v) / max(1, len(df))) * 100 for k, v in missing.to_dict().items()}

    # Numeric summary (mean, std, min, max, quartiles, skew, kurtosis)
    num_df = df.select_dtypes(include=[np.number])
    desc = None
    if not num_df.empty:
        desc = num_df.describe(percentiles=[0.25, 0.5, 0.75]).T
        desc["skew"] = num_df.skew(numeric_only=True)
        try:
            # pandas >= 2.5: kurtosis is named kurt; for older versions, .kurt() works too
            desc["kurtosis"] = num_df.kurt(numeric_only=True)
        except TypeError:
            desc["kurtosis"] = num_df.kurt()
        stats["numeric_summary"] = desc.round(4).to_dict(orient="index")

        # Identify potential outliers via IQR rule
        outlier_cols: Dict[str, int] = {}
        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count = int(((s < lower) | (s > upper)).sum())
            outlier_cols[col] = outlier_count
        stats["iqr_outlier_counts"] = outlier_cols

        # Visuals: histograms for a few key fields if present
        key_plots = [c for c in ["life_expectancy", "gdp", "bmi", "alcohol"] if c in num_df.columns]
        for col in key_plots:
            plt.figure(figsize=(6, 4))
            sns.histplot(num_df[col].dropna(), kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(art.figures_dir / f"II_hist_{col}.png", dpi=150)
            plt.close()

            # Additionally, save log10(GDP) histogram if plotting GDP
            if col == "gdp":
                s = pd.to_numeric(num_df[col], errors="coerce").dropna()
                s = s[s > 0]
                if not s.empty:
                    plt.figure(figsize=(6, 4))
                    sns.histplot(np.log10(s), kde=True, bins=30)
                    plt.title("Distribution of log10(gdp)")
                    plt.tight_layout()
                    plt.savefig(art.figures_dir / "II_hist_log_10_gdp.png", dpi=150)
                    plt.close()

    # Categorical summary for a few known categoricals
    cat_cols = [c for c in ["country", "status", "year"] if c in df.columns]
    cat_info: Dict[str, Any] = {}
    for c in cat_cols:
        vals = df[c]
        cat_info[c] = {
            "n_unique": int(vals.nunique(dropna=True)),
            "top_values": vals.value_counts(dropna=True).head(10).to_dict(),
        }
    stats["categorical_summary"] = cat_info

    _save_json(stats, art.reports_dir / "II_dataset_eval_stats.json")

    # ---------- Separate table figures (Overview, Missingness, Numeric Summary) ----------
    try:
        # 1) Overview table
        overview_df = pd.DataFrame({
            "metric": ["n_rows", "n_columns"],
            "value": [stats.get("n_rows", 0), stats.get("n_columns", 0)],
        })
        fig_h = 2 + 0.3 * len(overview_df)
        plt.figure(figsize=(8, fig_h))
        ax = plt.gca()
        ax.axis('off')
        ax.set_title("Overview", fontweight='bold', pad=10)
        tbl = ax.table(cellText=overview_df.values, colLabels=overview_df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)
        plt.tight_layout()
        plt.savefig(art.figures_dir / "II_dataset_eval_overview_table.png", dpi=150)
        plt.close()

        # 2) Missingness table (all)
        miss_df = pd.DataFrame({
            "column": list(stats["missing_counts"].keys()),
            "missing": list(stats["missing_counts"].values()),
        })
        if "missing_pct" in stats:
            miss_df["missing_pct"] = miss_df["column"].map(stats["missing_pct"])  # type: ignore
        miss_df = miss_df.sort_values("missing", ascending=False)
        miss_df["missing_pct"] = miss_df["missing_pct"].map(lambda x: round(float(x), 2) if pd.notna(x) else 0.0)
        fig_h = 2 + 0.18 * max(3, len(miss_df))
        plt.figure(figsize=(12, fig_h))
        ax = plt.gca()
        ax.axis('off')
        ax.set_title("Missingness (All)", fontweight='bold', pad=10)
        tbl = ax.table(cellText=miss_df.values, colLabels=miss_df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.1)
        plt.tight_layout()
        plt.savefig(art.figures_dir / "II_dataset_eval_missingness_table.png", dpi=150)
        plt.close()

        # 3) Numeric summary table (all)
        if desc is not None and not desc.empty:
            sel_cols = ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew"]
            present_cols = [c for c in sel_cols if c in desc.columns]
            num_table = desc[present_cols].round(3).copy()
            try:
                order = num_table["mean"].abs().sort_values(ascending=False).index
                num_table = num_table.loc[order]
            except Exception:
                pass
            num_table.insert(0, "feature", num_table.index)
            num_table = num_table.reset_index(drop=True)
            fig_h = 2 + 0.18 * max(3, len(num_table))
            plt.figure(figsize=(12, fig_h))
            ax = plt.gca()
            ax.axis('off')
            ax.set_title("Numeric Summary (All)", fontweight='bold', pad=10)
            tbl = ax.table(cellText=num_table.values, colLabels=num_table.columns, loc='center')
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.1)
            plt.tight_layout()
            plt.savefig(art.figures_dir / "II_dataset_eval_numeric_summary_table.png", dpi=150)
            plt.close()

            # 3b) Numeric summary table (gdp, log10(gdp), life_expectancy)
            try:
                present_cols = [c for c in ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skew"] if c in desc.columns]
                rows = []
                index_names = []

                # gdp
                if "gdp" in desc.index:
                    rows.append(desc.loc["gdp", present_cols])
                    index_names.append("gdp")

                # log10(gdp)
                if "gdp" in num_df.columns:
                    s = pd.to_numeric(num_df["gdp"], errors="coerce").dropna()
                    s = s[s > 0]
                    if not s.empty:
                        log_s = np.log10(s)
                        log_desc = log_s.describe(percentiles=[0.25, 0.5, 0.75])
                        log_skew = log_s.skew()
                        # Build a row aligned with present_cols
                        log_row = {k: None for k in present_cols}
                        for k in ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]:
                            if k in present_cols and k in log_desc:
                                log_row[k] = float(log_desc[k])
                        if "skew" in present_cols:
                            log_row["skew"] = float(log_skew)
                        rows.append(pd.Series(log_row))
                        index_names.append("log10(gdp)")

                # life_expectancy
                if "life_expectancy" in desc.index:
                    rows.append(desc.loc["life_expectancy", present_cols])
                    index_names.append("life_expectancy")

                if rows:
                    sub_tbl = pd.DataFrame(rows)
                    sub_tbl.index = index_names
                    sub_tbl = sub_tbl.round(3)
                    # Insert feature column and reset index
                    sub_tbl.insert(0, "feature", sub_tbl.index)
                    sub_tbl = sub_tbl.reset_index(drop=True)

                    fig_h = 2 + 0.18 * max(3, len(sub_tbl))
                    plt.figure(figsize=(10, fig_h))
                    ax = plt.gca()
                    ax.axis('off')
                    ax.set_title("Numeric Summary (gdp, log10(gdp), life_expectancy)", fontweight='bold', pad=10)
                    tbl = ax.table(cellText=sub_tbl.values, colLabels=sub_tbl.columns, loc='center')
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(8)
                    tbl.scale(1, 1.1)
                    plt.tight_layout()
                    plt.savefig(art.figures_dir / "II_dataset_eval_numeric_summary_table_gdp_life.png", dpi=150)
                    plt.close()
            except Exception:
                pass

    except Exception:
        # Fail-safe: don't block pipeline if table rendering fails
        pass

    return stats


def evaluate_dataset_visualization(df: pd.DataFrame, art: PipelineArtifacts):
    # ---- Lấy top 10 quốc gia theo tuổi thọ trung bình ----
    # đảm bảo cột life_expectancy tồn tại
    if "life_expectancy" not in df.columns:
        raise ValueError("Không tìm thấy cột 'life_expectancy' trong file. Kiểm tra lại tên cột.")

    top10_countries = (
        df.groupby("country")["life_expectancy"]
        .mean()
        .nlargest(10)
        .index
        .tolist()
    )

    df_top10 = df[df["country"].isin(top10_countries)].copy()

    # Đảm bảo cột year là numeric để vẽ
    df_top10["year"] = pd.to_numeric(df_top10["year"], errors="coerce")

    # Helper: tìm cột hiện có trong dataframe từ 1 danh sách tên khả dĩ
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: tìm theo chứa substring
        for cand in candidates:
            for col in df.columns:
                if cand in col:
                    return col
        return None

    # Danh sách các biến (mỗi mục là list tên khả dĩ) + nhãn tiếng Việt cho trục y và tiêu đề
    plots = [
        (["life_expectancy"], "Tuổi thọ trung bình", "Xu hướng tuổi thọ trung bình (Top 10 quốc gia)"),
        (["adult_mortality"], "Tử vong người lớn (15-60 tuổi)", "Xu hướng tử vong ở người lớn (15–60 tuổi)"),
        (["alcohol"], "Mức tiêu thụ rượu (lit/người/năm)", "Xu hướng sử dụng rượu bia"),
        (["percentage_expenditure"], "Tỷ lệ GDP chi cho y tế (cột percentage_expenditure)",
         "Xu hướng tỷ lệ GDP chi cho y tế"),
        (["hepatitis_b"], "Tỷ lệ tiêm Viêm gan B (%)", "Xu hướng tiêm phòng Viêm gan B"),
        (["bmi"], "BMI trung bình", "Xu hướng chỉ số BMI"),
        (["under_five_deaths", "under-five_deaths", "under_five_death"], "Tử vong trẻ dưới 5 tuổi",
         "Xu hướng tỷ lệ trẻ tử vong dưới 5 tuổi"),
        (["total_expenditure"], "Tỷ lệ chi tiêu y tế trên tổng chi tiêu chính phủ (%) (total_expenditure)",
         "Xu hướng tỷ lệ chi tiêu y tế trên tổng chi tiêu chính phủ"),
        (["gdp"], "GDP bình quân đầu người", "Xu hướng GDP"),
        (["thinness_1_19_years", "thinness_1_19", "thinness_10_19_years"], "Tỷ lệ suy dinh dưỡng trẻ 10-19 (%)",
         "Xu hướng tỷ lệ suy dinh dưỡng trẻ em & thanh thiếu niên 10–19"),
        (["income_composition_of_resources"], "Chỉ số phát triển con người (0-1) - proxy",
         "Xu hướng Chỉ số phát triển con người (proxy)"),
        (["schooling"], "Bình quân số năm đi học", "Xu hướng Bình quân tuổi đến trường (số năm)")
    ]

    # Cài đặt style seaborn
    # sns.set(style="whitegrid", context="talk", palette="tab10")

    # Vẽ tuần tự từng biểu đồ
    for candidates, ylabel, title in plots:
        col = find_col(df_top10, candidates)
        if col is None:
            print(f"[SKIP] Không tìm thấy cột phù hợp cho: {title} (các ứng viên: {candidates})")
            continue

        # chuyển thành numeric (nếu có chuỗi)
        df_top10[col] = pd.to_numeric(df_top10[col], errors="coerce")

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df_top10,
            x="year",
            y=col,
            hue="country",
            hue_order=top10_countries,
            marker="o",
            estimator=None,
            lw=1.5
        )
        plt.title(f"{title}")
        plt.ylabel(ylabel)
        plt.xlabel("Năm")
        # Giới hạn trục X để hiển thị đủ 2000–2015
        plt.xlim(2000, 2015)
        plt.xticks(range(2000, 2016))  # bắt buộc hiện đủ nhãn năm
        # ---- Legend chỉnh nhỏ + đặt ngoài hình ----
        plt.legend(
            title="Quốc gia",
            bbox_to_anchor=(1.05, 1),  # đặt legend ra ngoài bên phải
            loc="upper left",  # căn góc trên bên trái của legend vào điểm (1.05,1)
            fontsize="small",  # cỡ chữ nhỏ hơn
            title_fontsize="small"  # cỡ chữ tiêu đề nhỏ
        )
        plt.tight_layout()

        # Lưu file hình
        safe_col_name = col.replace("/", "_").replace(" ", "_")
        plt.savefig(art.reports_dir / f"II_trend_top10_{safe_col_name}.png", dpi=150)
        plt.close()


# -------------------- Analysis sections (III.1 - III.4) --------------------

# analyze_gdp_life_corr function moved to analyze_muc1.py


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
        # Dual-axis barplot: life expectancy (primary X) and GDP (secondary X) per country
        top_sorted = top.sort_values("avg_life_expectancy", ascending=False)
        countries = top_sorted["country"].tolist()
        y_pos = np.arange(len(countries))
        height = max(4, 0.4 * len(top_sorted))
        fig, ax1 = plt.subplots(figsize=(12, height))
        # Life expectancy bars (horizontal)
        ax1.barh(y_pos - 0.2, top_sorted["avg_life_expectancy"], height=0.4, color="#4C72B0", label="Tuổi thọ TB")
        for yi, val in zip(y_pos, top_sorted["avg_life_expectancy"]):
            ax1.text(val, yi - 0.2, f" {val:.1f}", va="center", ha="left", fontsize=9, color="#2F4B7C")
        ax1.set_xlabel("Tuổi thọ trung bình")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(countries)
        ax1.invert_yaxis()
        # GDP bars on twin x-axis
        if "avg_gdp" in top_sorted.columns:
            ax2 = ax1.twiny()
            ax2.barh(y_pos + 0.2, top_sorted["avg_gdp"], height=0.4, color="#DD8452", alpha=0.7, label="GDP bình quân")
            # Annotate GDP values on GDP bars
            try:
                for yi, val in zip(y_pos, top_sorted["avg_gdp"]):
                    if pd.notna(val):
                        ax2.text(val, yi + 0.2, f" {val:,.0f}", va="center", ha="left", fontsize=8, color="#8B4513")
            except Exception:
                pass
            ax2.set_xlabel("GDP bình quân đầu người")
            # Create a combined legend
            lines = [plt.Rectangle((0,0),1,1, color="#4C72B0"), plt.Rectangle((0,0),1,1, color="#DD8452", alpha=0.7)]
            labels = ["Tuổi thọ TB", "GDP bình quân"]
            ax1.legend(lines, labels, title="Chỉ số", loc="lower right")
        else:
            ax1.legend(title="Chỉ số", loc="lower right")
        plt.title(f"Top {top_n} quốc gia: Tuổi thọ TB và GDP bình quân (2 trục)")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_top_countries.png", dpi=150)
        plt.close()

        # Table image for top countries (country, avg_life_expectancy, avg_gdp)
        fig, ax_tbl = plt.subplots(figsize=(8, max(2.5, 0.35 * len(top_sorted) + 1)))
        ax_tbl.axis('off')
        tbl_df = top_sorted[["country", "avg_life_expectancy", "avg_gdp"]].copy()
        tbl_df["avg_life_expectancy"] = tbl_df["avg_life_expectancy"].round(2)
        if "avg_gdp" in tbl_df.columns:
            tbl_df["avg_gdp"] = tbl_df["avg_gdp"].round(0).astype("Int64").astype(str)
        table = ax_tbl.table(cellText=tbl_df.values,
                             colLabels=["Country", "Avg life exp.", "Avg GDP"],
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax_tbl.set_title(f"Top {top_n} Countries: Life Expectancy & GDP")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_top_countries_table.png", dpi=150)
        plt.close()

    if not bottom.empty:
        # Dual-axis barplot for bottom group
        bottom_sorted = bottom.sort_values("avg_life_expectancy")
        countries_b = bottom_sorted["country"].tolist()
        y_pos = np.arange(len(countries_b))
        height = max(4, 0.4 * len(bottom_sorted))
        fig, ax1 = plt.subplots(figsize=(12, height))
        ax1.barh(y_pos - 0.2, bottom_sorted["avg_life_expectancy"], height=0.4, color="#4C72B0", label="Tuổi thọ TB")
        for yi, val in zip(y_pos, bottom_sorted["avg_life_expectancy"]):
            ax1.text(val, yi - 0.2, f" {val:.1f}", va="center", ha="left", fontsize=9, color="#2F4B7C")
        ax1.set_xlabel("Tuổi thọ trung bình")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(countries_b)
        ax1.invert_yaxis()
        if "avg_gdp" in bottom_sorted.columns:
            ax2 = ax1.twiny()
            ax2.barh(y_pos + 0.2, bottom_sorted["avg_gdp"], height=0.4, color="#DD8452", alpha=0.7, label="GDP bình quân")
            # Annotate GDP values on GDP bars
            try:
                for yi, val in zip(y_pos, bottom_sorted["avg_gdp"]):
                    if pd.notna(val):
                        ax2.text(val, yi + 0.2, f" {val:,.0f}", va="center", ha="left", fontsize=8, color="#8B4513")
            except Exception:
                pass
            ax2.set_xlabel("GDP bình quân đầu người")
            lines = [plt.Rectangle((0,0),1,1, color="#4C72B0"), plt.Rectangle((0,0),1,1, color="#DD8452", alpha=0.7)]
            labels = ["Tuổi thọ TB", "GDP bình quân"]
            ax1.legend(lines, labels, title="Chỉ số", loc="lower right")
        else:
            ax1.legend(title="Chỉ số", loc="lower right")
        plt.title(f"Bottom {top_n} quốc gia: Tuổi thọ TB và GDP bình quân (2 trục)")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_bottom_countries.png", dpi=150)
        plt.close()

        # Table image for bottom countries
        fig, ax_tbl = plt.subplots(figsize=(8, max(2.5, 0.35 * len(bottom_sorted) + 1)))
        ax_tbl.axis('off')
        tbl_df = bottom_sorted[["country", "avg_life_expectancy", "avg_gdp"]].copy()
        tbl_df["avg_life_expectancy"] = tbl_df["avg_life_expectancy"].round(2)
        if "avg_gdp" in tbl_df.columns:
            tbl_df["avg_gdp"] = tbl_df["avg_gdp"].round(0).astype("Int64").astype(str)
        table = ax_tbl.table(cellText=tbl_df.values,
                             colLabels=["Country", "Avg life exp.", "Avg GDP"],
                             loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        ax_tbl.set_title(f"Bottom {top_n} Countries: Life Expectancy & GDP")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III2_bottom_countries_table.png", dpi=150)
        plt.close()

    # Scatter GDP vs life for top/bottom
    for name, subset in [("top", top), ("bottom", bottom)]:
        countries = subset["country"].tolist()
        subdf = df[df["country"].isin(countries)]
        if not subdf.empty and "gdp" in subdf.columns:
            # Compute correlation (Pearson and Spearman) on available pairs
            corr_info = ""
            try:
                pairs = subdf[["gdp", "life_expectancy"]].dropna()
                if len(pairs) >= 3:
                    r_p, p_p = pearsonr(pairs["gdp"], pairs["life_expectancy"])  # type: ignore[arg-type]
                    r_s, p_s = spearmanr(pairs["gdp"], pairs["life_expectancy"])  # type: ignore[arg-type]
                    corr_info = f"\nr(Pearson)={r_p:.2f} (p={p_p:.3g}), rho(Spearman)={r_s:.2f} (p={p_s:.3g})"
            except Exception:
                pass

            plt.figure(figsize=(9, 5))
            # Scatter by country with low alpha and small markers
            sns.scatterplot(data=subdf, x="gdp", y="life_expectancy", hue="country", alpha=0.7, s=40)
            # Add a global trend line (on raw GDP). For log relation, we keep x log-scale for visual linearity
            # Use regplot without hue to draw a single trend line over all points
            sns.regplot(data=subdf, x="gdp", y="life_expectancy", scatter=False, color="black", line_kws={"lw": 1.5, "alpha": 0.8})
            plt.xscale("log")
            plt.xlabel("GDP (log scale)")
            plt.ylabel("Life Expectancy")
            plt.title(f"Tương quan GDP và Tuổi thọ: nhóm {name}{corr_info}")
            plt.tight_layout()
            plt.savefig(art.figures_dir / f"III2_scatter_gdp_life_{name}.png", dpi=150)
            plt.close()

        # GDP trend over years for the same set of countries
        if not subdf.empty and set(["year", "gdp"]).issubset(subdf.columns):
            # aggregate by country-year in case of duplicates
            trend_df = (subdf.groupby(["country", "year"], as_index=False)
                              .agg(mean_gdp=("gdp", "mean")))
            trend_df = trend_df.sort_values(["country", "year"])  # ensure order
            if not trend_df.empty:
                plt.figure(figsize=(9, 5))
                sns.lineplot(data=trend_df, x="year", y="mean_gdp", hue="country", marker="o")
                plt.xlabel("Year")
                plt.ylabel("GDP (mean)")
                plt.title(f"Xu hướng GDP theo thời gian - Nhóm {name} 10 quốc gia")
                plt.tight_layout()
                plt.savefig(art.figures_dir / f"III2_trend_gdp_{name}.png", dpi=150)
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

    # Correlation matrix heatmap for potential features (low-life countries)
    feature_for_corr = features.copy()
    if "life_expectancy" in df_low.columns and "life_expectancy" not in feature_for_corr:
        feature_for_corr.append("life_expectancy")
    # Keep only numeric columns present
    feature_for_corr = [c for c in feature_for_corr if c in df_low.columns]
    if feature_for_corr:
        corr_mat = df_low[feature_for_corr].corr(method="pearson", numeric_only=True)
        # Drop rows/cols that are entirely NaN (if any)
        if corr_mat.notna().any().any():
            # Save CSV of correlation matrix as well
            corr_mat.to_csv(art.reports_dir / "III3_low_life_corr_matrix.csv")
            # Plot heatmap
            n = len(corr_mat.columns)
            plt.figure(figsize=(max(6, 0.6 * n), max(5, 0.6 * n)))
            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.5,
                        cbar_kws={"shrink": 0.8})
            plt.title("Correlation matrix - potential features (low-life countries)")
            plt.tight_layout()
            plt.savefig(art.figures_dir / "III3_low_life_corr_matrix.png", dpi=150)
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

    print("Evaluating dataset statistics after cleaning")
    evaluate_dataset_statistics(df, art)

    evaluate_dataset_visualization(df, art)

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
