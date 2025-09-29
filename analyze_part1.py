"""
Mục 1: Phân tích tương quan giữa GDP và Tuổi thọ theo địa lý
Hàm chuyên biệt cho việc phân tích correlation giữa GDP per capita và life expectancy

Tích hợp các phân tích nâng cao:
1. Descriptive Statistics - Thống kê mô tả chi tiết
2. Correlation Heatmap - Ma trận tương quan toàn diện
3. Scatter Plot Matrix - Ma trận scatter plots
4. Regional Analysis - Phân tích theo khu vực địa lý
5. Statistical Validation - Kiểm định thống kê
6. Polynomial Relationships - Mối quan hệ đa thức
7. Temporal Analysis - Phân tích theo thời gian
"""

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def _save_json(obj: Dict[str, Any], path: Path) -> None:
    """Utility function to save JSON objects"""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def analyze_gdp_life_corr(df: pd.DataFrame, art) -> pd.DataFrame:
    """
    III.1 Correlation between GDP and life expectancy (comprehensive analysis)
    
    Thực hiện phân tích toàn diện về tương quan giữa GDP per capita và life expectancy:
    1. Thống kê mô tả chi tiết (GDP raw vs log10(GDP))
    2. Ma trận tương quan toàn diện (correlation heatmap)
    3. Ma trận scatter plots cho các biến quan trọng
    4. Phân tích theo khu vực địa lý
    5. Phân tích tương quan toàn cầu và per-country
    6. Tạo comprehensive report cho báo cáo
    
    Args:
        df: DataFrame đã được cleaned
        art: PipelineArtifacts object chứa paths cho output
        
    Returns:
        pd.DataFrame: Bảng tương quan theo từng quốc gia (compatible với code cũ)
    """
    # 1. Descriptive Statistics - Thống kê mô tả
    desc_stats = calculate_descriptive_statistics(df, art)
    
    # 2. Correlation Heatmap - Ma trận tương quan
    create_correlation_heatmap(df, art)
    
    # 3. Scatter Plot Matrix - Ma trận scatter plots
    create_scatter_plot_matrix(df, art)
    
    # 4. Regional Analysis - Phân tích theo khu vực
    regional_corr = analyze_regional_correlations(df, art)
    
    # 5. Basic Correlation Analysis - Phân tích tương quan cơ bản (giữ nguyên logic cũ)
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

    # Tạo comprehensive report tổng hợp
    comprehensive_results = {
        "analysis_type": "Comprehensive GDP-Life Expectancy Correlation Analysis",
        "total_observations": len(df),
        "countries_analyzed": df["country"].nunique() if "country" in df.columns else 0,
        "analysis_components": [
            "Descriptive Statistics",
            "Correlation Heatmap", 
            "Scatter Plot Matrix",
            "Regional Analysis",
            "Basic Correlation Analysis"
        ],
        "key_findings": {
            "global_pearson_correlation": float(global_pearson[0]),
            "global_spearman_correlation": float(global_spearman.correlation),
            "strongest_regional_correlation": regional_corr["pearson_r"].max() if not regional_corr.empty else None,
            "weakest_regional_correlation": regional_corr["pearson_r"].min() if not regional_corr.empty else None
        },
        "descriptive_statistics": desc_stats.to_dict() if not desc_stats.empty else {},
        "regional_correlations": regional_corr.to_dict() if not regional_corr.empty else {}
    }
    
    _save_json(comprehensive_results, art.reports_dir / "III1_comprehensive_analysis_results.json")

    return corr_df


def calculate_descriptive_statistics(df: pd.DataFrame, art) -> pd.DataFrame:
    """
    Tính toán thống kê mô tả chi tiết cho GDP và Life Expectancy
    
    Mục đích: Cung cấp cái nhìn tổng quan về phân bố dữ liệu
    - So sánh GDP raw vs log10(GDP) để thấy sự cải thiện về phân bố
    - Tính các thước đo xu hướng trung tâm và độ phân tán
    - Tạo histograms để visualize phân bố
    
    Args:
        df: DataFrame đã được cleaned
        art: PipelineArtifacts object
        
    Returns:
        pd.DataFrame: Bảng thống kê mô tả
    """
    if not {"gdp", "life_expectancy"}.issubset(df.columns):
        return pd.DataFrame()

    # Lọc dữ liệu hợp lệ và tính log10(GDP)
    df_valid = df.replace({"gdp": {0: np.nan}}).dropna(subset=["gdp", "life_expectancy"])
    log_gdp = np.log10(df_valid["gdp"]).replace([-np.inf, np.inf], np.nan)

    # Tạo bảng thống kê mô tả
    stats_data = {
        "Variable": ["GDP", "log10(GDP)", "Life Expectancy"],
        "Count": [df_valid["gdp"].notna().sum(), log_gdp.notna().sum(), df_valid["life_expectancy"].notna().sum()],
        "Mean": [df_valid["gdp"].mean(), log_gdp.mean(), df_valid["life_expectancy"].mean()],
        "Median": [df_valid["gdp"].median(), log_gdp.median(), df_valid["life_expectancy"].median()],
        "Std": [df_valid["gdp"].std(), log_gdp.std(), df_valid["life_expectancy"].std()],
        "Min": [df_valid["gdp"].min(), log_gdp.min(), df_valid["life_expectancy"].min()],
        "Max": [df_valid["gdp"].max(), log_gdp.max(), df_valid["life_expectancy"].max()],
        "Skewness": [df_valid["gdp"].skew(), log_gdp.skew(), df_valid["life_expectancy"].skew()],
    }

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(art.reports_dir / "III1_descriptive_statistics.csv", index=False)

    # Tạo histograms để so sánh phân bố
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # GDP raw distribution
    axes[0].hist(df_valid["gdp"], bins=50, alpha=0.7, edgecolor="black")
    axes[0].set_title("GDP Distribution (Raw)")
    axes[0].set_xlabel("GDP per capita")
    axes[0].set_ylabel("Frequency")
    
    # Log10(GDP) distribution - cải thiện tính chuẩn
    axes[1].hist(log_gdp.dropna(), bins=50, alpha=0.7, edgecolor="black")
    axes[1].set_title("log10(GDP) Distribution (Normalized)")
    axes[1].set_xlabel("log10(GDP)")
    axes[1].set_ylabel("Frequency")
    
    # Life expectancy distribution
    axes[2].hist(df_valid["life_expectancy"], bins=50, alpha=0.7, edgecolor="black")
    axes[2].set_title("Life Expectancy Distribution")
    axes[2].set_xlabel("Life Expectancy (years)")
    axes[2].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(art.figures_dir / "III1_descriptive_statistics_histograms.png", dpi=150)
    plt.close()

    return stats_df


def create_correlation_heatmap(df: pd.DataFrame, art) -> pd.DataFrame:
    """
    Tạo correlation heatmap cho tất cả các biến số
    
    Mục đích: Khám phá mối tương quan giữa tất cả các biến
    - Xác định các biến có tương quan mạnh với GDP và Life Expectancy
    - Phát hiện multicollinearity giữa các predictors
    - Hướng dẫn cho việc chọn features trong modeling
    
    Args:
        df: DataFrame đã được cleaned
        art: PipelineArtifacts object
        
    Returns:
        pd.DataFrame: Ma trận tương quan
    """
    # Chọn chỉ các biến số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return pd.DataFrame()

    # Tính ma trận tương quan
    corr_matrix = df[numeric_cols].corr()
    corr_matrix.to_csv(art.reports_dir / "III1_correlation_matrix.csv")

    # Tạo heatmap với mask để chỉ hiển thị tam giác dưới
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix Heatmap - All Variables")
    plt.tight_layout()
    plt.savefig(art.figures_dir / "III1_correlation_heatmap.png", dpi=150)
    plt.close()

    return corr_matrix


def create_scatter_plot_matrix(df: pd.DataFrame, art) -> None:
    """
    Tạo ma trận scatter plots cho các biến quan trọng
    
    Mục đích: Visualize mối quan hệ giữa các biến chính
    - Xem xét mối quan hệ tuyến tính/phi tuyến
    - Phát hiện patterns và clusters
    - Đánh giá assumptions cho correlation analysis
    
    Args:
        df: DataFrame đã được cleaned
        art: PipelineArtifacts object
    """
    # Chọn các biến quan trọng cho phân tích
    key_vars = ["gdp", "life_expectancy", "adult_mortality", "schooling", "bmi"]
    available_vars = [var for var in key_vars if var in df.columns]
    if len(available_vars) < 2:
        return

    # Lọc dữ liệu và tạo pairplot
    df_clean = df[available_vars].dropna()
    fig = sns.pairplot(df_clean, diag_kind="hist", plot_kws={"alpha": 0.6})
    fig.fig.suptitle("Scatter Plot Matrix - Key Variables Relationship", y=1.02)
    plt.savefig(art.figures_dir / "III1_scatter_plot_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def analyze_regional_correlations(df: pd.DataFrame, art) -> pd.DataFrame:
    """
    Phân tích tương quan theo khu vực địa lý
    
    Mục đích: Khám phá sự khác biệt trong mối quan hệ GDP-Life Expectancy theo khu vực
    - So sánh correlation strength giữa các continents
    - Xác định khu vực có mối quan hệ mạnh/yếu nhất
    - Hiểu về regional patterns và economic development stages
    
    Args:
        df: DataFrame đã được cleaned
        art: PipelineArtifacts object
        
    Returns:
        pd.DataFrame: Kết quả tương quan theo khu vực
    """
    if not {"country", "gdp", "life_expectancy"}.issubset(df.columns):
        return pd.DataFrame()

    # Mapping countries to regions - phân loại theo địa lý
    region_mapping = {
        # Africa - các nước châu Phi
        'Algeria': 'Africa', 'Angola': 'Africa', 'Benin': 'Africa', 'Botswana': 'Africa',
        'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cameroon': 'Africa', 'Central African Republic': 'Africa',
        'Chad': 'Africa', 'Comoros': 'Africa', 'Congo': 'Africa', "Cote d'Ivoire": 'Africa',
        'Democratic Republic of the Congo': 'Africa', 'Djibouti': 'Africa', 'Egypt': 'Africa',
        'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa', 'Ethiopia': 'Africa', 'Gabon': 'Africa',
        'Gambia': 'Africa', 'Ghana': 'Africa', 'Guinea': 'Africa', 'Guinea-Bissau': 'Africa',
        'Kenya': 'Africa', 'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa',
        'Madagascar': 'Africa', 'Malawi': 'Africa', 'Mali': 'Africa', 'Mauritania': 'Africa',
        'Mauritius': 'Africa', 'Morocco': 'Africa', 'Mozambique': 'Africa', 'Namibia': 'Africa',
        'Niger': 'Africa', 'Nigeria': 'Africa', 'Rwanda': 'Africa', 'Senegal': 'Africa',
        'Sierra Leone': 'Africa', 'Somalia': 'Africa', 'South Africa': 'Africa', 'South Sudan': 'Africa',
        'Sudan': 'Africa', 'Swaziland': 'Africa', 'Tanzania': 'Africa', 'Togo': 'Africa',
        'Tunisia': 'Africa', 'Uganda': 'Africa', 'Zambia': 'Africa', 'Zimbabwe': 'Africa',

        # Asia - các nước châu Á
        'Afghanistan': 'Asia', 'Bangladesh': 'Asia', 'Bhutan': 'Asia', 'Brunei': 'Asia',
        'Cambodia': 'Asia', 'China': 'Asia', 'India': 'Asia', 'Indonesia': 'Asia',
        'Iran': 'Asia', 'Iraq': 'Asia', 'Israel': 'Asia', 'Japan': 'Asia',
        'Jordan': 'Asia', 'Kazakhstan': 'Asia', 'Kuwait': 'Asia', 'Kyrgyzstan': 'Asia',
        'Laos': 'Asia', 'Lebanon': 'Asia', 'Malaysia': 'Asia', 'Maldives': 'Asia',
        'Mongolia': 'Asia', 'Myanmar': 'Asia', 'Nepal': 'Asia', 'North Korea': 'Asia',
        'Oman': 'Asia', 'Pakistan': 'Asia', 'Palestine': 'Asia', 'Philippines': 'Asia',
        'Qatar': 'Asia', 'Saudi Arabia': 'Asia', 'Singapore': 'Asia', 'South Korea': 'Asia',
        'Sri Lanka': 'Asia', 'Syria': 'Asia', 'Tajikistan': 'Asia', 'Thailand': 'Asia',
        'Timor-Leste': 'Asia', 'Turkey': 'Asia', 'Turkmenistan': 'Asia', 'United Arab Emirates': 'Asia',
        'Uzbekistan': 'Asia', 'Vietnam': 'Asia', 'Yemen': 'Asia',

        # Europe - các nước châu Âu
        'Albania': 'Europe', 'Austria': 'Europe', 'Belarus': 'Europe', 'Belgium': 'Europe',
        'Bosnia and Herzegovina': 'Europe', 'Bulgaria': 'Europe', 'Croatia': 'Europe',
        'Cyprus': 'Europe', 'Czech Republic': 'Europe', 'Denmark': 'Europe', 'Estonia': 'Europe',
        'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe',
        'Hungary': 'Europe', 'Iceland': 'Europe', 'Ireland': 'Europe', 'Italy': 'Europe',
        'Latvia': 'Europe', 'Lithuania': 'Europe', 'Luxembourg': 'Europe', 'Malta': 'Europe',
        'Moldova': 'Europe', 'Montenegro': 'Europe', 'Netherlands': 'Europe', 'Norway': 'Europe',
        'Poland': 'Europe', 'Portugal': 'Europe', 'Romania': 'Europe', 'Russia': 'Europe',
        'Serbia': 'Europe', 'Slovakia': 'Europe', 'Slovenia': 'Europe', 'Spain': 'Europe',
        'Sweden': 'Europe', 'Switzerland': 'Europe', 'Ukraine': 'Europe', 'United Kingdom': 'Europe',

        # Americas - các nước châu Mỹ
        'Argentina': 'Americas', 'Bahamas': 'Americas', 'Barbados': 'Americas', 'Belize': 'Americas',
        'Bolivia': 'Americas', 'Brazil': 'Americas', 'Canada': 'Americas', 'Chile': 'Americas',
        'Colombia': 'Americas', 'Costa Rica': 'Americas', 'Cuba': 'Americas', 'Dominica': 'Americas',
        'Dominican Republic': 'Americas', 'Ecuador': 'Americas', 'El Salvador': 'Americas',
        'Grenada': 'Americas', 'Guatemala': 'Americas', 'Guyana': 'Americas', 'Haiti': 'Americas',
        'Honduras': 'Americas', 'Jamaica': 'Americas', 'Mexico': 'Americas', 'Nicaragua': 'Americas',
        'Panama': 'Americas', 'Paraguay': 'Americas', 'Peru': 'Americas', 'Saint Kitts and Nevis': 'Americas',
        'Saint Lucia': 'Americas', 'Saint Vincent and the Grenadines': 'Americas', 'Suriname': 'Americas',
        'Trinidad and Tobago': 'Americas', 'United States of America': 'Americas', 'Uruguay': 'Americas',
        'Venezuela': 'Americas',

        # Oceania - các nước châu Đại Dương
        'Australia': 'Oceania', 'Fiji': 'Oceania', 'Kiribati': 'Oceania', 'Marshall Islands': 'Oceania',
        'Micronesia': 'Oceania', 'Nauru': 'Oceania', 'New Zealand': 'Oceania', 'Palau': 'Oceania',
        'Papua New Guinea': 'Oceania', 'Samoa': 'Oceania', 'Solomon Islands': 'Oceania', 'Tonga': 'Oceania',
        'Tuvalu': 'Oceania', 'Vanuatu': 'Oceania',
    }

    # Thêm cột region vào dataframe
    df_regional = df.copy()
    df_regional["region"] = df_regional["country"].map(region_mapping)
    df_regional = df_regional.dropna(subset=["region"])

    # Tính correlation cho từng khu vực
    regional_results = []
    for region in df_regional["region"].unique():
        region_data = df_regional[df_regional["region"] == region]
        region_data = region_data.dropna(subset=["gdp", "life_expectancy"])
        if len(region_data) >= 10:  # Đảm bảo có đủ dữ liệu
            try:
                # Sử dụng log10(GDP) để ổn định correlation
                log_gdp = (
                    np.log10(region_data["gdp"].replace(0, np.nan))
                    .replace([-np.inf, np.inf], np.nan)
                )
                valid_idx = log_gdp.dropna().index.intersection(region_data.index)
                if len(valid_idx) >= 10:
                    # Tính Pearson và Spearman correlation
                    r_pearson, p_pearson = pearsonr(
                        log_gdp.loc[valid_idx], region_data.loc[valid_idx, "life_expectancy"]
                    )
                    r_spearman, p_spearman = spearmanr(
                        region_data.loc[valid_idx, "gdp"],
                        region_data.loc[valid_idx, "life_expectancy"],
                        nan_policy="omit",
                    )
                    regional_results.append(
                        {
                            "region": region,
                            "n_countries": region_data["country"].nunique(),
                            "n_observations": len(valid_idx),
                            "pearson_r": float(r_pearson),
                            "pearson_pvalue": float(p_pearson),
                            "spearman_r": float(r_spearman),
                            "spearman_pvalue": float(p_spearman),
                        }
                    )
            except Exception as e:
                print(f"Error processing region {region}: {e}")

    regional_df = pd.DataFrame(regional_results)
    regional_df.to_csv(art.reports_dir / "III1_regional_correlations.csv", index=False)

    # Visualize kết quả
    if not regional_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=regional_df.sort_values("pearson_r"), x="pearson_r", y="region")
        plt.title("GDP-Life Expectancy Correlation by Region")
        plt.xlabel("Pearson Correlation Coefficient")
        plt.tight_layout()
        plt.savefig(art.figures_dir / "III1_regional_correlations.png", dpi=150)
        plt.close()

    return regional_df

