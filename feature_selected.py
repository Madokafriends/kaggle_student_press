import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

# ppscore 可选：不兼容时自动回退到 PPS*
try:
    import ppscore as pps
    _HAVE_PPS = True
except Exception:
    _HAVE_PPS = False

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score

# ============ 全局输出目录 ============
OUTDIR = Path("Feature_selected")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============ 小工具 ============
def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _task_from_target(y: pd.Series) -> str:
    # auto: 数值且类别数>20 则回归，否则分类
    if _is_numeric(y) and y.nunique(dropna=True) > 20:
        return "reg"
    return "class"

def _factorize(s: pd.Series) -> np.ndarray:
    s2 = s.astype("object").fillna("__MISSING__")
    codes, _ = pd.factorize(s2, sort=True)
    return codes.astype(int)

def _label_encode(y: pd.Series) -> np.ndarray:
    return LabelEncoder().fit_transform(y.astype("object").fillna("__MISSING__"))

def _save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def _pps_star_classification(df: pd.DataFrame, target: str) -> pd.Series:
    """PPS* 近似：单特征树 + 5 折宏 F1 相对多数类基线提升 ∈[0,1]"""
    y_enc = _label_encode(df[target])
    X = df.drop(columns=[target])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_out = {}
    for col in X.columns:
        xi = X[[col]]
        if _is_numeric(xi[col]):
            Xi = SimpleImputer(strategy="median").fit_transform(xi.values)
        else:
            Xi = _factorize(xi[col]).reshape(-1, 1)
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        fold_scores, fold_bases = [], []
        for tr, te in skf.split(Xi, y_enc):
            Xtr, Xte = Xi[tr], Xi[te]
            ytr, yte = y_enc[tr], y_enc[te]
            clf.fit(Xtr, ytr)
            ypred = clf.predict(Xte)
            fold_scores.append(f1_score(yte, ypred, average="macro"))
            # 多数类基线
            vals, cnts = np.unique(ytr, return_counts=True)
            maj = vals[np.argmax(cnts)]
            ybase = np.full_like(yte, maj)
            fold_bases.append(f1_score(yte, ybase, average="macro"))
        cv = float(np.mean(fold_scores))
        base = float(np.mean(fold_bases))
        denom = (1.0 - base) if (1.0 - base) > 1e-12 else 1e-12
        scores_out[col] = max(0.0, min(1.0, (cv - base) / denom))
    return pd.Series(scores_out, name="ppscore").sort_values(ascending=False)

def _pps_star_regression(df: pd.DataFrame, target: str) -> pd.Series:
    """PPS* 近似（回归）：单特征树 + 5 折 R²，裁剪至 [0,1]"""
    y = pd.to_numeric(df[target], errors="coerce")
    X = df.drop(columns=[target])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_out = {}
    for col in X.columns:
        xi = X[[col]]
        if _is_numeric(xi[col]):
            Xi = SimpleImputer(strategy="median").fit_transform(xi.values)
        else:
            Xi = _factorize(xi[col]).reshape(-1, 1)
        fold_scores = []
        for tr, te in kf.split(Xi):
            Xtr, Xte = Xi[tr], Xi[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            reg = DecisionTreeRegressor(max_depth=3, random_state=42)
            reg.fit(Xtr, ytr)
            ypred = reg.predict(Xte)
            fold_scores.append(r2_score(yte, ypred))
        r2 = float(np.mean(fold_scores))
        scores_out[col] = max(0.0, min(1.0, r2))
    return pd.Series(scores_out, name="ppscore").sort_values(ascending=False)

# ============ 1. 数据质量评估 ============
def data_quality_report(df):
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_%': df.isnull().mean() * 100,
        'unique': df.nunique()
    })
    num_cols = df.select_dtypes(include=np.number).columns
    report['skewness'] = pd.Series({c: df[c].skew() for c in num_cols})
    report['kurtosis'] = pd.Series({c: df[c].kurtosis() for c in num_cols})
    desc = df.describe().T
    report = report.join(desc, how='left')
    # 保存
    report.to_csv(OUTDIR / "data_quality_report.csv")
    # 缺失率柱状图（独立图）
    miss = df.isna().mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(miss.index[:30], miss.values[:30])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Missing rate")
    plt.title("Missingness (top 30)")
    _save_fig(OUTDIR / "missingness_bar.png")
    return report

# ============ 2. 独立图可视化（不拼图） ============
def create_visual_matrix(df, target):
    # 目标分布（独立图）
    plt.figure(figsize=(7, 5))
    if _is_numeric(df[target]):
        plt.hist(df[target].dropna().values, bins=30)
        plt.title(f"Target distribution: {target} (hist)")
        plt.xlabel(target); plt.ylabel("Count")
    else:
        vc = df[target].astype("object").fillna("__MISSING__").value_counts()
        plt.bar(vc.index.astype(str), vc.values)
        plt.xticks(rotation=75, ha="right"); plt.ylabel("Count")
        plt.title(f"Target distribution: {target}")
    _save_fig(OUTDIR / "target_distribution.png")

    # 数值特征箱线（独立图）
    num_cols = df.select_dtypes(include=np.number).columns.drop(target, errors='ignore')
    if len(num_cols) > 0:
        plt.figure(figsize=(10, max(4, 0.35 * len(num_cols))))
        sns.boxplot(data=df[num_cols], orient="h")
        plt.title("Numerical Features Distribution")
        _save_fig(OUTDIR / "numeric_boxplot.png")

    # 数值特征相关矩阵（独立图 + CSV）
    if len(df.select_dtypes(include=np.number).columns) >= 2:
        corr_matrix = df.select_dtypes(include=np.number).corr()
        corr_matrix.to_csv(OUTDIR / "pearson_feat_feat_matrix.csv")
        plt.figure(figsize=(max(6, 0.4 * corr_matrix.shape[0]), max(5, 0.4 * corr_matrix.shape[1])))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, cbar=True, square=True)
        plt.title("Feature Correlation Matrix (numeric only)")
        _save_fig(OUTDIR / "pearson_feat_feat_heatmap.png")

    # 特征-目标关系（独立图）
    if _is_numeric(df[target]):
        # Pearson(feature→target)
        num_cols_y = df.select_dtypes(include=np.number).columns.drop(target, errors='ignore')
        if len(num_cols_y) > 0:
            target_corr = df[num_cols_y].corrwith(df[target]).sort_values(key=np.abs, ascending=False)
            target_corr.to_csv(OUTDIR / "pearson_feat_target.csv", header=["pearson_to_target"])
            plt.figure(figsize=(8, 5))
            top10 = target_corr.head(10)[::-1]
            plt.barh(top10.index, top10.values)
            plt.xlabel("corr(feature, target)")
            plt.title("Top-10 Pearson(feature→target)")
            _save_fig(OUTDIR / "pearson_feat_target_top10_bar.png")
    else:
        # 分类目标：用 MI(feature→target) 代替
        num_cols_y = df.select_dtypes(include=np.number).columns
        if len(num_cols_y) > 0:
            X_num = df[num_cols_y].copy()
            X_num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X_num), columns=num_cols_y)
            y_enc = _label_encode(df[target])
            mi = mutual_info_classif(X_num, y_enc, random_state=42)
            mi_series = pd.Series(mi, index=num_cols_y).sort_values(ascending=False)
            mi_series.to_csv(OUTDIR / "mi_numeric_only_to_target.csv", header=["MI_to_target"])
            plt.figure(figsize=(8, 5))
            top10 = mi_series.head(10)[::-1]
            plt.barh(top10.index, top10.values)
            plt.xlabel("MI(feature → target)")
            plt.title("Top-10 MI (numeric features)")
            _save_fig(OUTDIR / "mi_numeric_only_top10_bar.png")

    # PPS（独立图）
    task = _task_from_target(df[target])
    try:
        if _HAVE_PPS:
            pps_df = pps.predictors(df, y=target).sort_values("ppscore", ascending=False)
            pps_series = pps_df.set_index("x")["ppscore"]
        else:
            raise RuntimeError("ppscore not available")
    except Exception:
        pps_series = _pps_star_regression(df, target) if task == "reg" else _pps_star_classification(df, target)
    pps_series.sort_values(ascending=False).to_csv(OUTDIR / "pps_scores.csv", header=["ppscore"])
    plt.figure(figsize=(10, 5))
    top10 = pps_series.head(10)[::-1]
    plt.barh(top10.index, top10.values)
    plt.xlabel("PPS")
    plt.title("Top-10 PPS")
    _save_fig(OUTDIR / "pps_top10_bar.png")

    # Pearson 成对 |ρ| Top-10（独立 CSV + 图）
    num_cols_ff = df.select_dtypes(include=np.number).columns.drop(target, errors='ignore')
    if len(num_cols_ff) >= 2:
        corr = df[num_cols_ff].corr()
        pairs = []
        for i, a in enumerate(num_cols_ff):
            for j in range(i + 1, len(num_cols_ff)):
                b = num_cols_ff[j]
                pairs.append((a, b, float(corr.loc[a, b])))
        pairs_df = pd.DataFrame(pairs, columns=["feature_i", "feature_j", "pearson"])
        pairs_df["abs_corr"] = pairs_df["pearson"].abs()
        pairs_top10 = pairs_df.sort_values("abs_corr", ascending=False).head(10)
        pairs_top10.to_csv(OUTDIR / "pearson_pairs_top10.csv", index=False)

        plt.figure(figsize=(10, 5))
        labels = [f"{r.feature_i} | {r.feature_j}" for _, r in pairs_top10.iterrows()][::-1]
        plt.barh(labels, pairs_top10["abs_corr"].values[::-1])
        plt.xlabel("|corr|")
        plt.title("Top-10 absolute Pearson pairs")
        _save_fig(OUTDIR / "pearson_pairs_top10_bar.png")

    return True  # 仅表示已完成保存

# ============ 3. 高级特征分析（独立图 + 文件） ============
def comprehensive_feature_analysis(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    task = _task_from_target(y)

    # ---- Pearson(feature→target)：仅回归/数值目标可用
    if task == "reg":
        pearson = X.select_dtypes(include=np.number).corrwith(y).abs().sort_values(ascending=False)
    else:
        pearson = pd.Series(dtype=float, name="Pearson")

    # ---- 互信息（对所有列：数值填补中位数，类别 factorize）
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_mi = pd.DataFrame(index=X.index)
    disc_mask = []
    if num_cols:
        X_mi[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
        disc_mask.extend([False] * len(num_cols))
    for c in [c for c in X.columns if c not in num_cols]:
        X_mi[c] = _factorize(X[c])
        disc_mask.append(True)
    disc_mask = np.array(disc_mask, dtype=bool)

    if task == "class":
        y_enc = _label_encode(y)
        mi_vals = mutual_info_classif(X_mi.values, y_enc, discrete_features=disc_mask, random_state=42)
    else:
        y_num = pd.to_numeric(y, errors="coerce")
        mi_vals = mutual_info_regression(X_mi.values, y_num, discrete_features=disc_mask, random_state=42)
    mi_series = pd.Series(mi_vals, index=X_mi.columns, name="Mutual_Info").sort_values(ascending=False)

    # ---- PPS：ppscore 优先，失败回退 PPS*
    try:
        if _HAVE_PPS:
            pps_df = pps.predictors(df, y=target).sort_values("ppscore", ascending=False)
            pps_series = pps_df.set_index("x")["ppscore"]
        else:
            raise RuntimeError("ppscore not available")
    except Exception:
        pps_series = _pps_star_regression(df, target) if task == "reg" else _pps_star_classification(df, target)

    # ---- 汇总 / 归一 / 综合分
    comp = pd.DataFrame({
        'Pearson': pearson,
        'Mutual_Info': mi_series,
        'PPS': pps_series
    }).fillna(0.0)

    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / (rng if rng > 1e-12 else 1.0)

    comp_norm = comp.apply(_norm)
    comp["Composite"] = comp_norm.mean(axis=1)
    comp = comp.sort_values("Composite", ascending=False)

    # ---- 导出 CSV
    comp.to_csv(OUTDIR / "feature_scores_comparison.csv")

    # ---- 各自 Top-10（独立图）
    # MI
    plt.figure(figsize=(10, 5))
    mi_top10 = comp.sort_values("Mutual_Info", ascending=False).head(10)["Mutual_Info"][::-1]
    plt.barh(mi_top10.index, mi_top10.values)
    plt.title("Top-10 Mutual Information")
    plt.xlabel("MI(feature → target)")
    _save_fig(OUTDIR / "mi_top10_bar.png")

    # PPS
    plt.figure(figsize=(10, 5))
    pps_top10 = comp.sort_values("PPS", ascending=False).head(10)["PPS"][::-1]
    plt.barh(pps_top10.index, pps_top10.values)
    plt.title("Top-10 PPS")
    plt.xlabel("PPS")
    _save_fig(OUTDIR / "pps_top10_bar.png")

    # Composite
    plt.figure(figsize=(10, 5))
    comp_top10 = comp["Composite"].head(10)[::-1]
    plt.barh(comp_top10.index, comp_top10.values)
    plt.title("Top-10 Composite Feature Importance")
    plt.xlabel("Composite score")
    _save_fig(OUTDIR / "composite_top10_bar.png")

    # 额外：归一化矩阵热图（Top-10）
    plt.figure(figsize=(8, 6))
    sns.heatmap(comp_norm.head(10), annot=True, cmap="viridis", fmt=".2f")
    plt.title("Normalized Feature Scores (Top-10)")
    _save_fig(OUTDIR / "normalized_scores_top10_heatmap.png")

    return comp

# ============ 4. 工作流：保存文件到 Feature_selected ============
def feature_selection_workflow(df, target):
    print("="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    dq = data_quality_report(df)
    print(f"→ saved: {OUTDIR/'data_quality_report.csv'}")

    print("\n" + "="*50)
    print("VISUALS (SEPARATE FIGURES)")
    print("="*50)
    create_visual_matrix(df, target)
    print(f"→ saved figures under: {OUTDIR}")

    print("\n" + "="*50)
    print("COMPREHENSIVE FEATURE ANALYSIS")
    print("="*50)
    comp = comprehensive_feature_analysis(df, target)
    print(f"→ saved: {OUTDIR/'feature_scores_comparison.csv'}")

    print("\n" + "="*50)
    print("FEATURE SELECTION RECOMMENDATION (Top-10 Composite)")
    print("="*50)
    top_features = comp.head(10).index.tolist()
    print(top_features)

    
    # 选择特征：去除 "self_esteem" 因为这个分类错误率是最高的（如果存在）并添加目标列
    # selected_cols = [c for c in top_features if str(c).lower() != "self_esteem"] + [target]
    # print(f"Selected columns: {selected_cols}")
    # df_optimized = df[selected_cols].copy()
    # print(df_optimized.head())
    

    # 如果只是保留前10个特征，可以直接使用以下行：
    df_optimized = df[top_features + [target]].copy()

    df_optimized.to_csv(OUTDIR / "dataset_top10_features.csv", index=False)
    print(f"→ saved: {OUTDIR/'dataset_top10_features.csv'}")

    return df_optimized, comp

# ============ 5. 直接运行示例 ============
if __name__ == "__main__":
    # 读取你的数据
    df = pd.read_csv("StressLevelDataset.csv")
    optimized_df, feature_report = feature_selection_workflow(df, 'stress_level')
