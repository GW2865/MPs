# -*- coding: utf-8 -*-
import io
import json
import math
import shutil
import tempfile
import warnings
from contextlib import ExitStack
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
import streamlit as st
from rasterio.io import MemoryFile
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, RepeatedKFold

try:
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
except Exception:
    WarpedVRT = None
    Resampling = None

st.set_page_config(
    page_title="MicroFragment Atlas Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- style ----------

def bi(zh, en):
    return f"{zh} / {en}"

def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(32, 84, 147, 0.10), transparent 24%),
                radial-gradient(circle at 90% 12%, rgba(0, 119, 145, 0.10), transparent 22%),
                radial-gradient(circle at 78% 78%, rgba(71, 111, 150, 0.08), transparent 28%),
                linear-gradient(180deg, #f3f7fb 0%, #edf2f7 46%, #f8fbfd 100%);
            color: #102033;
        }
        .block-container {
            max-width: 1320px;
            padding-top: 1rem;
            padding-bottom: 2rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 1.6rem 1.8rem;
            background: linear-gradient(135deg, rgba(18,52,86,0.92), rgba(31,92,125,0.88));
            box-shadow: 0 24px 60px rgba(16, 32, 51, 0.16);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .hero:after {
            content: "";
            position: absolute;
            right: -30px;
            bottom: -30px;
            width: 240px;
            height: 240px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.20), transparent 62%);
            pointer-events: none;
        }
        .hero h1 {
            margin: 0 0 .35rem 0;
            color: #f8fcff;
            font-size: 2.35rem;
            letter-spacing: -.03em;
            font-weight: 800;
        }
        .hero p {
            margin: 0;
            color: rgba(248,252,255,0.86);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 880px;
        }
        .kicker {
            display: inline-block;
            margin-bottom: .7rem;
            padding: .32rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            color: #d7eef8;
            font-size: .78rem;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            border: 1px solid rgba(255,255,255,0.10);
        }
        .glass {
            border-radius: 22px;
            border: 1px solid rgba(16,32,51,.08);
            background: rgba(255,255,255,.84);
            box-shadow: 0 14px 40px rgba(16,32,51,.06);
            padding: 1rem 1.05rem .9rem 1.05rem;
        }
        .section-title {
            margin-top: .15rem;
            margin-bottom: .55rem;
            color: #102033;
            font-weight: 800;
            letter-spacing: -.02em;
        }
        .tiny {
            color: #52657b;
            font-size: .92rem;
        }
        .soft-note {
            padding: .85rem 1rem;
            border-radius: 16px;
            background: rgba(16,32,51,.04);
            border: 1px solid rgba(16,32,51,.08);
            color: #435467;
            line-height: 1.6;
            font-size: .95rem;
        }
        .stMetric {
            background: rgba(255,255,255,.9);
            border: 1px solid rgba(16,32,51,.06);
            padding: .65rem .85rem;
            border-radius: 18px;
            box-shadow: 0 10px 26px rgba(16,32,51,.04);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(16,32,51,.08);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            font-weight: 700;
            border: 1px solid rgba(16,32,51,.10);
        }
        .sidebar-note {
            color: #52657b;
            font-size: .92rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.markdown(
    """
    <div class="glass">
        <div class="section-title">研究流程建议 / Recommended workflow</div>
        <div class="tiny">
            1. 上传采样 CSV。 2. 选择目标变量与坐标字段。 3. 运行建模。 4. 仅在需要解释时启用 SHAP。 5. 上传对齐后的 TIFF 并进行栅格预测。<br/>
            1. Upload the sampling CSV. 2. Select the target and coordinate fields. 3. Run modelling. 4. Enable SHAP only when interpretability is needed. 5. Upload aligned TIFFs for raster prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <div class="kicker">科研建模平台 / Scientific Modelling Platform</div>
        <h1>MicroFragment Atlas</h1>
        <p>
            用于微塑料破碎建模、随机森林解释分析与栅格空间预测的双语科研应用。<br/>
            A bilingual scientific workspace for microplastic fragmentation modelling, interpretable random forest analysis, and raster-based spatial prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def metric_dict(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")

def _shap_values_tree_explainer(explainer, X):
    try:
        out = explainer(X)
        sv = out.values if hasattr(out, "values") else out
    except Exception:
        sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    return sv

def sanitize_feature_name(name: str):
    return Path(name).stem.lower().replace("1000", "")

def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file, na_values=["#VALUE!", "NaN", "nan", "Inf", "inf"])

def prepare_simple_training_data(df, target, x_coord=None, y_coord=None, drop_cols=None):
    drop_cols = drop_cols or []
    df2 = df.copy()

    required = [target]
    if x_coord:
        required.append(x_coord)
    if y_coord:
        required.append(y_coord)

    keep_cols = [c for c in df2.columns if c not in set(drop_cols)]
    df2 = df2[keep_cols].dropna()

    if target not in df2.columns:
        raise ValueError(f"Target column '{target}' is not present after exclusions.")

    predictors = [c for c in df2.columns if c != target and c not in {x_coord, y_coord}]
    if not predictors:
        raise ValueError("No predictors remain. Please keep at least one predictor column.")

    X_df = df2[predictors].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df2[target], errors="coerce").values.astype(float)

    coord_df = None
    if x_coord and y_coord and x_coord in df2.columns and y_coord in df2.columns:
        coord_df = df2[[x_coord, y_coord]].apply(pd.to_numeric, errors="coerce")

    good = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    if coord_df is not None:
        good &= np.isfinite(coord_df.values).all(axis=1)

    X_df = X_df.loc[good].copy()
    y = y[good]
    if coord_df is not None:
        coord_df = coord_df.loc[good].copy()

    if len(X_df) < 5:
        raise ValueError(f"Only {len(X_df)} valid rows remain after dropna()/numeric conversion. Please check the CSV.")
    if X_df.shape[1] < 1:
        raise ValueError("No usable predictors remain after preprocessing.")
    if np.unique(y).size < 2:
        raise ValueError("Target variable has fewer than 2 unique values after preprocessing.")

    return X_df, y, coord_df

def fit_simple_rf(X, y, n_estimators=300, random_state=42):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X, y)
    return model

def cross_val_summary_for_fixed_model(X, y, n_estimators=300, random_state=42, cv_splits=5):
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold_id, (tr_idx, te_idx) in enumerate(cv.split(X), start=1):
        m = fit_simple_rf(X[tr_idx], y[tr_idx], n_estimators=n_estimators, random_state=random_state)
        pred = m.predict(X[te_idx])
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    df_folds = pd.DataFrame(rows)
    return df_folds, float(df_folds["R2"].mean())

def compute_shap_sample(model, X_df, sample_size=200, random_state=42):
    if len(X_df) > sample_size:
        X_use = X_df.sample(sample_size, random_state=random_state).copy()
    else:
        X_use = X_df.copy()
    explainer = shap.TreeExplainer(model)
    sv = _shap_values_tree_explainer(explainer, X_use.values.astype("float32", copy=False))
    shap_df = pd.DataFrame(sv, columns=X_use.columns, index=X_use.index)
    imp = pd.DataFrame({
        "变量 / feature": X_use.columns,
        "mean_abs_shap": np.abs(shap_df.values).mean(axis=0),
        "mean_shap": shap_df.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return X_use, shap_df, imp

def prepare_predictors(df, target, x_coord=None, y_coord=None, drop_cols=None):
    drop_cols = drop_cols or []
    exclude = {target}
    if x_coord:
        exclude.add(x_coord)
    if y_coord:
        exclude.add(y_coord)

    predictors = [c for c in df.columns if c not in exclude and c not in drop_cols]
    X_df = df[predictors].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce").values.astype(float)

    coord_df = None
    if x_coord and y_coord and x_coord in df.columns and y_coord in df.columns:
        coord_df = df[[x_coord, y_coord]].apply(pd.to_numeric, errors="coerce")

    good = np.isfinite(X_df.values).all(axis=1) & np.isfinite(y)
    if coord_df is not None:
        good &= np.isfinite(coord_df.values).all(axis=1)

    return X_df.loc[good].copy(), y[good], None if coord_df is None else coord_df.loc[good].copy()

def collinearity_filter(X_df, threshold=0.85, method="spearman", always_keep=None):
    always_keep = set(always_keep or [])
    feats0 = list(X_df.columns)
    if len(feats0) <= 1:
        return X_df.copy(), pd.DataFrame({"变量 / feature": feats0, "kept": True, "reason": "not_filtered"}), pd.DataFrame()

    corr = X_df.corr(method=method).abs()
    remaining = list(feats0)
    removed = {}
    removed_pairs = []

    while True:
        pairs = []
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                a, b = remaining[i], remaining[j]
                v = corr.loc[a, b]
                if pd.notna(v) and v >= threshold:
                    pairs.append((a, b, float(v)))
        if not pairs:
            break

        a, b, v = sorted(pairs, key=lambda x: x[2], reverse=True)[0]
        if a in always_keep and b in always_keep:
            corr.loc[a, b] = -np.inf
            corr.loc[b, a] = -np.inf
            continue
        elif a in always_keep:
            drop, keep = b, a
        elif b in always_keep:
            drop, keep = a, b
        else:
            a_score = corr.loc[a, remaining].drop(a).mean()
            b_score = corr.loc[b, remaining].drop(b).mean()
            drop, keep = (a, b) if a_score >= b_score else (b, a)

        remaining.remove(drop)
        removed[drop] = f"removed_due_to_collinearity_with_{keep}"
        removed_pairs.append({"feature_a": a, "feature_b": b, "abs_corr": v, "dropped": drop, "kept": keep})

    report_rows = [{"变量 / feature": f, "kept": f in remaining, "reason": "kept" if f in remaining else removed.get(f, "removed")} for f in feats0]
    return X_df[remaining].copy(), pd.DataFrame(report_rows), pd.DataFrame(removed_pairs)

def get_param_distributions():
    return {
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": [1.0, "sqrt", 0.5, 0.7],
        "bootstrap": [True],
    }

def fit_best_rf(X, y, random_state=42, search_iter=20, cv_splits=5):
    base_model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=1,
    )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=get_param_distributions(),
        n_iter=search_iter,
        scoring="r2",
        cv=cv,
        n_jobs=1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, float(search.best_score_)

def evaluate_repeated_cv(model, X, y, random_state=42, cv_splits=5, cv_repeats=3):
    rkf = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=random_state)
    rows = []
    oof = np.full(y.shape[0], np.nan, dtype=float)
    for fold_id, (tr_idx, te_idx) in enumerate(rkf.split(X), start=1):
        m = RandomForestRegressor(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        first_time = np.isnan(oof[te_idx])
        oof[te_idx[first_time]] = pred[first_time]
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    return pd.DataFrame(rows), pd.DataFrame({"observed": y, "predicted_oof": oof, "残差 / residual": y - oof})

def evaluate_spatial_cv(model, X, y, coord_df, random_state=42, spatial_blocks=5):
    if coord_df is None or coord_df.empty:
        return None, None
    km = KMeans(n_clusters=spatial_blocks, random_state=random_state, n_init=20)
    groups = km.fit_predict(coord_df.values)
    n_splits = min(spatial_blocks, len(np.unique(groups)))
    if n_splits < 2:
        return None, None
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    oof = np.full(y.shape[0], np.nan, dtype=float)
    for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = RandomForestRegressor(**model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        pred = m.predict(X[te_idx])
        oof[te_idx] = pred
        rows.append({"fold": fold_id, **metric_dict(y[te_idx], pred)})
    return pd.DataFrame(rows), pd.DataFrame({"observed": y, "predicted_spatial_oof": oof, "残差 / residual": y - oof, "group": groups})

def compute_permutation_importance(model, X, y, feature_names, random_state=42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pi = permutation_importance(model, X, y, n_repeats=15, random_state=random_state, n_jobs=1, scoring="r2")
    return pd.DataFrame({
        "变量 / feature": feature_names,
        "perm_importance_mean": pi.importances_mean,
        "perm_importance_std": pi.importances_std,
    }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

def compute_shap_summary(model, X_df):
    explainer = shap.TreeExplainer(model)
    sv = _shap_values_tree_explainer(explainer, X_df.values.astype("float32", copy=False))
    shap_df = pd.DataFrame(sv, columns=X_df.columns)
    imp = pd.DataFrame({
        "变量 / feature": X_df.columns,
        "mean_abs_shap": np.abs(shap_df.values).mean(axis=0),
        "mean_shap": shap_df.values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return shap_df, imp

def sample_for_shap(X_df, max_rows=200, random_state=42):
    if len(X_df) <= max_rows:
        return X_df.copy()
    return X_df.sample(max_rows, random_state=random_state).copy()

def compute_single_feature_shap(model, X_df, feature_name, max_rows=200, random_state=42):
    Xs = sample_for_shap(X_df, max_rows=max_rows, random_state=random_state)
    explainer = shap.TreeExplainer(model)
    sv = _shap_values_tree_explainer(explainer, Xs.values.astype("float32", copy=False))
    shap_df = pd.DataFrame(sv, columns=Xs.columns)
    if feature_name not in shap_df.columns:
        raise ValueError(f"Feature '{feature_name}' not found in SHAP results.")
    return pd.DataFrame({
        "feature_value": Xs[feature_name].values,
        "shap_value": shap_df[feature_name].values,
    }).sort_values("feature_value").reset_index(drop=True)

def fig_single_feature_shap(df_plot, feature_name):
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    ax.scatter(df_plot["feature_value"], df_plot["shap_value"], s=24, alpha=0.68)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP response for {feature_name}")
    return fig


def fig_observed_pred(df_oof, pred_col="predicted_oof", title="观测值与预测值 / Observed vs predicted"):
    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    good = np.isfinite(df_oof["observed"]) & np.isfinite(df_oof[pred_col])
    x = df_oof.loc[good, "observed"].values
    y = df_oof.loc[good, pred_col].values
    ax.scatter(x, y, s=28, alpha=0.72)
    if len(x) > 0:
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], linewidth=1.2)
    ax.set_xlabel("观测值 / Observed")
    ax.set_ylabel("预测值 / Predicted")
    ax.set_title(title)
    return fig

def fig_barh(df, value_col, label_col="变量 / feature", title="", top_n=15, xlabel=None):
    d = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(4, 0.35 * len(d))))
    ax.barh(d[label_col], d[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel or value_col)
    ax.set_ylabel("")
    return fig

# ---------- raster helpers ----------
def save_uploaded_rasters_to_temp(uploaded_rasters):
    tempdir = Path(tempfile.mkdtemp(prefix="microfragment_rasters_"))
    saved = []
    for up in uploaded_rasters:
        p = tempdir / up.name
        p.write_bytes(up.getbuffer())
        saved.append(p)
    return tempdir, saved

def find_raster_for_feature(feature, raster_paths):
    feature_norm = feature.lower().strip()
    for p in raster_paths:
        stem = sanitize_feature_name(p.name)
        if stem == feature_norm:
            return str(p)
    for p in raster_paths:
        stem = sanitize_feature_name(p.name)
        if feature_norm in stem or stem in feature_norm:
            return str(p)
    raise FileNotFoundError(f"No uploaded TIFF matched feature '{feature}'.")

def open_and_align_datasets(feature_names, raster_paths, resampling_name="bilinear"):
    path_map = {f: find_raster_for_feature(f, raster_paths) for f in feature_names}
    stack = ExitStack()
    raw = {}
    try:
        for feat, p in path_map.items():
            raw[feat] = stack.enter_context(rasterio.open(p))
        ref_key = next(iter(raw.keys()))
        ref_ds = raw[ref_key]
        datasets = {}
        if WarpedVRT is None or Resampling is None:
            for feat, ds in raw.items():
                datasets[feat] = ds
            return stack, datasets

        resampling = getattr(Resampling, resampling_name, Resampling.bilinear)
        for feat, ds in raw.items():
            same = (
                ds.crs == ref_ds.crs and
                ds.transform == ref_ds.transform and
                ds.width == ref_ds.width and
                ds.height == ref_ds.height
            )
            if same:
                datasets[feat] = ds
            else:
                vrt = WarpedVRT(
                    ds,
                    crs=ref_ds.crs,
                    transform=ref_ds.transform,
                    width=ref_ds.width,
                    height=ref_ds.height,
                    resampling=resampling,
                )
                datasets[feat] = stack.enter_context(vrt)
        return stack, datasets
    except Exception:
        stack.close()
        raise

def _make_profile(tmpl_ds, nodata=-9999.0):
    profile = tmpl_ds.profile.copy()
    profile["driver"] = "GTiff"
    profile["count"] = 1
    profile["dtype"] = "float32"
    profile["nodata"] = float(nodata)
    profile["compress"] = "lzw"
    return profile

def read_predictor_block(datasets, feature_names, win):
    h = int(win.height)
    w = int(win.width)
    p = len(feature_names)
    stack_x = np.empty((h, w, p), dtype="float32")
    invalid = None
    for j, fn in enumerate(feature_names):
        ds = datasets[fn]
        arr = ds.read(1, window=win).astype("float32", copy=False)
        inv = ~np.isfinite(arr)
        if ds.nodata is not None:
            inv |= (arr == ds.nodata)
        invalid = inv if invalid is None else (invalid | inv)
        stack_x[:, :, j] = arr
    return stack_x, invalid

def build_kfold_models(best_model, X, y, cv_splits=5, random_state=42):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = []
    for tr_idx, te_idx in kf.split(X):
        m = RandomForestRegressor(**best_model.get_params())
        m.fit(X[tr_idx], y[tr_idx])
        models.append(m)
    return models

def run_raster_prediction(best_model, feature_names, raster_paths, X_train_df, y, cv_splits=5, random_state=42, block=512):
    tempdir = Path(tempfile.mkdtemp(prefix="microfragment_outputs_"))
    stack, datasets = open_and_align_datasets(feature_names, raster_paths)
    try:
        tmpl = datasets[next(iter(datasets.keys()))]
        profile = _make_profile(tmpl, nodata=-9999.0)
        pred_path = tempdir / "prediction.tif"
        mean_unc_path = tempdir / "prediction_cv_mean.tif"
        std_unc_path = tempdir / "prediction_cv_std.tif"

        X_train = X_train_df.values.astype("float32", copy=False)
        kfold_models = build_kfold_models(best_model, X_train, y, cv_splits=cv_splits, random_state=random_state)

        with rasterio.open(pred_path, "w", **profile) as dst_pred, \
             rasterio.open(mean_unc_path, "w", **profile) as dst_mean, \
             rasterio.open(std_unc_path, "w", **profile) as dst_std:
            for row0 in range(0, tmpl.height, block):
                for col0 in range(0, tmpl.width, block):
                    h = min(block, tmpl.height - row0)
                    w = min(block, tmpl.width - col0)
                    win = Window(col0, row0, w, h)

                    stack_x, invalid = read_predictor_block(datasets, feature_names, win)
                    Xw = stack_x.reshape(-1, len(feature_names))
                    valid = ~invalid.reshape(-1)

                    pred = np.full(h * w, -9999.0, dtype="float32")
                    pred_mean = np.full(h * w, -9999.0, dtype="float32")
                    pred_std = np.full(h * w, -9999.0, dtype="float32")

                    if np.any(valid):
                        Xv = Xw[valid]
                        pred[valid] = best_model.predict(Xv).astype("float32", copy=False)
                        preds = np.column_stack([m.predict(Xv).astype("float32", copy=False) for m in kfold_models])
                        pred_mean[valid] = preds.mean(axis=1)
                        pred_std[valid] = preds.std(axis=1, ddof=1 if preds.shape[1] > 1 else 0)

                    dst_pred.write(pred.reshape(h, w), 1, window=win)
                    dst_mean.write(pred_mean.reshape(h, w), 1, window=win)
                    dst_std.write(pred_std.reshape(h, w), 1, window=win)
        return tempdir, pred_path, mean_unc_path, std_unc_path
    finally:
        stack.close()

def preview_raster_png(raster_path):
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1).astype("float32")
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        im = ax.imshow(arr, cmap="viridis")
        ax.set_title(Path(raster_path).name)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return fig

def export_bundle(model, feature_names, config, tables):
    bundle = io.BytesIO()
    with io.BytesIO() as m:
        joblib.dump({"model": model, "features": feature_names, "config": config}, m)
        model_bytes = m.getvalue()

    import zipfile
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.joblib", model_bytes)
        z.writestr("config.json", json.dumps(config, ensure_ascii=False, indent=2))
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                z.writestr(f"{name}.csv", df.to_csv(index=False))
    bundle.seek(0)
    return bundle

# ---------- sidebar ----------
with st.sidebar:

    st.markdown(f'<div class="soft-note">{bi("SHAP 为可选功能，默认关闭，以降低部署时的内存压力；仅在需要解释分析时再启用。", "SHAP is optional and disabled by default to reduce memory pressure during deployment; enable it only when interpretability is needed.")}</div>', unsafe_allow_html=True)
    enable_shap = st.checkbox("启用 SHAP 分析 / Enable SHAP analysis", value=False, help="默认关闭以节省在线部署内存 / Disabled by default to save memory in online deployment.")
    shap_sample_size = st.slider("SHAP 抽样数量 / SHAP sample size", min_value=50, max_value=500, value=150, step=25, help="仅在启用 SHAP 时使用 / Only used when SHAP is enabled.")
    st.title("MicroFragment Atlas")
    st.markdown('<div class="sidebar-note">A premium research app for environmental microplastic modelling. Upload a sampling table and optional TIFF predictor layers to fit, validate and project the model.</div>', unsafe_allow_html=True)
    csv_file = st.file_uploader("上传采样 CSV / Upload sampling CSV", type=["csv"])
    raster_files = st.file_uploader("Upload predictor TIFF files", type=["tif", "tiff"], accept_multiple_files=True)

    st.markdown("---")
    random_state = st.number_input("Random state", 1, 9999, 42, 1)
    n_estimators = st.slider("Number of trees", 100, 1000, 300, 50)
    cv_splits = st.slider("交叉验证折数 / CV splits", 3, 10, 5)
    cv_repeats = st.slider("重复次数 / CV repeats", 1, 5, 3)
    spatial_blocks = st.slider("空间分块数 / Spatial blocks", 3, 10, 5)
    shap_sample_size = st.slider("SHAP 抽样数量 / SHAP sample size", 50, 500, 200, 25)
    compute_perm = st.checkbox("Compute permutation importance", value=True)
    compute_shap = st.checkbox("Compute SHAP outputs", value=True)

# ---------- hero ----------
st.markdown(
    """
    <div class="hero">
      <div class="kicker">Interactive research application</div>
      <h1>Environmental microplastic fragmentation potential</h1>
      <p>Upload sample data and predictor rasters, optimize a spatially explicit random forest model, compare repeated and spatial cross-validation, interpret drivers with SHAP, and export regional GeoTIFF predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_data, tab_model, tab_validation, tab_interpret, tab_raster, tab_export = st.tabs(
    ["Overview", "Data audit", "Model", "Validation", "Interpretation", "栅格预测 / Raster prediction", "Export"]
)

if csv_file is None:
    with tab_overview:
        st.info("Upload a CSV from the sidebar to begin.")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="glass"><h4 class="section-title">Premium interface</h4><p class="tiny">A cleaner visual system designed more like a project app than a default dashboard.</p></div>', unsafe_allow_html=True)
        c2.markdown('<div class="glass"><h4 class="section-title">Full modelling flow</h4><p class="tiny">Includes predictor screening, tuning, repeated CV, spatial CV and SHAP interpretation.</p></div>', unsafe_allow_html=True)
        c3.markdown('<div class="glass"><h4 class="section-title">Raster-ready</h4><p class="tiny">Accepts TIFF predictors and exports regional prediction and uncertainty GeoTIFFs.</p></div>', unsafe_allow_html=True)
    st.stop()

df = load_csv(csv_file)

with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isna().sum().sum()))
    c4.metric("Uploaded TIFFs", len(raster_files) if raster_files else 0)
    st.subheader("Data preview")
    st.dataframe(df.head(12), width="stretch", height=420)

with tab_data:
    st.subheader("Dataset configuration")
    all_cols = list(df.columns)
    col1, col2, col3 = st.columns(3)
    default_target = all_cols.index("MPs") if "MPs" in all_cols else 0
    target = col1.selectbox("响应变量 / Response variable", all_cols, index=default_target)
    x_default = ([""] + all_cols).index("经度 / Longitude") if "经度 / Longitude" in all_cols else 0
    y_default = ([""] + all_cols).index("纬度 / Latitude") if "纬度 / Latitude" in all_cols else 0
    x_coord = col2.selectbox("X coordinate (optional)", [""] + all_cols, index=x_default)
    y_coord = col3.selectbox("Y coordinate (optional)", [""] + all_cols, index=y_default)

    possible_drop = [c for c in all_cols if c not in {target, x_coord, y_coord}]
    drop_cols = st.multiselect("Exclude columns from predictors", possible_drop, default=[])

    try:
        X_filt, y, coord_df = prepare_simple_training_data(df, target, x_coord or None, y_coord or None, drop_cols)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Rows used for modelling", len(X_filt))
        mc2.metric("Predictor count", X_filt.shape[1])
        mc3.metric("目标变量 / Target", target)
        st.markdown("**Predictors used**")
        st.write(", ".join(list(X_filt.columns)))
    except Exception as e:
        X_filt, y, coord_df = None, None, None
        st.error(f"Dataset configuration error: {e}")

run_model = tab_model.button("Run modelling workflow", type="primary", width="stretch")

if "results" not in st.session_state:
    st.session_state["results"] = None

if run_model:
    if X_filt is None:
        st.error("Please fix the dataset configuration first.")
    else:
        X = X_filt.values.astype("float32", copy=False)

        with st.spinner("Training fixed RandomForest model, validating performance and computing interpretation outputs..."):
            best_model = fit_simple_rf(X, y, n_estimators=n_estimators, random_state=random_state)
            cv_summary, mean_cv_r2 = cross_val_summary_for_fixed_model(X, y, n_estimators=n_estimators, random_state=random_state, cv_splits=cv_splits)
            rep_folds, rep_oof = evaluate_repeated_cv(best_model, X, y, random_state=random_state, cv_splits=cv_splits, cv_repeats=cv_repeats)
            spatial_folds, spatial_oof = evaluate_spatial_cv(best_model, X, y, coord_df, random_state=random_state, spatial_blocks=spatial_blocks)

            rf_imp = pd.DataFrame({
                "变量 / feature": X_filt.columns,
                "rf_importance": best_model.feature_importances_,
            }).sort_values("rf_importance", ascending=False).reset_index(drop=True)

            if compute_perm:
                perm_imp = compute_permutation_importance(best_model, X, y, list(X_filt.columns), random_state=random_state)
            else:
                perm_imp = pd.DataFrame(columns=["变量 / feature", "perm_importance_mean", "perm_importance_std"])

            if compute_shap:
                shap_X, shap_df, shap_imp = compute_shap_sample(best_model, X_filt, sample_size=shap_sample_size, random_state=random_state)
            else:
                shap_X = X_filt.head(0).copy()
                shap_df = pd.DataFrame(columns=X_filt.columns)
                shap_imp = pd.DataFrame(columns=["变量 / feature", "mean_abs_shap", "mean_shap"])

        st.session_state["results"] = {
            "target": target,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "drop_cols": drop_cols,
            "X_filt": X_filt,
            "y": y,
            "coord_df": coord_df,
            "best_model": best_model,
            "best_params": {"model_type": "Fixed RandomForestRegressor", "n_estimators": int(n_estimators), "random_state": int(random_state)},
            "best_cv_score": mean_cv_r2,
            "rep_folds": rep_folds,
            "rep_oof": rep_oof,
            "spatial_folds": spatial_folds,
            "spatial_oof": spatial_oof,
            "rf_imp": rf_imp,
            "perm_imp": perm_imp,
            "shap_X": shap_X,
            "shap_df": shap_df,
            "shap_imp": shap_imp,
            "col_report": pd.DataFrame({"变量 / feature": list(X_filt.columns), "kept": True, "reason": "used_in_model"}),
            "col_pairs": pd.DataFrame(),
            "raster_outputs": None,
        }

res = st.session_state["results"]

with tab_model:
    if res is None:
        st.info("Configure the data in the Data audit tab, then run the modelling workflow.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Response", res["target"])
        c2.metric("Mean CV R²", f'{res["best_cv_score"]:.3f}')
        c3.metric("Retained predictors", res["X_filt"].shape[1])
        st.markdown("**Model settings**")
        st.json(res["best_params"])

with tab_validation:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Repeated cross-validation")
        c1, c2 = st.columns([1.05, 0.95])
        rep_summary = res["rep_folds"][["R2", "RMSE", "MAE"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
        c1.dataframe(rep_summary, width="stretch")
        c2.pyplot(fig_observed_pred(res["rep_oof"], "predicted_oof", "Repeated-CV prediction"))
        st.dataframe(res["rep_folds"], width="stretch", height=260)

        st.subheader("Spatial cross-validation")
        if res["spatial_folds"] is None:
            st.warning("Spatial validation was not run because valid coordinate columns were not provided.")
        else:
            s1, s2 = st.columns([1, 1])
            spatial_summary = res["spatial_folds"][["R2", "RMSE", "MAE"]].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
            s1.dataframe(spatial_summary, width="stretch")
            s2.pyplot(fig_observed_pred(res["spatial_oof"], "predicted_spatial_oof", "Spatial-CV prediction"))

with tab_interpret:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        c1, c2 = st.columns(2)
        c1.pyplot(fig_barh(res["rf_imp"], "rf_importance", "变量 / feature", "Random forest importance", 15, "Importance"))
        if res["perm_imp"].empty:
            c2.info("Permutation importance was not computed.")
        else:
            c2.pyplot(fig_barh(res["perm_imp"], "perm_importance_mean", "变量 / feature", "置换重要性 / Permutation importance", 15, "Mean permutation importance"))

        if res["shap_imp"].empty:
            st.info("SHAP outputs were not computed.")
        else:
            st.pyplot(fig_barh(res["shap_imp"], "mean_abs_shap", "变量 / feature", "SHAP 重要性 / SHAP importance", 15, "Mean |SHAP|"))

            st.subheader("Value–SHAP explorer")
            feat = st.selectbox("Select a predictor", list(res["shap_X"].columns))
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.scatter(res["shap_X"][feat], res["shap_df"][feat], s=18, alpha=0.58)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Value–SHAP relationship: {feat}")
        st.pyplot(fig)

        with st.expander("Show interpretation tables"):
            t1, t2 = st.columns(2)
            t1.dataframe(res["perm_imp"], width="stretch", height=280)
            if res.get("shap_imp") is not None:
                t2.dataframe(res["shap_imp"], width="stretch", height=280)

with tab_raster:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Regional raster prediction")
        st.caption("Upload one TIFF per retained predictor. TIFF file names should match predictor names, optionally with the suffix 1000, for example sand1000.tif.")
        retained = list(res["X_filt"].columns)
        st.write("Retained predictors:", ", ".join(retained))

        if not raster_files:
            st.warning("No TIFF files uploaded yet.")
        else:
            raster_names = [f.name for f in raster_files]
            st.write("Uploaded TIFF files:", ", ".join(raster_names))

            if st.button("运行栅格预测并导出不确定性 / Run raster prediction and uncertainty export", width="stretch"):
                with st.spinner("Matching TIFF predictors, aligning rasters and generating outputs..."):
                    raster_tempdir, saved_rasters = save_uploaded_rasters_to_temp(raster_files)
                    try:
                        out_dir, pred_path, mean_path, std_path = run_raster_prediction(
                            res["best_model"],
                            retained,
                            saved_rasters,
                            res["X_filt"],
                            res["y"],
                            cv_splits=cv_splits,
                            random_state=random_state,
                        )
                        res["raster_outputs"] = {
                            "temp_rasters": str(raster_tempdir),
                            "out_dir": str(out_dir),
                            "prediction": str(pred_path),
                            "mean": str(mean_path),
                            "std": str(std_path),
                        }
                        st.success("Raster prediction completed.")
                    except Exception as e:
                        st.error(f"Raster prediction failed: {e}")

        if res.get("raster_outputs"):
            outputs = res["raster_outputs"]
            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(preview_raster_png(outputs["prediction"]))
                st.download_button(
                    "Download prediction.tif",
                    data=Path(outputs["prediction"]).read_bytes(),
                    file_name="prediction.tif",
                    mime="application/octet-stream",
                )
            with p2:
                st.pyplot(preview_raster_png(outputs["std"]))
                st.download_button(
                    "Download prediction_cv_std.tif",
                    data=Path(outputs["std"]).read_bytes(),
                    file_name="prediction_cv_std.tif",
                    mime="application/octet-stream",
                )
            st.download_button(
                "Download prediction_cv_mean.tif",
                data=Path(outputs["mean"]).read_bytes(),
                file_name="prediction_cv_mean.tif",
                mime="application/octet-stream",
            )

with tab_export:
    if res is None:
        st.info("Run the modelling workflow first.")
    else:
        st.subheader("Export model bundle")
        config = {
            "target": res["target"],
            "x_coord": res["x_coord"],
            "y_coord": res["y_coord"],
            "drop_cols": res["drop_cols"],
            "random_state": random_state,
            "cv_splits": cv_splits,
            "n_estimators": n_estimators,
            "cv_repeats": cv_repeats,
            "spatial_blocks": spatial_blocks,
        }
        bundle = export_bundle(
            res["best_model"],
            list(res["X_filt"].columns),
            config,
            {
                "feature_selection_report": res["col_report"],
                "collinearity_pairs": res["col_pairs"],
                "repeated_cv_metrics": res["rep_folds"],
                "repeated_cv_oof": res["rep_oof"],
                "spatial_cv_metrics": res["spatial_folds"],
                "spatial_cv_oof": res["spatial_oof"],
                "rf_importance": res["rf_imp"],
                "permutation_importance": res["perm_imp"],
                "shap_importance": res["shap_imp"],
            },
        )
        
        if "best_model" in locals() and "X_filt" in locals() and enable_shap:
            st.markdown("### SHAP feature explorer")
            st.caption("按需解释视图，仅处理所选变量以降低内存占用 / Optional interpretability view. Only selected variable is processed to keep memory usage low.")
            shap_feature = st.selectbox("选择一个变量进行 SHAP 解析 / Choose one variable for SHAP inspection", options=list(X_filt.columns), index=0)
            if st.button("生成所选变量的 SHAP 图 / Generate selected-variable SHAP", width="stretch"):
                with st.spinner("正在计算所选变量的 SHAP… / Computing SHAP for selected variable..."):
                    shap_single = compute_single_feature_shap(best_model, X_filt, shap_feature, max_rows=shap_sample_size, random_state=random_state)
                    fig = fig_single_feature_shap(shap_single, shap_feature)
                    st.pyplot(fig, width="stretch")
                    st.dataframe(shap_single.head(30), width="stretch", height=260)

        st.download_button(
            "Download model bundle (.zip)",
            data=bundle,
            file_name="microfragment_model_bundle.zip",
            mime="application/zip",
            width="stretch",
        )
        st.markdown("**Quick notes**")
        st.write(
            "The bundle includes the trained model, retained feature list, full configuration, and the key validation and interpretation tables. "
            "Raster GeoTIFF outputs are downloaded separately from the Raster prediction tab."
        )
