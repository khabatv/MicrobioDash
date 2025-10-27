
import pandas as pd
import numpy as np
# --- Compatibility shim for NumPy < 2.0 (scikit-bio expects np.isdtype) ---
if not hasattr(np, "isdtype"):
    def _np_isdtype(dt, kind):
        try:
            dt = np.dtype(dt)
        except Exception:
            return False
        if kind in ("numeric", "number"):
            return np.issubdtype(dt, np.number)
        if kind == "bool":
            return np.issubdtype(dt, np.bool_)
        if kind == "integer":
            return np.issubdtype(dt, np.integer)
        if kind in ("floating", "float"):
            return np.issubdtype(dt, np.floating)
        # fallback: best-effort False
        return False
    np.isdtype = _np_isdtype
import scikit_posthocs as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
try:
    import pymc as pm
    import arviz as az
    _PYMC_AVAILABLE = True
except Exception:  # ImportError is fine, but this catches env issues too
    pm = None
    az = None
    _PYMC_AVAILABLE = False
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from skbio.stats.distance import permanova
from skbio.stats.composition import ancom
from dash import html
from .config import logger, CPU_CORES

def perform_pairwise_alpha_tests(alpha_df, treatment_col, p_adjust_method='fdr_bh'):
    groups = alpha_df[treatment_col].unique()
    if len(groups) < 2: return pd.DataFrame()

    pairs = list(combinations(groups, 2))
    results = []
    for group1, group2 in pairs:
        data1, data2 = alpha_df['Shannon'][alpha_df[treatment_col] == group1], alpha_df['Shannon'][alpha_df[treatment_col] == group2]
        if len(data1) > 0 and len(data2) > 0:
            stat, p_raw = mannwhitneyu(data1, data2, alternative='two-sided')
            results.append({'group1': group1, 'group2': group2, 'p_raw': p_raw})

    if not results: return pd.DataFrame()
    results_df = pd.DataFrame(results)
    reject, p_adjusted, _, _ = multipletests(results_df['p_raw'], alpha=0.05, method=p_adjust_method)
    results_df['p_adj'], results_df['significant'] = p_adjusted, reject
    return results_df

def _ensure_numpy_distance(distance_matrix, metadata_df=None):
    """
    Accepts a pandas DataFrame with the same index/columns or a numpy array.
    If DataFrame, aligns metadata to its index order.
    """
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values.astype(float, copy=False)
        if metadata_df is not None:
            metadata_df = metadata_df.loc[distance_matrix.index]
        return D, metadata_df
    D = np.asarray(distance_matrix, dtype=float)
    return D, metadata_df

def _drop_missing(metadata_df, factors):
    """Drop rows with NA in any factor and return kept index mask."""
    mask = ~metadata_df[factors].isnull().any(axis=1)
    return metadata_df.loc[mask], mask

def _gower_center(D):
    """
    Gower-centering of a full pairwise distance matrix.
    Returns G = -0.5 * J * D^2 * J
    """
    D2 = D**2
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    return -0.5 * (J @ D2 @ J)

def _onehot_design(metadata_df, cols, level_map=None):
    """
    Build a design matrix X (without intercept).
    - Categorical columns: one-hot with drop_first=True (treatment coding).
      Optional 'level_map' lets you enforce specific level order per column.
    - Numeric columns: included as is (float).
    Returns X (n x p), df_per_term (dict), and per-term column slices.
    """
    mats = []
    term_slices = {}
    df_per_term = {}
    start = 0

    for col in cols:
        x = metadata_df[col]
        if (pd.api.types.is_categorical_dtype(x) or x.dtype == object):
            if level_map and col in level_map:
                # enforce level order
                cats = level_map[col]
                x = pd.Categorical(x, categories=cats, ordered=True)
            else:
                # ensure deterministic order (sorted unique)
                x = pd.Categorical(x, categories=pd.unique(x.astype(str)), ordered=True)

            M = pd.get_dummies(x, drop_first=True)
            block = M.to_numpy(dtype=float, copy=False)
            p = block.shape[1]  # Df for this term
            if p == 0:
                # single level -> contributes nothing (skip)
                term_slices[col] = slice(start, start)
                df_per_term[col] = 0
                continue
            mats.append(block)
            term_slices[col] = slice(start, start + p)
            df_per_term[col] = p
            start += p
        else:
            block = x.to_numpy(dtype=float).reshape(-1, 1)
            mats.append(block)
            term_slices[col] = slice(start, start + 1)
            df_per_term[col] = 1
            start += 1

    X = np.concatenate(mats, axis=0 if len(mats) and mats[0].ndim == 1 else 1) if mats else np.zeros((len(metadata_df), 0))
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, df_per_term, term_slices

def _proj_hat(X):
    """
    Return projection (hat) matrix H for design X with an intercept.
    H = X_ (X_' X_)^+ X_'  where X_ = [1, X]
    Uses pinv for numerical stability.
    """
    n = X.shape[0]
    X_ = np.column_stack([np.ones((n, 1)), X])
    XtX_inv = np.linalg.pinv(X_.T @ X_, rcond=1e-12)
    H = X_ @ XtX_inv @ X_.T
    return H

def _perm_indices_within_strata(strata_series, rng):
    """
    Produce a permutation of indices that shuffles within each stratum (block).
    """
    codes = pd.Series(strata_series).astype('category').cat.codes.to_numpy()
    perm = np.arange(len(codes))
    for code in np.unique(codes):
        idx = np.where(codes == code)[0]
        perm[idx] = rng.permutation(idx)
    return perm

# ------------------------------
# Core: Multifactor PERMANOVA (marginal tests via Freedman–Lane)
# ------------------------------

def permanova_marginal(
    distance_matrix,
    metadata_df,
    factors,
    permutations=999,
    strata=None,
    random_state=None,
    level_map=None,
):
    """
    Multifactor PERMANOVA with marginal (Type III-like) tests via Freedman–Lane permutations.

    Parameters
    ----------
    distance_matrix : (n x n) numpy.ndarray or pandas.DataFrame
        Symmetric pairwise distance matrix. If DataFrame, index/columns should be sample IDs.
    metadata_df : pandas.DataFrame
        Rows indexed by sample IDs (will be aligned to distance if DataFrame was passed).
    factors : list[str]
        Column names in metadata_df to include in the model.
    permutations : int, default 999
        Number of permutations for p-values.
    strata : array-like or pandas Series, optional
        Blocking factor; permutations are restricted within levels of 'strata'.
    random_state : int or np.random.Generator, optional
        Seed or RNG for reproducibility.
    level_map : dict[str, list], optional
        Dict mapping factor -> ordered list of levels (for categorical factors),
        to enforce specific reference/contrast coding (match R if needed).

    Returns
    -------
    pandas.DataFrame with columns:
        term, Df, SS, R2, F, p_value
        (sorted by R2 descending)
    """
    # Align inputs
    D, metadata_df = _ensure_numpy_distance(distance_matrix, metadata_df)
    if metadata_df is None:
        raise ValueError("metadata_df must be provided when passing a numpy distance matrix.")

    # Drop missing rows in factors (like R does)
    metadata_df, mask = _drop_missing(metadata_df, factors)
    if isinstance(distance_matrix, pd.DataFrame):
        D = D[mask.values, :][:, mask.values]
    else:
        # assume already aligned; user must drop rows in D too if there are NAs
        pass

    n = D.shape[0]
    if n != len(metadata_df):
        raise ValueError("Distance matrix and metadata row counts do not match after NA dropping.")

    # Build design
    X_full, df_per_term, term_slices = _onehot_design(metadata_df, factors, level_map=level_map)

    # Edge case: if all columns ended up empty (e.g., single-level factors only)
    if X_full.shape[1] == 0:
        raise ValueError("All factors collapsed (single level). No testable terms.")

    # Projections and Gower-centered matrix
    G = _gower_center(D)
    SS_tot = np.trace(G)

    H_full = _proj_hat(X_full)

    rng = np.random.default_rng(random_state)

    # Prepare strata as Series aligned to metadata if provided
    strata_series = None
    if strata is not None:
        if isinstance(strata, (pd.Series, pd.Categorical)):
            strata_series = strata.loc[metadata_df.index] if hasattr(strata, "index") else pd.Series(strata, index=metadata_df.index)
        else:
            # assume array-like aligned to metadata_df
            strata_series = pd.Series(strata, index=metadata_df.index)

    results = []

    # Precompute residual DoF
    df_resid = n - (X_full.shape[1] + 1)  # +1 for intercept
    if df_resid < 1:
        raise ValueError("Residual degrees of freedom < 1. Model is saturated.")

    # For each term, compute marginal effect (full vs. reduced model without that term)
    for term in factors:
        p_term = df_per_term.get(term, 0)
        if p_term == 0:
            # no degrees for this term (e.g., only one level) -> skip with zeros
            results.append({"term": term, "Df": 0, "SS": 0.0, "R2": 0.0, "F": np.nan, "p_value": 1.0})
            continue

        # Reduced design: drop columns of this term
        cols_keep = np.ones(X_full.shape[1], dtype=bool)
        cols_keep[term_slices[term]] = False
        X_others = X_full[:, cols_keep] if cols_keep.any() else np.zeros((n, 0))
        H_others = _proj_hat(X_others) if X_others.shape[1] > 0 else np.zeros((n, n))

        # Sums of squares
        SS_full   = np.trace(H_full   @ G @ H_full)
        SS_others = np.trace(H_others @ G @ H_others)
        SS_term   = SS_full - SS_others

        MS_term = SS_term / p_term
        MS_res  = (SS_tot - SS_full) / df_resid
        F_obs   = MS_term / MS_res if MS_res > 0 else np.inf
        R2      = SS_term / SS_tot if SS_tot > 0 else np.nan

        # Freedman–Lane permutations under reduced model
        # Residualized G under reduced model
        I = np.eye(n)
        R = (I - H_others) @ G @ (I - H_others)

        # Spectral decomposition to get residual coordinates for shuffling
        eigvals, eigvecs = np.linalg.eigh(R)
        pos = eigvals > 1e-12
        if not np.any(pos):
            # If residual space is nearly zero, permutations won't change anything
            p_value = 1.0
        else:
            U = eigvecs[:, pos] * np.sqrt(eigvals[pos])

            exceed = 0
            for _ in range(permutations):
                if strata_series is None:
                    perm = rng.permutation(n)
                else:
                    perm = _perm_indices_within_strata(strata_series, rng)

                U_perm = U[perm, :]                 # permute residual coords
                G_star = U_perm @ U_perm.T          # reconstructed permuted residual

                # Recompose permuted G under the alternative:
                G_perm = (H_others @ G @ H_others) + ((I - H_others) @ G_star @ (I - H_others))

                SS_full_p   = np.trace(H_full   @ G_perm @ H_full)
                SS_others_p = np.trace(H_others @ G_perm @ H_others)
                SS_term_p   = SS_full_p - SS_others_p

                MS_term_p = SS_term_p / p_term
                MS_res_p  = (SS_tot - SS_full_p) / df_resid
                F_p       = MS_term_p / MS_res_p if MS_res_p > 0 else np.inf

                if F_p >= F_obs:
                    exceed += 1

            # add-one correction
            p_value = (exceed + 1) / (permutations + 1)

        results.append({"term": term, "Df": int(p_term), "SS": float(SS_term), "R2": float(R2), "F": float(F_obs), "p_value": float(p_value)})

    out = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return out

# ------------------------------
# (Optional) quick permdisp (betadisper) analog
# ------------------------------

def betadisper_anova(distance_matrix, groups):
    """
    Simple homogeneity-of-dispersion test:
    1) Compute distances to each group's centroid in principal coordinate space (from Gower-centered distances).
    2) One-way ANOVA on distances; F and p via permutations.

    Returns pandas DataFrame with F and p-value (overall), plus group means.
    """
    if isinstance(distance_matrix, pd.DataFrame):
        D = distance_matrix.values.astype(float, copy=False)
        groups = pd.Series(groups).loc[distance_matrix.index]
    else:
        D = np.asarray(distance_matrix, dtype=float)
        groups = pd.Series(groups)

    G = _gower_center(D)
    eigvals, eigvecs = np.linalg.eigh(G)
    pos = eigvals > 1e-12
    if not np.any(pos):
        raise ValueError("No positive eigenvalues found for PCoA space.")
    X = eigvecs[:, pos] * np.sqrt(eigvals[pos])  # PCoA coordinates

    groups = pd.Series(groups).astype('category')
    codes = groups.cat.codes.to_numpy()
    levels = list(groups.cat.categories)

    # Group centroids in Euclidean PCoA
    centroids = np.vstack([X[codes == k].mean(axis=0) for k in range(len(levels))])
    dists = np.sqrt(((X - centroids[codes])**2).sum(axis=1))

    # One-way ANOVA components
    grand_mean = dists.mean()
    n = len(dists)
    k = len(levels)
    ss_total = ((dists - grand_mean)**2).sum()
    ss_between = sum([((dists[codes == i].mean() - grand_mean)**2) * (codes == i).sum()
                      for i in range(k)])
    ss_within = ss_total - ss_between
    df_between = k - 1
    df_within = n - k
    ms_between = ss_between / df_between if df_between > 0 else np.nan
    ms_within = ss_within / df_within if df_within > 0 else np.nan
    F_obs = ms_between / ms_within

    # Permutation p-value by shuffling group labels
    rng = np.random.default_rng(0)
    perms = 999
    exceed = 0
    for _ in range(perms):
        perm_codes = rng.permutation(codes)
        centroids_p = np.vstack([X[perm_codes == i].mean(axis=0) for i in range(k)])
        dists_p = np.sqrt(((X - centroids_p[perm_codes])**2).sum(axis=1))
        grand_mean_p = dists_p.mean()
        ss_between_p = sum([((dists_p[perm_codes == i].mean() - grand_mean_p)**2) * (perm_codes == i).sum()
                            for i in range(k)])
        ms_between_p = ss_between_p / df_between if df_between > 0 else np.nan
        # reuse ms_within under permutation for speed? better recompute:
        ss_total_p = ((dists_p - grand_mean_p)**2).sum()
        ss_within_p = ss_total_p - ss_between_p
        ms_within_p = ss_within_p / df_within if df_within > 0 else np.nan
        F_p = ms_between_p / ms_within_p
        if F_p >= F_obs:
            exceed += 1
    p_value = (exceed + 1) / (perms + 1)

    return pd.DataFrame({
        "stat": ["F", "p_value"],
        "value": [float(F_obs), float(p_value)]
    }), pd.DataFrame({"group": levels, "mean_distance": [dists[codes == i].mean() for i in range(k)]})

def run_differential_abundance(ps1_object, treatment_column):
    try:
        logger.info("Running differential abundance analysis with Kruskal-Wallis and Dunn's post-hoc test...")
        asv_rel = ps1_object['asv'].div(ps1_object['asv'].sum(axis=0), axis=1) # Normalize by sample
        taxa, meta = ps1_object['tax'], ps1_object['meta']
        groups = meta[treatment_column].unique()

        if len(groups) < 2:
            return html.P("At least two groups are needed for this analysis.")

        order_abundance = asv_rel.T.join(taxa['Order']).groupby('Order').sum().T.dropna(axis=1)

        significant_orders_kw = []
        for order in order_abundance.columns:
            grouped_values = [order_abundance[order][meta[treatment_column] == g] for g in groups]
            if all(len(v) > 0 for v in grouped_values):
                try:
                    stat, p_raw = kruskal(*grouped_values)
                    if p_raw < 0.05:
                        significant_orders_kw.append(order)
                except ValueError:
                    continue

        if not significant_orders_kw:
            return html.Div([
                html.H4("Differential Abundance by Order"),
                html.P("No Orders were found to be significantly different in overall abundance across groups.")
            ])

        final_results = []
        for order in significant_orders_kw:
            order_data_df = pd.DataFrame({'Abundance': order_abundance[order], 'Group': meta[treatment_column]})
            dunn_results = sp.posthoc_dunn(order_data_df, val_col='Abundance', group_col='Group', p_adjust='fdr_bh')

            sig_pairs = dunn_results.stack().reset_index()
            sig_pairs.columns = ['Group 1', 'Group 2', 'p_adj']
            sig_pairs = sig_pairs[sig_pairs['p_adj'] < 0.05]

            for _, row in sig_pairs.iterrows():
                final_results.append({
                    'Taxonomic Order': order,
                    'Comparison': f"{row['Group 1']} vs {row['Group 2']}",
                    'Adjusted p-value': f"{row['p_adj']:.4f}"
                })

        if not final_results:
            return html.Div([
                html.H4("Differential Abundance by Order"),
                html.P("Found Orders with overall significance, but no specific pairwise differences were significant after post-hoc correction.")
            ])

        final_df = pd.DataFrame(final_results)
        table = html.Table([html.Thead(html.Tr([html.Th(col) for col in final_df.columns]))] + [html.Tbody([html.Tr([html.Td(final_df.iloc[i][col]) for col in final_df.columns]) for i in range(len(final_df))])], style={'marginLeft': 'auto', 'marginRight': 'auto', 'marginTop': '20px'})
        return html.Div([html.H4("Significant Pairwise Differences in Abundance (by Order)"), table])
    except Exception as e:
        logger.error(f"Differential abundance analysis failed: {e}", exc_info=True)
        return html.P(f"Error during differential abundance analysis: {e}")

def run_indicator_species(ps1_object, treatment_column, n_permutations=999):
    try:
        logger.info("Running Indicator Species Analysis...")
        asv_pa_table = (ps1_object['asv'] > 0).astype(int).T
        groups, taxa = ps1_object['meta'][treatment_column], ps1_object['tax']
        unique_groups = sorted(groups.unique())
        if len(unique_groups) < 2: return html.P("At least two groups are needed."), pd.DataFrame()

        indicator_results = []
        for asv in asv_pa_table.columns:
            target_group, max_indval = None, -1
            for group in unique_groups:
                target_samples = groups.index[groups == group]
                if len(target_samples) == 0: continue
                in_target_count, total_count = asv_pa_table.loc[target_samples, asv].sum(), asv_pa_table[asv].sum()
                if total_count == 0: continue
                specificity, fidelity = in_target_count / total_count, in_target_count / len(target_samples)
                indval_score = specificity * fidelity
                if indval_score > max_indval:
                    max_indval, target_group = indval_score, group

            if max_indval <= 0: continue
            perm_stats = []
            for _ in range(n_permutations):
                perm_groups = np.random.permutation(groups)
                perm_target_samples = groups.index[perm_groups == target_group]
                if len(perm_target_samples) == 0:
                    perm_stats.append(0)
                    continue
                perm_in_target = asv_pa_table.loc[perm_target_samples, asv].sum()
                perm_spec, perm_fid = perm_in_target / total_count, perm_in_target / len(perm_target_samples)
                perm_stats.append(perm_spec * perm_fid)
            p_value = (np.sum(np.array(perm_stats) >= max_indval) + 1) / (n_permutations + 1)
            if p_value < 0.05:
                indicator_results.append({'ASV': asv, 'Associated Group': target_group, 'Indicator Score': max_indval, 'p_value': p_value})

        if not indicator_results: return html.Div([html.H4("Indicator ASV Analysis"), html.P("No significant indicator ASVs found.")]), pd.DataFrame()

        results_df = pd.DataFrame(indicator_results).sort_values('Indicator Score', ascending=False)
        significant_indicators_with_taxa = results_df.merge(taxa, left_on='ASV', right_index=True)
        significant_indicators_with_taxa.fillna('', inplace=True)
        significant_indicators_with_taxa['Taxon Name'] = significant_indicators_with_taxa['Genus'] + ' ' + significant_indicators_with_taxa['Species']
        significant_indicators_with_taxa['Taxon Name'] = significant_indicators_with_taxa['Taxon Name'].str.strip().replace('', 'Unclassified')

        df_for_display = significant_indicators_with_taxa[['Associated Group', 'Indicator Score', 'p_value', 'Taxon Name']].round(4).head(25)
        table = html.Table([html.Thead(html.Tr([html.Th(col) for col in df_for_display.columns]))] + [html.Tbody([html.Tr([html.Td(df_for_display.iloc[i][col]) for col in df_for_display.columns]) for i in range(len(df_for_display))])])
        return html.Div([html.H4("Indicator ASV Analysis (Top 25)"), table]), significant_indicators_with_taxa
    except Exception as e:
        logger.error(f"Indicator Species Analysis failed: {e}", exc_info=True)
        return html.P("Error during Indicator Species Analysis."), pd.DataFrame()
def _ancom_direction(tbl_samples_x_features: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """
    Infer direction by CLR means per group: for each feature, which group is highest/lowest.
    Assumes tbl already has a small pseudocount added.
    """
    # CLR transform
    gm = np.exp(np.log(tbl_samples_x_features).mean(axis=1))  # geometric mean per sample
    clr = np.log(tbl_samples_x_features.div(gm, axis=0))

    # Average per group
    clr_means = clr.groupby(groups).mean()
    top = clr_means.idxmax(axis=0)
    bottom = clr_means.idxmin(axis=0)
    return pd.DataFrame({"group_highest": top, "group_lowest": bottom})


def run_ancom_skbio(ps, group_col, alpha=0.05, zero_pseudocount=1, add_clr_means=True):
    """
    Run ANCOM (scikit-bio) and return a table with:
      - Feature_ID, W, reject
      - group_highest / group_lowest (based on CLR means)
      - One column per treatment level with the CLR group mean (if add_clr_means=True)
      - (optional) taxonomy columns merged if available in ps['tax'].

    Notes:
      * ANCOM uses log-ratios internally; per-group columns are **CLR means** (log scale).
      * Higher CLR mean ~ relatively more abundant.
    """
    # ------- Prepare data: samples x features
    tbl = ps['asv'].T.copy()           # samples x features
    meta = ps['meta'].copy()

    # Align and validate grouping column
    meta = meta.loc[meta.index.intersection(tbl.index)]
    if group_col not in meta.columns:
        return pd.DataFrame(columns=['Feature_ID', 'W', 'reject', 'alpha'])
    meta = meta[~meta[group_col].isna()]
    tbl  = tbl.loc[meta.index]
    grp  = meta[group_col].astype('category')

    if grp.nunique() < 2:
        return pd.DataFrame(columns=['Feature_ID', 'W', 'reject', 'alpha'])

    # Pseudocount → avoid log(0)
    if zero_pseudocount and zero_pseudocount > 0:
        tbl = tbl + zero_pseudocount

    # Drop constant (no variability) features
    const_cols = tbl.columns[(tbl.nunique(dropna=False) <= 1)]
    if len(const_cols):
        tbl = tbl.drop(columns=const_cols)

    if tbl.shape[1] == 0:
        return pd.DataFrame(columns=['Feature_ID', 'W', 'reject', 'alpha'])

    # ------- Compute CLR (for interpretation & per-group means)
    # CLR(x) = log(x) - mean(log(x)) per sample
    log_tbl = np.log(tbl)
    clr_tbl = log_tbl.sub(log_tbl.mean(axis=1), axis=0)  # samples x features

    # Group-wise CLR means (features x groups)
    # These are the values you can plot later
    group_means = clr_tbl.groupby(grp).mean().T  # index: features, columns: groups

    # Highest / lowest group per feature (by CLR mean)
    group_highest = group_means.idxmax(axis=1)
    group_lowest  = group_means.idxmin(axis=1)

    # ------- Run ANCOM
    rejections, W = ancom(tbl, grouping=grp, alpha=alpha, p_adjust='holm')

    # Normalize outputs to 1-D Series with feature index
    feature_index = tbl.columns

    def _ensure_series(x, default_index):
        if isinstance(x, pd.Series):
            return x.reindex(default_index)
        if isinstance(x, pd.DataFrame):
            # try common names, else first column
            for c in ('reject', 'W'):
                if c in x.columns:
                    return x[c].reindex(default_index)
            if x.shape[1] >= 1:
                return x.iloc[:, 0].reindex(default_index)
            return pd.Series(index=default_index, dtype=float)
        arr = np.asarray(x).ravel()
        if arr.shape[0] != len(default_index):
            arr = arr[:len(default_index)]
        return pd.Series(arr, index=default_index)

    rej  = _ensure_series(rejections, feature_index)
    Wser = _ensure_series(W,          feature_index)

    # ------- Build result table
    res = pd.DataFrame({
        'Feature_ID': feature_index,
        'W':          Wser.values,
        'reject':     rej.values.astype(bool),
        'group_highest': group_highest.reindex(feature_index).values,
        'group_lowest':  group_lowest.reindex(feature_index).values,
    })

    # Add compact ASV IDs
    res.insert(0, "ASV", [f"ASV_{i+1:03d}" for i in range(len(res))])

    # Merge taxonomy if present
    tax = ps.get('tax')
    if isinstance(tax, pd.DataFrame):
        res = res.merge(tax, left_on='Feature_ID', right_index=True, how='left')

    # Append per-group CLR means (one column per treatment)
    if add_clr_means:
        gm = group_means.copy()
        # Make nice, unique column names: e.g. CLR_mean::<GroupName>
        gm.columns = [f"CLR_mean::{str(c)}" for c in gm.columns]
        res = res.merge(gm, left_on='Feature_ID', right_index=True, how='left')

    res['alpha'] = alpha
    res = res.sort_values('W', ascending=False).reset_index(drop=True)

    return res
def run_mixed_effect_model(ps1_object, treatment_col, random_effect_cols, time_col,
                   analysis_level, top_n_features=20, reference_group=None, force_features=None):
    logger.info(f"Running Negative Binomial GEE at the {analysis_level} level...")
    try:
        asv_table, metadata, taxa = ps1_object['asv'].astype(int), ps1_object['meta'], ps1_object['tax']

        required_cols = [treatment_col]
        if random_effect_cols: required_cols.extend(random_effect_cols)
        use_time_col = time_col and time_col.strip() and time_col in metadata.columns
        if use_time_col: required_cols.append(time_col)
        if any(col not in metadata.columns for col in required_cols if col):
            raise ValueError(f"Missing columns in metadata: {[c for c in required_cols if c not in metadata.columns]}")

        model_formula = f"abundance ~ C({treatment_col}, Treatment('{reference_group}'))" if reference_group else f"abundance ~ {treatment_col}"
        if use_time_col: model_formula += f" * {time_col}"

        if analysis_level == 'ASV':
            feature_table = asv_table.T
        else:
            feature_table = asv_table.T.join(taxa[analysis_level]).groupby(analysis_level).sum()

        top_features = feature_table.sum(axis=0).nlargest(top_n_features).index
        if force_features:
            valid_forced = [f for f in force_features if f in feature_table.columns]
            top_features = pd.Index(list(set(top_features).union(set(valid_forced))))

        all_results = []
        for feature_id in top_features:
            df_long = metadata[list(set(required_cols))].join(feature_table[feature_id].rename('abundance')).dropna()
            if df_long.empty or df_long['abundance'].var() == 0: continue
            try:
                gee_model = smf.gee(
                    formula=model_formula,
                    groups=df_long[random_effect_cols[0]] if random_effect_cols else df_long.index, # GEE takes one grouping var
                    data=df_long,
                    cov_struct=sm.cov_struct.Exchangeable(),
                    family=sm.families.NegativeBinomial()
                )
                gee_results = gee_model.fit()
                pvals, coeffs, conf_int = gee_results.pvalues, gee_results.params, gee_results.conf_int()
                for var in pvals.index:
                    all_results.append({
                        'Feature_ID': feature_id, 'Variable': var, 'Coefficient': coeffs[var],
                        'P-value': pvals[var], 'Conf. Int. Lower': conf_int.loc[var, 0],
                        'Conf. Int. Upper': conf_int.loc[var, 1], 'Significant': pvals[var] < 0.05
                    })
            except Exception as e:
                logger.warning(f"FAILED to fit GEE for {feature_id}. Reason: {e}")

        if not all_results: return pd.DataFrame()
        results_df = pd.DataFrame(all_results)

        if analysis_level == 'ASV':
            return results_df.merge(taxa, left_on='Feature_ID', right_index=True, how='left')
        else:
            rep_taxa = taxa.groupby(taxa[analysis_level]).first()
            return results_df.merge(rep_taxa, left_on='Feature_ID', right_index=True, how='left')
    except Exception as e:
        logger.error(f"CRITICAL ERROR in mixed-effect model: {e}", exc_info=True)
        return pd.DataFrame()

def run_pymc_zinb_mixed_model(ps, mem_treat, rand_eff, time_col, analysis_lvl, mem_top_n, mem_ref, force_feat):
    if not _PYMC_AVAILABLE:
        raise RuntimeError(
            "PyMC is not available in this environment. "
            "Switch the 'Model Type' to 'GEE' in the UI, "
            "or install the optional dependencies: "
            "pip install 'pymc>=5' 'arviz>=0.16' 'cachetools>=5'"
        )
    # In a real scenario, the full PyMC implementation would go here.
    # For now, we return an empty DataFrame to prevent errors.
    return pd.DataFrame()

