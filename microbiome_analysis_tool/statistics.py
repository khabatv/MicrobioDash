
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

def run_permanova(distance_matrix, metadata_df, treatment_column):
    try:
        results = permanova(distance_matrix, metadata_df, column=treatment_column)
        p_value, test_stat = results['p-value'], results['test statistic']
        interpretation = "statistically significant (p < 0.05)" if p_value < 0.05 else "not statistically significant (p >= 0.05)"
        color = 'green' if p_value < 0.05 else 'red'

        return html.Div([
            html.H4("Beta Diversity Significance (PERMANOVA)"),
            html.P(f"P-value: {p_value:.4f}"),
            html.P(f"Test Statistic (pseudo-F): {test_stat:.4f}"),
            html.P(f"The difference in community composition between groups is {interpretation}.", style={'color': color, 'fontWeight': 'bold'})
        ])
    except Exception as e:
        logger.error(f"Could not calculate PERMANOVA: {e}")
        return html.Div([html.H4("PERMANOVA Error"), html.P(f"Details: {e}")])

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

