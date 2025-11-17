
import os
import io
import base64
import tempfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ete3 import Tree, TreeStyle, NodeStyle, CircleFace, TextFace
from skbio import DistanceMatrix
from skbio.tree import nj as skbio_nj
from Bio import Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from dash import html
from .config import logger
# ---------- Publication style helpers ----------
from plotly.colors import qualitative as qual
from typing import Optional, List, Dict, Tuple
import plotly.colors as pc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_pub_palette(n: int) -> list:
    """
    Returns up to `n` distinct, publication-quality colors.
    Automatically expands if more colors are needed.
    """

    # ✅ 1. Combine safe qualitative palettes from Plotly
    base_palettes = (
        pc.qualitative.Plotly
        + pc.qualitative.D3
        + pc.qualitative.G10
        + pc.qualitative.Set3
        + pc.qualitative.Light24
        + pc.qualitative.Dark24
    )

    # Remove duplicates while preserving order
    seen = set()
    base_palettes = [c for c in base_palettes if not (c in seen or seen.add(c))]

    # ✅ 2. If enough colors exist, return subset
    if n <= len(base_palettes):
        return base_palettes[:n]

    # ✅ 3. Otherwise generate smooth extended palette from matplotlib colormap
    cmap = plt.get_cmap("turbo")  # vivid and publication-safe
    extra = [mcolors.to_hex(cmap(i / n)) for i in range(n)]
    return extra

def apply_pub_style(fig, portrait=True, base_font=16, legend_right=True,
                    width=None, height=None):
    """Consistent journal-style layout with optional explicit width/height."""
    # sensible defaults
    if width is None:
        width  = 900 if portrait else 1100   # was 700/1000
    if height is None:
        height = 900 if portrait else 600

    fig.update_layout(
        width=width, height=height,
        font=dict(size=base_font),
        margin=dict(l=90, r=160, t=60, b=90),
        legend_title_text=''
    )

    if legend_right:
        fig.update_layout(
            legend=dict(
                orientation='v',
                y=0.5, yanchor='middle',
                x=1.02, xanchor='left',
                bgcolor='rgba(255,255,255,0.6)'
            )
        )

    fig.update_xaxes(tickangle=35, automargin=True)
    fig.update_yaxes(automargin=True)

    return fig
# helper: get plotting order from metadata ------------------------------------
def _infer_group_order_from_meta(meta_df: pd.DataFrame, treat_col: str):
    s = meta_df[treat_col]

    # 1) ordered categorical -> use categories
    if pd.api.types.is_categorical_dtype(s) and getattr(s.cat, "ordered", False):
        return list(s.cat.categories)

    # 2) look for an explicit numeric "order" column per group
    #    (accept several common names or "<treat_col>_order")
    candidate_cols = [
        f"{treat_col}_order", "PlotOrder", "GroupOrder", "order", "Order",
        "Run", "Day", "Time"
    ]
    for c in candidate_cols:
        if c in meta_df.columns:
            # keep only numeric-like
            try:
                ord_series = pd.to_numeric(meta_df[c], errors="coerce")
                if ord_series.notna().any():
                    order = (
                        meta_df.assign(__ord=ord_series)
                               .groupby(treat_col)["__ord"].min()  # min per group
                               .sort_values()
                               .index.tolist()
                    )
                    if len(order) > 0:
                        return order
            except Exception:
                pass

    # 3) fallback
    return sorted(pd.unique(s.astype(str)))
def consistent_color_mapping(labels, order=None):
    """
    Return a {label: color} map.
    - If `order` is given, use that exact order.
    - Else, preserve first-seen order in `labels` (not alphabetical).
    """
    if order is None:
        # preserve appearance order, not sorted()
        unique = list(dict.fromkeys(list(labels)))
    else:
        unique = list(order)

    palette = get_pub_palette(len(unique))
    return dict(zip(unique, palette))
def _compact_label(x: str) -> str:
    """
    Creates a shortened, human-readable label from treatment IDs.
    Example: 'UC_PET_2_50%_Brine_Plate_T' → 'U. compressa · PET · 50% Brine · Plate'
    """
    # Simplify frequent patterns
    x = x.replace("_1/UCM", " · UCM")
    x = x.replace("_1/TM", " · TM")
    x = x.replace("_2/50% Brine", " · 50% Brine")
    x = x.replace("_3/Brine", " · Brine")
    x = x.replace("_Brine", " · Brine")
    x = x.replace("_Plate_T", " · Plate")
    x = x.replace("_Plate_S", " · Plate")
    x = x.replace("_Tank_T", " · Tank")

    # Replace remaining underscores with spaces
    x = x.replace("_", " ")

    return x.strip()

def _apply_labels_and_order(
    df: pd.DataFrame,
    col: str,
    meta: pd.DataFrame,
    order_mode: str = "meta",                 # "meta" | "alpha" | "first" | "custom"
    custom_order: Optional[List[str]] = None,
    label_map: Optional[Dict[str, str]] = None,  # {"old": "Pretty", ...}
    auto_compact: bool = True
):
    """
    Unifies order + label handling for all plots.
    - Renames values in `col` using label_map (if given) and/or _compact_label().
    - Computes present_order based on order_mode.
    Returns: (df2, present_order, label_map_used)
    """
    df2 = df.copy()

    # --- determine base order from metadata (ordered categorical wins) ---
    if col in meta.columns:
        s = meta[col]
        if pd.api.types.is_categorical_dtype(s) and getattr(s.cat, "ordered", False):
            meta_order = list(s.cat.categories)
        else:
            meta_order = list(dict.fromkeys(s.astype(str).tolist()))
    else:
        meta_order = []

    vals = df2[col].astype(str)
    if order_mode == "alpha":
        present_order = sorted(vals.unique().tolist())
    elif order_mode == "first":
        present_order = list(dict.fromkeys(vals.tolist()))
    elif order_mode == "custom" and custom_order:
        present_order = [x for x in custom_order if x in vals.unique()]
    else:  # "meta" default
        present_order = [g for g in meta_order if g in vals.unique()]
        if not present_order:
            present_order = sorted(vals.unique().tolist())

    # --- build unified label map ---
    label_map_used: Dict[str, str] = {}
    for v in vals.unique():
        pretty = label_map[v] if (label_map and v in label_map) else v
        if auto_compact:
            pretty = _compact_label(pretty)
        label_map_used[v] = pretty

    df2[col] = vals.map(lambda x: label_map_used.get(x, x))
    present_order = [label_map_used.get(x, x) for x in present_order if x in label_map_used]

    return df2, present_order, label_map_used


# -----------------------------------------------------------------------------
def add_stat_annotations(fig, alpha_df, treatment_col, stats_df, group_order=None):
    if group_order is None:
        group_order = _infer_group_order_from_meta(alpha_df, treatment_col)
    group_positions = {g: i for i, g in enumerate(group_order)}
    y_max = alpha_df['Shannon'].max()
    y_step = y_max * 0.15
    y_current = y_max + y_step
    significant_pairs = stats_df[stats_df['significant']].sort_values('p_adj')

    for _, row in significant_pairs.iterrows():
        x1, x2 = group_positions.get(row['group1']), group_positions.get(row['group2'])
        if x1 is None or x2 is None:
            continue
        p_text = "p < 0.001" if row['p_adj'] < 0.001 else f"p = {row['p_adj']:.3f}"
        fig.add_shape(type="line", x0=x1, y0=y_current, x1=x2, y1=y_current, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=x1, y0=y_current*0.99, x1=x1, y1=y_current, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=x2, y0=y_current*0.99, x1=x2, y1=y_current, line=dict(color='black', width=1))
        fig.add_annotation(x=(x1 + x2) / 2, y=y_current + (y_step*0.1), text=p_text, showarrow=False)
        y_current += y_step

    fig.update_xaxes(categoryorder="array", categoryarray=group_order)
    fig.update_yaxes(range=[alpha_df['Shannon'].min()*0.9, y_current])
    return fig

def plot_phylogenetic_tree(seqtab, taxa, indicator_df=None):
    try:
        unique_seqs = {seq: SeqRecord(Seq(seq), id=seq) for seq in seqtab.columns}
        if len(unique_seqs) < 4:
            logger.warning("Cannot generate a tree with fewer than 4 unique sequences.")
            return None

        seqs_for_tree = list(unique_seqs.values())
        names = [s.id for s in seqs_for_tree]
        dm_data = np.zeros((len(seqs_for_tree), len(seqs_for_tree)))
        aligner = Align.PairwiseAligner(mode='global')
        for i in range(len(seqs_for_tree)):
            for j in range(i + 1, len(seqs_for_tree)):
                score = aligner.align(seqs_for_tree[i].seq, seqs_for_tree[j].seq).score
                max_len = max(len(seqs_for_tree[i].seq), len(seqs_for_tree[j].seq))
                distance = 1 - (score / max_len) if max_len > 0 else 1
                dm_data[i, j] = dm_data[j, i] = distance

        dm = DistanceMatrix(dm_data, ids=names)
        skbio_tree = skbio_nj(dm)
        handle = io.StringIO()
        skbio_tree.write(handle, format='newick')
        ete_tree = Tree(handle.getvalue())

        top_phyla = taxa['Phylum'].value_counts().nlargest(10).index
        colors = px.colors.qualitative.Plotly
        phylum_colors = {phylum: colors[i % len(colors)] for i, phylum in enumerate(top_phyla)}

        for leaf in ete_tree.iter_leaves():
            asv_seq = leaf.name
            nstyle = NodeStyle(size=8, fgcolor="black")
            if asv_seq in taxa.index:
                phylum = taxa.loc[asv_seq, 'Phylum']
                nstyle["bgcolor"] = phylum_colors.get(phylum, "lightgrey")
                if indicator_df is not None and not indicator_df.empty and asv_seq in indicator_df['ASV'].values:
                    nstyle["fgcolor"], nstyle["size"] = "red", 12
            leaf.set_style(nstyle)
            leaf.name = f"ASV_{taxa.index.get_loc(asv_seq) + 1}"

        ts = TreeStyle()
        ts.mode = "c"                 # "c" = circular; use "r" for rectangular
        ts.scale = 20
        ts.branch_vertical_margin = 10
        ts.show_leaf_name = True
        for phylum, color in phylum_colors.items():
            if pd.notna(phylum):
                ts.legend.add_face(CircleFace(10, color), column=0)
                ts.legend.add_face(TextFace(f" {phylum}", fsize=10), column=1)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmp_file_path = f.name
        ete_tree.render(tmp_file_path, w=1200, units='px', tree_style=ts)
        with open(tmp_file_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        os.remove(tmp_file_path)
        logger.info("Generated phylogenetic tree successfully.")
        return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        logger.error(f"Error plotting phylogenetic tree: {e}", exc_info=True)
        return None

def plot_abundance_by_order(ps1_object, treatment_column, threshold=0.01, group_order=None):
    """
    Stacked bar of mean relative abundance by treatment, aggregated at Order.
    Collapses rare Orders (< threshold mean share) into 'Other'.
    Respects an explicit `group_order` if provided.
    """
    try:
        asv_table = ps1_object['asv']  # ASVs x Samples
        tax_df    = ps1_object['tax']  # taxonomy (index = ASV)
        meta      = ps1_object['meta'] # metadata (index = SampleID)

        melted_df = (
            asv_table
            .stack()
            .reset_index(name='Abundance')
            .rename(columns={'level_0': 'ASV', 'level_1': 'SampleID'})
            .merge(tax_df[['Order']], left_on='ASV', right_index=True, how='left')
            .merge(meta[[treatment_column]], left_on='SampleID', right_index=True, how='left')
        )
        melted_df['Order'] = melted_df['Order'].fillna('Unassigned')

        grp = melted_df.groupby([treatment_column, 'Order'])['Abundance'].sum().reset_index()
        totals = grp.groupby(treatment_column)['Abundance'].transform('sum')
        grp['RelativeAbundance'] = grp['Abundance'] / totals

        mean_by_order = grp.groupby('Order')['RelativeAbundance'].mean()
        rare_orders = mean_by_order[mean_by_order < threshold].index
        grp['Order'] = grp['Order'].where(~grp['Order'].isin(rare_orders), 'Other')

        final_df = (
            grp.groupby([treatment_column, 'Order'])['RelativeAbundance']
               .sum().reset_index()
        )

        # x-axis order ("present")
        if group_order:
            present = [g for g in group_order if g in final_df[treatment_column].unique()]
        else:
            s = meta[treatment_column]
            if pd.api.types.is_categorical_dtype(s) and getattr(s.cat, "ordered", False):
                present = [g for g in s.cat.categories if g in final_df[treatment_column].unique()]
            else:
                present = list(dict.fromkeys(final_df[treatment_column].astype(str).tolist()))

        # decide stack/legend order by global mean abundance
        mean_by_order2 = final_df.groupby('Order')['RelativeAbundance'].mean().sort_values(ascending=False)
        order_order = mean_by_order2.index.tolist()
        if "Other" in order_order:
            order_order = [x for x in order_order if x != "Other"] + ["Other"]

        color_map = consistent_color_mapping(final_df['Order'], order=order_order)

        fig = px.bar(
            final_df,
            x=treatment_column,
            y='RelativeAbundance',
            color='Order',
            category_orders={
                treatment_column: present,
                'Order': order_order
            },
            color_discrete_map=color_map,
            title=f"Mean Relative Abundance by Order (>{threshold*100:.0f}%)",
            height=700
        )
        fig.update_layout(
            legend_traceorder="reversed",  # legend matches stack top→bottom
            xaxis_title=treatment_column,
            yaxis_title="Mean Relative Abundance",
            yaxis_tickformat='.0%'
        )
        apply_pub_style(fig, portrait=True, base_font=16)
        return fig
    except Exception as e:
        logger.error(f"Could not generate Order-level abundance plot: {e}", exc_info=True)
        return go.Figure(layout_title_text=f"Error: {e}")



def plot_abundance_by_taxlevel(ps1_object, treatment_column, tax_level="Genus", threshold=0.01, group_order=None):
    """
    Stacked bar of mean relative abundance by treatment, aggregated at `tax_level`.
    Collapses rare taxa (< threshold mean share) into 'Other'.
    Respects an explicit `group_order` if provided.
    """
    try:
        asv_table = ps1_object['asv']
        tax_df    = ps1_object['tax']
        meta      = ps1_object['meta']

        if tax_level not in tax_df.columns:
            return go.Figure(layout_title_text=f"{tax_level} not found in taxonomy table.")

        melted = (
            asv_table
            .stack()
            .reset_index(name='Abundance')
            .rename(columns={'level_0': 'ASV', 'level_1': 'SampleID'})
            .merge(tax_df[[tax_level]], left_on='ASV', right_index=True, how='left')
            .merge(meta[[treatment_column]], left_on='SampleID', right_index=True, how='left')
        )
        melted[tax_level] = melted[tax_level].fillna('Unassigned')

        group_tax = melted.groupby([treatment_column, tax_level])['Abundance'].sum().reset_index()
        totals = group_tax.groupby(treatment_column)['Abundance'].transform('sum')
        group_tax['RelativeAbundance'] = group_tax['Abundance'] / totals

        mean_by_tax = group_tax.groupby(tax_level)['RelativeAbundance'].mean()
        rare = mean_by_tax[mean_by_tax < threshold].index
        group_tax[tax_level] = group_tax[tax_level].where(~group_tax[tax_level].isin(rare), 'Other')

        final_df = (group_tax
                    .groupby([treatment_column, tax_level])['RelativeAbundance']
                    .sum().reset_index())

        # Category order for x-axis
        if group_order:
            present = [g for g in group_order if g in final_df[treatment_column].unique()]
        else:
            s = meta[treatment_column]
            if pd.api.types.is_categorical_dtype(s) and getattr(s.cat, "ordered", False):
                present = [g for g in s.cat.categories if g in final_df[treatment_column].unique()]
            else:
                present = list(dict.fromkeys(final_df[treatment_column].astype(str).tolist()))

               # ---------- NEW: decide stack/legend order by global mean abundance ----------
        mean_by_tax2 = (
    final_df.groupby(tax_level)['RelativeAbundance']
            .mean()
            .sort_values(ascending=False)
)
        tax_order = mean_by_tax2.index.tolist()
        if "Other" in tax_order:
            tax_order = [x for x in tax_order if x != "Other"] + ["Other"]
        
        color_map = consistent_color_mapping(final_df[tax_level], order=tax_order)

        fig = px.bar(
            final_df,
            x=treatment_column,
            y='RelativeAbundance',
            color=tax_level,
            category_orders={
                treatment_column: present,
                tax_level: tax_order                     # <- controls trace stacking order
            },
            color_discrete_map=color_map,
            title=f"Mean Relative Abundance by {tax_level} (>{threshold*100:.0f}%)",
            height=700
        )
        fig.update_layout(legend_traceorder="reversed")

        apply_pub_style(fig, portrait=True, base_font=16)
        return fig
    except Exception as e:
        logger.error(f"Could not generate {tax_level}-level abundance plot: {e}", exc_info=True)
        return go.Figure(layout_title_text=f"Error: {e}")


def plot_ancom_clr_heatmap(
    ancom_df,
    top_k=None,
    clr_threshold=1.0,
    w_threshold=1,
    short_id_col='ASV',
    max_id_len=12,
    label_with_genus=True,
    group_order=None,          # order of *raw* groups
    label_map=None,            # {"raw_name": "Pretty label"}
):

    import numpy as np
    import pandas as pd
    import plotly.express as px

    if ancom_df is None or ancom_df.empty:
        return px.imshow([[0]], labels=dict(color="CLR mean"),
                         title="No ANCOM results")

    df = ancom_df.copy()
    clr_cols_all = [c for c in df.columns if c.startswith('CLR_mean::')]
    if not clr_cols_all:
        return px.imshow([[0]], labels=dict(color="CLR mean"),
                         title="No CLR mean columns found")

    # Respect desired column order if provided
    if group_order:
        wanted = [f"CLR_mean::{g}" for g in group_order if f"CLR_mean::{g}" in clr_cols_all]
        clr_cols = wanted + [c for c in clr_cols_all if c not in wanted]
    else:
        clr_cols = clr_cols_all

    # coerce numerics
    df['W'] = pd.to_numeric(df.get('W', np.nan), errors='coerce').fillna(-np.inf)
    for c in clr_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(-np.inf)

    # filters
    mask_w   = df['W'] > float(w_threshold)
    mask_clr = df[clr_cols].gt(float(clr_threshold)).any(axis=1)
    df_filt  = df.loc[mask_w & mask_clr].copy()
    n_after_filters = df_filt.shape[0]
    if df_filt.empty:
        return px.imshow([[0]], labels=dict(color="CLR mean"),
                         title=f"No features with W>{w_threshold} and CLR mean >{clr_threshold}")

    # --- Label: avoid "Genus Genus_*" duplication
    def _label(row):
        genus   = (str(row.get('Genus', '')   or '')).strip()
        species = (str(row.get('Species', '') or '')).strip()
        asv     = (str(row.get(short_id_col, '') or '')).strip()

        # fallbacks
        if not asv:
            fid = str(row.get('Feature_ID', ''))
            asv = (fid[:max_id_len] + '…') if len(fid) > max_id_len else (fid or 'Feature')

        # If species already starts with genus, don't repeat genus
        if species and genus and species.lower().startswith(genus.lower()):
            base = species
        else:
            base = f"{genus} {species}".strip() if (label_with_genus and (genus or species)) else (species or genus or 'Unclassified')

        return f"{base} | {asv}"

    df_filt['TaxLabel'] = df_filt.apply(_label, axis=1)

    # order by W, then keep top_k if requested
    df_filt = df_filt.sort_values('W', ascending=False)
    if top_k is not None and top_k > 0:
        df_filt = df_filt.head(top_k)

    # build wide matrix with desired column order
    wide = df_filt.set_index('TaxLabel')[clr_cols].copy()
    # raw group names, e.g. "UC_PET_2/50%_Brine_Plate_T"
    wide.columns = [c.replace('CLR_mean::', '') for c in clr_cols]
    
    # --- apply pretty labels on columns, if provided ---
    if label_map is not None:
        rename_dict = {g: label_map.get(g, g) for g in wide.columns}
        wide = wide.rename(columns=rename_dict)
    
    # --- optional column ordering based on group_order ---
    if group_order is not None:
        if label_map is not None:
            # order by PRETTY names, but using the raw group_order
            pretty_order = [
                label_map.get(g, g)
                for g in group_order
                if label_map.get(g, g) in wide.columns
            ]
        else:
            pretty_order = [g for g in group_order if g in wide.columns]
    
        if pretty_order:
            wide = wide[pretty_order]
    
    # optional row ordering: by column of max CLR
    try:
        wide = wide.loc[wide.apply(np.argmax, axis=1).sort_values().index]
    except Exception:
        pass



    fig = px.imshow(
        wide,
        color_continuous_midpoint=0,
        color_continuous_scale='RdBu',   # diverging, 0-centered
        aspect='auto',
        labels=dict(color="CLR mean (log)"),
        title=(f"ANCOM: CLR means (W>{w_threshold}, CLR>{clr_threshold}; n={wide.shape[0]} features)"
               if top_k is None else
               f"ANCOM: CLR means (W>{w_threshold}, CLR>{clr_threshold}; showing {wide.shape[0]} of {n_after_filters} features)")
    )

    # readable axes
    row_count = wide.shape[0]
    fig.update_layout(
        height=max(540, 22 * row_count + 160),
        margin=dict(l=330, r=100, t=70, b=110),  # more room left/right
        width=1100,
        coloraxis_colorbar=dict(title="CLR mean", thickness=14)
    )
    fig.update_yaxes(type="category", tickfont=dict(size=11), automargin=True)
    fig.update_xaxes(tickfont=dict(size=11), tickangle=35, automargin=True)

    return fig

def plot_lme_results(results_df, show_insignificant=False):
    """
    Simple viz for mixed-effect model results.

    Expects columns at least:
      - 'Feature_ID' (or 'feature')
      - 'Variable'   (effect term)
      - 'Coefficient'
      - optionally: 'P-value', 'qval', 'Significant' (bool)

    Returns a Plotly Figure.
    """
    import pandas as pd
    import numpy as np
    import plotly.express as px

    if results_df is None or (hasattr(results_df, "empty") and results_df.empty):
        return px.scatter(x=[0], y=[0], title="Mixed-effect model: no results")

    df = results_df.copy()

    # Prefer to drop intercepts for clarity
    if 'Variable' in df.columns:
        df = df[df['Variable'].str.lower() != 'intercept'] if df['Variable'].dtype == object else df

    # Filter insignif if requested (prefer qval, else 'Significant' flag, else p-value)
    if not show_insignificant:
        if 'qval' in df.columns:
            df = df[df['qval'] < 0.05]
        elif 'Significant' in df.columns:
            df = df[df['Significant']]
        elif 'P-value' in df.columns:
            df = df[df['P-value'] < 0.05]

    if df.empty:
        return px.scatter(x=[0], y=[0], title="Mixed-effect model: no significant results")

    # Build a readable y-axis label: Feature_ID · Variable
    feat_col = 'Feature_ID' if 'Feature_ID' in df.columns else ('feature' if 'feature' in df.columns else None)
    var_col  = 'Variable'   if 'Variable'   in df.columns else None
    if feat_col and var_col:
        df['_label'] = df[feat_col].astype(str) + " · " + df[var_col].astype(str)
    elif feat_col:
        df['_label'] = df[feat_col].astype(str)
    elif var_col:
        df['_label'] = df[var_col].astype(str)
    else:
        df['_label'] = df.index.astype(str)

    # Numeric x: Coefficient (fallback to first numeric)
    x_col = 'Coefficient' if 'Coefficient' in df.columns else df.select_dtypes(include='number').columns[0]

    # Optional color by significance/effect
    color_col = None
    if 'Significant' in df.columns:
        color_col = 'Significant'
    elif 'effect' in df.columns:
        color_col = 'effect'

    fig = px.bar(
        df.sort_values(x_col),
        y='_label',
        x=x_col,
        color=color_col,
        title="Mixed-effect model estimates",
        orientation='h'
    )
    # Add CI whiskers if SE exists
    if 'se' in df.columns:
        fig.update_traces(error_x=dict(array=df['se'], visible=True))

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending', 'title': ''},
        xaxis_title="Coefficient (log scale if modelled on log-link)",
        height=max(500, 18 * len(df) + 160),
        margin=dict(l=220, r=20, t=60, b=60)
    )
    return fig
def format_lme_results_for_display(results_df, ps1_object, treatment_col, time_col, reference_group=None, show_insignificant=False, force_features=None):
    if results_df is None or results_df.empty:
        return html.Div([html.H4("Mixed-Effect Model Results"), html.P("No associations were found.")])

    df = results_df[results_df['Variable'] != 'Intercept'].copy()
    if not show_insignificant:
        df = df[df['Significant']].copy()
    if df.empty:
        return html.Div([html.H4("Mixed-Effect Model Results"), html.P("No significant associations found.")])

    baseline = reference_group if reference_group else sorted(ps1_object['meta'][treatment_col].unique())[0]
    df['Fold Change'] = np.exp(df['Coefficient']).map('{:.2f}x'.format)
    df['P-value'] = df['P-value'].map('{:.2e}'.format)

    def get_interpretation(row):
        var = row['Variable']
        if treatment_col in var:
            return f"Effect of {var.split('.')[-1].replace(']', '')} (vs {baseline})"
        if time_col and time_col in var and ":" not in var:
            return f"Effect of one-unit increase in {time_col}"
        if time_col and ":" in var:
            return "Interaction Effect"
        return var

    df['Interpretation'] = df.apply(get_interpretation, axis=1)

    display_cols = ['Feature_ID', 'Interpretation', 'Fold Change', 'P-value', 'Phylum', 'Order', 'Family', 'Genus']
    final_df = df[[col for col in display_cols if col in df.columns]].sort_values('Feature_ID').fillna('-')

    header = [html.Th(col) for col in final_df.columns]
    rows = [html.Tr([html.Td(cell) for cell in row_tuple]) for row_tuple in final_df.itertuples(index=False)]
    table = html.Table([html.Thead(html.Tr(header))] + [html.Tbody(rows)])

    return html.Div([
        html.H4("Significant Associations from Mixed-Effect Model"),
        html.P(f"Baseline for comparison is '{baseline}'."),
        table
    ])
