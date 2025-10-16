
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

def add_stat_annotations(fig, alpha_df, treatment_col, stats_df):
    unique_groups = list(alpha_df[treatment_col].unique())
    group_positions = {group: i for i, group in enumerate(unique_groups)}
    y_max = alpha_df['Shannon'].max()
    y_step = y_max * 0.15
    y_current = y_max + y_step
    significant_pairs = stats_df[stats_df['significant']].sort_values('p_adj')

    for _, row in significant_pairs.iterrows():
        x1, x2 = group_positions.get(row['group1']), group_positions.get(row['group2'])
        if x1 is None or x2 is None: continue
        p_text = "p < 0.001" if row['p_adj'] < 0.001 else f"p = {row['p_adj']:.3f}"
        fig.add_shape(type="line", x0=x1, y0=y_current, x1=x2, y1=y_current, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=x1, y0=y_current*0.99, x1=x1, y1=y_current, line=dict(color='black', width=1))
        fig.add_shape(type="line", x0=x2, y0=y_current*0.99, x1=x2, y1=y_current, line=dict(color='black', width=1))
        fig.add_annotation(x=(x1 + x2) / 2, y=y_current + (y_step*0.1), text=p_text, showarrow=False)
        y_current += y_step
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

def plot_abundance_by_order(ps1_object, treatment_column, threshold=0.01):
    try:
        asv_table = ps1_object['asv'] # ASVs x Samples
        melted_df = asv_table.stack().reset_index(name='Abundance').rename(columns={'level_0': 'ASV', 'level_1': 'SampleID'})
        melted_df = melted_df.merge(ps1_object['tax']['Order'], on='ASV').merge(ps1_object['meta'][[treatment_column]], left_on='SampleID', right_index=True)

        group_order_sums = melted_df.groupby([treatment_column, 'Order'])['Abundance'].sum().reset_index()
        total_sum_per_group = group_order_sums.groupby(treatment_column)['Abundance'].transform('sum')
        group_order_sums['RelativeAbundance'] = group_order_sums['Abundance'] / total_sum_per_group

        rare_orders = group_order_sums.groupby('Order')['RelativeAbundance'].mean()
        rare_orders = rare_orders[rare_orders < threshold].index
        group_order_sums['Order'] = group_order_sums['Order'].replace(list(rare_orders), 'Other')

        final_plot_df = group_order_sums.groupby([treatment_column, 'Order'])['RelativeAbundance'].sum().reset_index()

        fig = px.bar(final_plot_df, x=treatment_column, y='RelativeAbundance', color='Order', title=f"Mean Relative Abundance by Order (>{threshold*100}%)", height=700)
        fig.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_title="Mean Relative Abundance", yaxis_tickformat='.0%', legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5))
        return fig
    except Exception as e:
        logger.error(f"Could not generate Order-level abundance plot: {e}", exc_info=True)
        return go.Figure(layout_title_text=f"Error: {e}")
def plot_abundance_by_taxlevel(ps1_object, treatment_column, tax_level="Genus", threshold=0.01):
    """
    Stacked bar of mean relative abundance by treatment, aggregated at `tax_level`.
    `threshold` collapses low-mean taxa into 'Other'.
    """
    try:
        asv_table = ps1_object['asv']  # ASVs x Samples
        tax_df = ps1_object['tax']     # index = ASV
        meta = ps1_object['meta']

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

        fig = px.bar(
            final_df,
            x=treatment_column,
            y='RelativeAbundance',
            color=tax_level,
            title=f"Mean Relative Abundance by {tax_level} (>{threshold*100:.0f}%)",
            height=700
        )
        fig.update_layout(
            xaxis={'categoryorder': 'total descending'},
            yaxis_title="Mean Relative Abundance",
            yaxis_tickformat='.0%',
            legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5)
        )
        return fig
    except Exception as e:
        logger.error(f"Could not generate {tax_level}-level abundance plot: {e}", exc_info=True)
        return go.Figure(layout_title_text=f"Error: {e}")
def plot_ancom_clr_heatmap(ancom_df, tax_level_cols=('Genus','Species'), top_k=30):
    """
    ancom_df: output of run_ancom_skbio() that contains
      - 'W', 'reject'
      - one column per treatment named 'CLR_mean::<treatment>'
      - taxonomy columns (optional)
    tax_level_cols: tuple/list of taxonomy columns to build a display label
    top_k: number of features to show (ordered by W)

    Returns: a Plotly Figure (heatmap)
    """
    if ancom_df is None or ancom_df.empty:
        return px.imshow(np.zeros((1,1)), labels=dict(color="CLR mean"), title="No ANCOM results")

    # keep only significant & top by W
    cols_clr = [c for c in ancom_df.columns if c.startswith('CLR_mean::')]
    df = ancom_df.copy()
    df_sig = df[df['reject']].sort_values('W', ascending=False).head(top_k)

    if df_sig.empty or not cols_clr:
        return px.imshow(np.zeros((1,1)), labels=dict(color="CLR mean"), title="No significant features")

    # display label: prefer requested taxonomy level(s), fallback to ASV id
    def _label(row):
        parts = [str(row[c]) for c in tax_level_cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
        return " ".join(parts) if parts else row.get('ASV', row.get('Feature_ID'))

    df_sig = df_sig.assign(Display=df_sig.apply(_label, axis=1))

    # long → wide (features × groups)
    wide = df_sig.set_index('Display')[cols_clr]
    # a nicer group name without prefix
    wide.columns = [c.replace('CLR_mean::','') for c in wide.columns]

    # sort rows by which group is highest (optional)
    wide = wide.loc[wide.apply(np.argmax, axis=1).sort_values().index]

    fig = px.imshow(
        wide,
        color_continuous_midpoint=0,
        aspect='auto',
        labels=dict(color="CLR mean (log)"),
        title=f"ANCOM: Per-group CLR mean (top {len(wide)} features)"
    )
    fig.update_layout(margin=dict(l=60, r=10, t=40, b=40))
    return fig
def plot_lme_results(results_df, show_insignificant=False):
    if results_df is None or results_df.empty:
        return go.Figure(layout_title_text="No model results to plot.")

    df = results_df[results_df['Variable'] != 'Intercept'].copy()
    if not show_insignificant: df = df[df['Significant']].copy()
    if df.empty: return go.Figure(layout_title_text="No significant results to plot.")

    df['Fold Change'] = np.exp(df['Coefficient'])
    df['FC Lower'] = np.exp(df['Conf. Int. Lower'])
    df['FC Upper'] = np.exp(df['Conf. Int. Upper'])
    df['Plot Label'] = df['Feature_ID'] + " | " + df['Variable']
    df = df.sort_values('Feature_ID')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Fold Change'], y=df['Plot Label'], mode='markers', marker=dict(color=df['Significant'].map({True: 'blue', False: 'gray'}), size=10), name='Fold Change'))

    error_bars = []
    for i, row in df.iterrows():
        error_bars.append(go.Scatter(
            x=[row['FC Lower'], row['FC Upper']],
            y=[row['Plot Label'], row['Plot Label']],
            mode='lines',
            line=dict(color='blue' if row['Significant'] else 'gray', width=1),
            showlegend=False
        ))
    fig.add_traces(error_bars)

    fig.update_layout(xaxis_type='log', xaxis_title='Fold Change (log scale)', title='Mixed-Effect Model Results (Forest Plot)', height=max(600, len(df)*30), showlegend=False)
    fig.add_vline(x=1.0, line_width=2, line_dash="dash", line_color="grey")
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
