import os
import base64
import uuid
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
# Internal module imports
from .config import logger
from .data.sample_data import sample_metadata_df, sample_seqtab, sample_taxa
from .ai_utils import ai_available, ai_analyze_background, ai_analyze_quality_profiles, ai_analyze_metadata, ai_interpret_results
from .pipeline_steps import filter_and_trim_parallel, denoise_and_create_asv_table_vsearch, assign_taxonomy
from .analysis import (
    create_phyloseq_object, calculate_alpha_diversity, calculate_beta_diversity,
    perform_pcoa, perform_pca, perform_nmds, perform_pcoa_aitchison
)
from .statistics import (perform_pairwise_alpha_tests, permanova_marginal, betadisper_anova, run_differential_abundance, 
                         run_indicator_species, run_mixed_effect_model, run_pymc_zinb_mixed_model, run_ancom_skbio, valid_interaction)
from .plotting import (add_stat_annotations, plot_phylogenetic_tree, plot_abundance_by_order, plot_abundance_by_taxlevel, plot_ancom_clr_heatmap, 
                       plot_lme_results, format_lme_results_for_display,_apply_labels_and_order, consistent_color_mapping)
from .reporting import generate_pdf_report
from .utils import fill_taxonomy_forward
from dash import no_update
from dash import dash_table
from .plotting import apply_pub_style, get_pub_palette
from typing import List, Optional
from .statistics import add_interactions
# --- Initialize Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Global Data Store ---
global_data = {}

# --- App Layout ---
app.layout = html.Div([
    html.H1("Microbiome Analysis Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='1. Setup & Inputs', children=[
            html.H3("Input Data"),
            html.Button('Use Internal Sample Data', id='load-sample-data', n_clicks=0),
            html.Br(), html.Br(),
            html.Label("Project Data Folder Path:"),
            dcc.Input(id='data-folder-path', value='./uploads', type='text', style={'width': '80%'}),
            html.Button('List Files', id='list-files-button', n_clicks=0, style={'marginLeft': '10px'}),
            html.P("Place your data in a folder (e.g., 'uploads') and provide the path.", style={'fontSize': 'small'}),
            html.Div(id='file-listing-output', style={'marginTop': '10px'}),
            html.Label("Upload SILVA Taxonomy Database (or place `silva.fasta` in data folder)"),
            dcc.Upload(id='upload-silva', children=html.Button('Upload SILVA File')),
            html.Div(id='silva-status-output'),
            html.Label("Output Directory:"),
            dcc.Input(id='output-dir', value='./output', type='text'),
            html.H3("Study Information"),
            dcc.Textarea(id='background-info', placeholder='Describe your study goals...', style={'width': '100%', 'height': 100}),
            html.Div(id='ai-clarification-questions'),
        ]),
        dcc.Tab(label='2. Parameters & Execution', children=[
            html.H3("Analysis Starting Point"),
            dcc.RadioItems(id='analysis-mode', options=[
                {'label': 'Start from raw FASTQ files (Full pipeline)', 'value': 'fastq'},
                {'label': 'Start from Processed ASV/Taxonomy Tables (Fast)', 'value': 'asv'}
            ], value='fastq', labelStyle={'display': 'block'}),
            html.Div(id='manual-params', children=[
                html.Div(id='preprocessing-params-div', children=[
                    html.H4("Pre-processing (FASTQ mode)"),
                    html.Label("Truncation Length (Fwd/Rev):"),
                    dcc.Input(id='trunc-len-f', value=240, type='number'), dcc.Input(id='trunc-len-r', value=200, type='number'),
                    html.Label("Max Expected Errors (Fwd,Rev):"),
                    dcc.Input(id='max-ee', value='2,2', type='text'),
                ]),
                html.H4("Downstream Analysis"),
                html.Label("Treatment Group Column:"), dcc.Input(id='treatment-group', value='Substrate', type='text'),
                html.Label("Groups to Compare (subsetting):"), dcc.Dropdown(id='subset-groups-dropdown', multi=True, placeholder="Leave blank for all"),
                html.Label("Top ASVs for PCA:"), dcc.Input(id='top-asvs', value=50, type='number'),

html.Label("Differential abundance method:"),
dcc.Dropdown(
    id='da-method',
    options=[
        {'label': 'Kruskal–Wallis (current)', 'value': 'kw'},
        {'label': 'ANCOM (Python, scikit-bio)', 'value': 'ancom'},
    ],
    value='kw',
),

                        html.Summary("PERMANOVA & PERMDISP Settings"),
                        html.Div([
                            html.P("Select factors for PERMANOVA (main effects):"),
                            dcc.Dropdown(
                                id='permanova-factors-dropdown',
                                multi=True,
                                placeholder="Select factors (e.g., Substrate, Media)"
                            ),
                            html.P("Enter interaction terms (comma-separated, e.g., Cultivar:Media):"),
                            dcc.Input(
                                id='permanova-interactions-input',
                                type='text',
                                value='',
                                placeholder="Cultivar:Media, Media:TIME_D"
                            ),
                            html.P("Select Strata (blocking factor, optional):"),
                            dcc.Dropdown(
                                id='permanova-strata-dropdown',
                                placeholder="Select a blocking factor (e.g., Run)"
                            ),
                            html.P("Number of permutations:"),
                            dcc.Input(
                                id='permanova-permutations',
                                type='number',
                                value=999
                            ),
                        ], style={'padding': '10px', 'border': '1px solid #eee', 'borderRadius': '5px', 'marginBottom': '10px'}),

                html.H4("Mixed-Effect Model"),
                html.Label("Model Type:"),
                dcc.Dropdown(id='model-type-dropdown', options=[
                    {'label': 'Negative Binomial GEE (Faster)', 'value': 'gee'},
                    {'label': 'Bayesian ZINB (PyMC - Placeholder)', 'value': 'pymc_zinb'}
                ], value='gee'),
                html.Label("Main Factor:"), dcc.Input(id='mem-treatment-col', value='Substrate', type='text'),
                html.Label("Time/Second Factor (Optional):"), dcc.Input(id='time-col', value='', type='text'),
                html.Label("Reference Group:"), dcc.Input(id='mem-reference-group-input', type='text', placeholder="e.g., Control"),
                html.Label("Analysis Level:"), dcc.Dropdown(id='analysis-level-dropdown', options=[{'label': lvl, 'value': lvl} for lvl in ['ASV', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']], value='Genus'),
                html.Label("Top Features for Model:"), dcc.Input(id='mem-top-n-features', value=20, type='number'),
                html.Label("Forced Features:", id='force-features-label'), dcc.Dropdown(id='force_features', multi=True),
                html.Label("Grouping Variable (Random Effect):"), dcc.Dropdown(id='random-effect-cols', multi=True),
                html.Label("Show Insignificant Results:"), dcc.Dropdown(id='show-insignificant', options=[{'label': 'No', 'value': False}, {'label': 'Yes', 'value': True}], value=False),
            ]),
            html.Br(),
            html.Button('Run Analysis', id='run-analysis', n_clicks=0, style={'fontSize': '1.2em', 'padding': '10px'}),
        ]),
        dcc.Tab(label='3. Results & Visualization', children=[
            dcc.Loading(id="loading-results", type="default", children=[
                html.Div(id='results-output', children=[
                    html.H3("Sequencing Depth"), dcc.Graph(id='seq-depth-plot'),
                    html.H3("Total Reads by Group"), dcc.Graph(id='total-reads-plot'),
                    html.H3("Alpha Diversity"), dcc.Graph(id='alpha-diversity-plot'),
                    html.H3("Beta Diversity & Ordination"),
                    html.Div(id='permanova-results', style={'textAlign': 'center'}),
                    dcc.Graph(id='pcoa-plot'),
                    dcc.Graph(id='pcoa-aitchison-plot'),
                    dcc.Graph(id='pca-plot'),
                    dcc.Graph(id='nmds-plot'),
                    html.H3("Taxonomic Composition"),
                    dcc.Graph(id='abundance-order-plot'),
                    dcc.Graph(id='abundance-genus-plot'),
                    html.H3("Statistical Comparisons"),
                    html.Div(id='differential-abundance-results'),
                    html.H4("ANCOM – CLR heatmap (significant taxa)"),
                    dcc.Graph(id='ancom-heatmap'),
                    html.Div(id='indicator-species-results'),
                    html.H3("Phylogenetic Tree"),
                    html.Div(id='phylogenetic-tree'),
                    html.H3("Mixed-Effect Model Results"),
                    html.Div(id='mixed-model-results'),
                    dcc.Graph(id='mixed-model-plot'),
                ])
            ])
        ]),
        dcc.Tab(label='4. Report & Interpretation', children=[
            html.H3("AI-Powered Interpretation"),
            dcc.Markdown(id='ai-interpretations', style={'border': '1px solid #ccc', 'padding': '10px', 'minHeight': '200px'}),
            html.Br(),
            html.Button('Download PDF Report', id='download-report-button', n_clicks=0),
            dcc.Download(id='download-report'),
            html.H3("Output Files"),
            html.Div(id='output-files'),
        ])
    ])
])

def _infer_group_order(meta_df: pd.DataFrame, treat_col: str) -> list:
    s = meta_df[treat_col]

    # (a) ordered Categorical in metadata → use categories
    if pd.api.types.is_categorical_dtype(s) and getattr(s.cat, "ordered", False):
        return list(s.cat.categories)

    # (b) look for a numeric order column ...
    for c in [f"{treat_col}_order", "PlotOrder", "GroupOrder", "order", "Order", "Run", "Day", "Time"]:
        if c in meta_df.columns:
            ords = pd.to_numeric(meta_df[c], errors="coerce")
            if ords.notna().any():
                return (meta_df.assign(__ord=ords)
                               .groupby(treat_col)["__ord"].min()
                               .sort_values()
                               .index.tolist())

    # (c) fallback: **alphabetical**
    return sorted(pd.unique(s.astype(str)))

def _pick_multifactor_terms(
    meta_df: pd.DataFrame,
    extra: Optional[List[str]] = None,
    max_levels: int = 50
) -> List[str]:
    """
    Choose multifactor terms present in metadata (case-insensitive).
    - Includes a default set of common factors.
    - 'extra' lets you force-include columns (e.g., treat_col, custom names).
    - Drops columns that have <2 non-NA unique values.
    - Skips very high-cardinality categoricals (> max_levels).
    Returns actual column names in metadata order.
    """
    default_candidates = [
        "time", "run", "substrate", "media", "cultivar",
        "cultivation_unit", "time_d"
    ]

    lower_map = {c.lower(): c for c in meta_df.columns}

    wanted = list(default_candidates)
    if extra:
        wanted.extend([e.lower() for e in extra])

    seen = set()
    wanted = [w for w in wanted if not (w in seen or seen.add(w))]

    terms = [lower_map[w] for w in wanted if w in lower_map]

    cleaned: List[str] = []
    for t in terms:
        s = meta_df[t].dropna()
        if s.nunique() < 2:
            continue
        if not pd.api.types.is_numeric_dtype(s) and s.astype(str).nunique() > max_levels:
            continue
        cleaned.append(t)
    return cleaned
# --- Callbacks ---

@app.callback(
    [Output('permanova-factors-dropdown', 'options'),
     Output('permanova-factors-dropdown', 'value'),
     Output('permanova-strata-dropdown', 'options'),
     Output('permanova-strata-dropdown', 'value'),
     Output('random-effect-cols', 'options')], # Also update random effect options
    [Input('load-sample-data', 'n_clicks'),
     Input('list-files-button', 'n_clicks')],
    State('data-folder-path', 'value'),
    prevent_initial_call=False # Allow initial call to populate if data exists
)
def populate_permanova_and_random_effect_options(n_sample, n_list, data_path):
    # This logic is similar to populate_subset_options, but we need to ensure
    # metadata is loaded to infer column names.
    meta_df = pd.DataFrame() # Default empty

    try:
        if 'sample_data_mode' in global_data or (n_sample > 0 and ctx.triggered_id == 'load-sample-data'):
            meta_df = sample_metadata_df.copy()
            logger.info("Populating PERMANOVA/Random Effect dropdowns from sample metadata.")
        elif data_path and os.path.isdir(data_path):
            files = os.listdir(data_path)
            meta_file = next((f for f in files if f.lower().endswith(('.csv', '.tsv', '.txt'))), None)
            if meta_file:
                meta_path = os.path.join(data_path, meta_file)
                sep = '\t' if os.path.splitext(meta_file)[1].lower() in ('.tsv', '.txt') else ','
                meta_df = pd.read_csv(meta_path, sep=sep, index_col=0)
                logger.info(f"Populating PERMANOVA/Random Effect dropdowns from {meta_file}.")

        if meta_df.empty:
            return [], [], [], None, [] # Empty options

        # Filter out high-cardinality columns for factors/strata
        available_cols = []
        for col in meta_df.columns:
            s = meta_df[col].dropna()
            # Must have at least two unique values
            if s.nunique() < 2:
                continue
            # Skip very high cardinality for categorical factors (e.g., unique identifiers)
            if not pd.api.types.is_numeric_dtype(s) and s.nunique() > (len(s) * 0.8): # Heuristic for too many unique values
                 continue
            available_cols.append({'label': col, 'value': col})

        # Pre-select a few common factors if they exist in metadata
        default_factors = [c for c in ["Substrate", "Media", "TIME_D", "Cultivar"] if {'label': c, 'value': c} in available_cols]
        # Pre-select 'Run' or similar for strata if it exists
        default_strata = next((c['value'] for c in available_cols if c['value'].lower() in ["run", "block"]), None)

        # For random effect columns, allow any reasonable categorical or numeric column
        random_effect_options = []
        for col in meta_df.columns:
            s = meta_df[col].dropna()
            if s.nunique() >= 1: # Can be 1 if it's a random effect, but generally >1
                random_effect_options.append({'label': col, 'value': col})

        return (
            available_cols, default_factors,
            available_cols, default_strata,
            random_effect_options
        )

    except Exception as e:
        logger.error(f"Error populating PERMANOVA dropdowns: {e}", exc_info=True)
        return [], [], [], None, [] # Return empty on error


@app.callback(
    Output('file-listing-output', 'children'),
    Input('list-files-button', 'n_clicks'),
    State('data-folder-path', 'value'),
    prevent_initial_call=True
)
def list_files(n_clicks, folder_path):
    if not folder_path or not os.path.isdir(folder_path):
        return html.P(f"Error: Folder not found: '{folder_path}'", style={'color': 'red'})
    try:
        files = os.listdir(folder_path)
        r1 = sorted([f for f in files if '_R1' in f.upper() and f.lower().endswith(('.fastq', '.fastq.gz'))])
        r2 = sorted([f for f in files if '_R2' in f.upper() and f.lower().endswith(('.fastq', '.fastq.gz'))])
        meta = sorted([f for f in files if f.lower().endswith(('.csv', '.tsv', '.txt'))])
        return html.Div([html.P(f"Found {len(r1)} R1 files, {len(r2)} R2 files, and {len(meta)} metadata files.")])
    except Exception as e:
        return html.P(f"Error accessing folder: {e}", style={'color': 'red'})

@app.callback(
    Output('preprocessing-params-div', 'style'),
    Input('analysis-mode', 'value')
)
def toggle_preprocessing_params(mode):
    return {'display': 'block'} if mode == 'fastq' else {'display': 'none'}
@app.callback(
    [Output('subset-groups-dropdown', 'options'),
     Output('subset-groups-dropdown', 'value'),
     Output('subset-groups-dropdown', 'disabled')],
    [Input('treatment-group', 'value'),
     Input('load-sample-data', 'n_clicks'),
     Input('list-files-button', 'n_clicks')],
    State('data-folder-path', 'value'),
    prevent_initial_call=True
)
def populate_subset_options(treat_col, n_sample, n_list, data_path):
    try:
        if 'sample_data_mode' in global_data:
            meta_df = sample_metadata_df.copy()
        else:
            if not data_path or not os.path.isdir(data_path):
                return [], None, True
            files = os.listdir(data_path)
            meta_file = next((f for f in files if f.lower().endswith(('.csv', '.tsv', '.txt'))), None)
            if not meta_file:
                return [], None, True
            meta_path = os.path.join(data_path, meta_file)
            ext = os.path.splitext(meta_file)[1].lower()
            sep = '\t' if ext in ('.tsv', '.txt') else ','
            meta_df = pd.read_csv(meta_path, sep=sep, index_col=0)

        if not treat_col or treat_col not in meta_df.columns:
            return [], None, True

        groups = (
            meta_df[treat_col]
            .astype(str).str.strip()
            .replace({'nan': np.nan})
            .dropna().unique().tolist()
        )
        groups = sorted(groups)
        options = [{'label': g, 'value': g} for g in groups]
        return options, None, False

    except Exception as e:
        logger.error(f"Could not populate subsetting options: {e}", exc_info=True)
        return [], None, True
@app.callback(
    [
        Output('seq-depth-plot', 'figure'),
        Output('total-reads-plot', 'figure'),
        Output('alpha-diversity-plot', 'figure'),
        Output('pcoa-plot', 'figure'),
        Output('pcoa-aitchison-plot', 'figure'),
        Output('pca-plot', 'figure'),
        Output('nmds-plot', 'figure'),
        Output('permanova-results', 'children'),
        Output('abundance-order-plot', 'figure'),
        Output('abundance-genus-plot', 'figure'),
        Output('differential-abundance-results', 'children'),
        Output('ancom-heatmap', 'figure'),
        Output('indicator-species-results', 'children'),
        Output('phylogenetic-tree', 'children'),
        Output('mixed-model-results', 'children'),
        Output('mixed-model-plot', 'figure'),
        Output('ai-interpretations', 'children'),
        Output('output-files', 'children'),
    ],
    Input('run-analysis', 'n_clicks'),
    [
        State('analysis-mode', 'value'),
        State('data-folder-path', 'value'),
        State('output-dir', 'value'),
        State('trunc-len-f', 'value'),
        State('trunc-len-r', 'value'),
        State('max-ee', 'value'),
        State('treatment-group', 'value'),
        State('subset-groups-dropdown', 'value'),
        State('top-asvs', 'value'),
        State('background-info', 'value'),
        State('upload-silva', 'contents'),
        State('model-type-dropdown', 'value'),
        State('mem-treatment-col', 'value'),
        State('time-col', 'value'),
        State('random-effect-cols', 'value'),
        State('analysis-level-dropdown', 'value'),
        State('mem-top-n-features', 'value'),
        State('mem-reference-group-input', 'value'),
        State('show-insignificant', 'value'),
        State('force_features', 'value'),
        State('da-method', 'value'),
        State('permanova-factors-dropdown', 'value'),
        State('permanova-interactions-input', 'value'),
        State('permanova-strata-dropdown', 'value'),
        State('permanova-permutations', 'value'),
    ],
    prevent_initial_call=True,
)
def run_full_analysis(n_clicks, analysis_mode, data_path, out_dir, trunc_f, trunc_r, max_ee, treat_col,
                      subset, top_asvs, background, silva_content, model_type, mem_treat, time_col,
                      rand_eff, analysis_lvl, mem_top_n, mem_ref, mem_show_insig, force_feat, da_method,
                      permanova_selected_factors, permanova_interaction_input, permanova_selected_strata, permanova_num_permutations):
    if n_clicks == 0:
        empty_fig = go.Figure()
        empty_div = html.Div()
        return [
            empty_fig,  # 1 seq-depth-plot
            empty_fig,
            empty_fig,  # 2 alpha-diversity-plot
            empty_fig,  # 3 pcoa-plot
            empty_fig,  # 4 pcoa-aitchison-plot
            empty_fig,  # 5 pca-plot
            empty_fig,  # 6 nmds-plot
            empty_div,  # 7 permanova-results
            empty_fig,  # 8 abundance-order-plot
            empty_fig,  # 9 abundance-genus-plot
            empty_div,  # 10 differential-abundance-results
            empty_fig,  # 11 ancom-heatmap.figure
            empty_div,  # 12 indicator-species-results
            empty_div,  # 13 phylogenetic-tree
            empty_div,  # 14 mixed-model-results
            empty_fig,  # 15 mixed-model-plot
            "",         # 16 ai-interpretations
            empty_div   # 17 output-files
        ]

    try:
        # --- (Re)initialize run state ---
        sample_mode = global_data.get('sample_data_mode', False)
        global_data.clear()  # Reset data for new run
        if sample_mode:
            global_data['sample_data_mode'] = True
        os.makedirs(out_dir, exist_ok=True)

        # --- DATA LOADING ---
        use_sample_data = 'sample_data_mode' in global_data

        if use_sample_data:
            logger.info("Using internal sample data for analysis.")
            seqtab, taxa_df, meta_df = (
                sample_seqtab.copy(),
                sample_taxa.copy(),
                sample_metadata_df.copy(),
            )
        else:
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"Data folder '{data_path}' not found.")
            files = os.listdir(data_path)
            # Add '.txt' here too if your metadata might be .txt
            meta_file = next((f for f in files if f.lower().endswith(('.csv', '.tsv'))), None)
            if not meta_file:
                raise FileNotFoundError("Metadata file not found in data folder.")
            meta_df = pd.read_csv(os.path.join(data_path, meta_file), index_col=0)
            meta_df.index = meta_df.index.astype(str)

            if analysis_mode == 'fastq':
                logger.info("Starting analysis from FASTQ files.")
                fnFs = sorted([os.path.join(data_path, f) for f in files if '_R1' in f.upper()])
                fnRs = sorted([os.path.join(data_path, f) for f in files if '_R2' in f.upper()])
                # Build sample names and align metadata safely
                s_names = [os.path.basename(f).split('_')[0].strip() for f in fnFs]
                meta_df.index = meta_df.index.astype(str).str.strip()
                meta_df = meta_df.reindex(s_names)
                filtFs, filtRs = filter_and_trim_parallel(fnFs, fnRs, s_names, out_dir, trunc_f, trunc_r, max_ee)
                seqtab = denoise_and_create_asv_table_vsearch(filtFs, filtRs, s_names, out_dir)
                silva_path = os.path.join(data_path, 'silva.fasta')
                if silva_content:
                    _, content_string = silva_content.split(',')
                    silva_path = os.path.join(out_dir, 'uploaded_silva.fasta')
                    with open(silva_path, 'wb') as f:
                        f.write(base64.b64decode(content_string))
                taxa_df = assign_taxonomy(list(seqtab.columns), os.path.join(out_dir, 'asvs.fa'), silva_path, out_dir)
            else:  # ASV mode
                logger.info("Starting analysis from pre-processed tables.")
                asv_path = os.path.join(out_dir, 'microbiome_ai_16s_asv.csv')
                taxa_path = os.path.join(out_dir, 'microbiome_ai_taxonomy.csv')
                if not (os.path.exists(asv_path) and os.path.exists(taxa_path)):
                    raise FileNotFoundError("Run in FASTQ mode first to generate ASV/Taxonomy tables in the output directory.")
                # If file is comma-separated despite .csv, change sep to ','
                seqtab = pd.read_csv(asv_path, sep='\t', index_col=0, engine='python')
                taxa_df = pd.read_csv(taxa_path, index_col=0)

        # --- Normalize sample IDs so ASV table and metadata align (both modes) ---
        seqtab.index = seqtab.index.map(lambda x: str(x).strip())

        if 'SampleID' in meta_df.columns:
            meta_df = meta_df.copy()
            meta_df['SampleID'] = meta_df['SampleID'].astype(str).str.strip()
            meta_df = meta_df.set_index('SampleID')
        else:
            meta_df.index = meta_df.index.map(lambda x: str(x).strip())

        missing_in_meta = sorted(set(seqtab.index) - set(meta_df.index))
        missing_in_seq  = sorted(set(meta_df.index) - set(seqtab.index))
        if missing_in_meta:
            logger.warning(f"Samples in ASV table but missing in metadata (up to 10): "
                           f"{missing_in_meta[:10]}{'...' if len(missing_in_meta) > 10 else ''}")
        if missing_in_seq:
            logger.warning(f"Samples in metadata but not in ASV table (up to 10): "
                           f"{missing_in_seq[:10]}{'...' if len(missing_in_seq) > 10 else ''}")

        meta_df = meta_df.reindex(seqtab.index)

        taxa_filled = fill_taxonomy_forward(taxa_df)
        global_data['seqtab_nochim'], global_data['taxa'] = seqtab, taxa_filled

        # --- CORE ANALYSIS ---
        ps1_full = create_phyloseq_object(seqtab, taxa_filled, meta_df)
        logger.info(f"Before subsetting: {ps1_full['asv'].shape[1]} Samples, Gruppen: {sorted(ps1_full['meta'][treat_col].unique())}")
        if subset:
            logger.info(f"Subset requested: {subset}")
        else:
            logger.info("No subset requested")
        ps1 = ps1_full
        if subset:
            logger.info(f"Subsetting requested: {subset} (Typ: {[type(x).__name__ for x in subset]})")
            logger.info(f"Available levels in '{treat_col}': {sorted(ps1_full['meta'][treat_col].unique())}")

            subset_str = [str(s).strip() for s in subset]
            meta_str = ps1_full['meta'][treat_col].astype(str).str.strip()

            mask = meta_str.isin(subset_str)
            matched_samples = mask.sum()

            logger.info(f"→ {matched_samples} samples matched to {subset_str}")

            if matched_samples == 0:
                raise ValueError(
                    f"SUBSETTING FAILED\n"
                    f"Requested subset: {subset}\n"
                    f"VSaved as CSV: {sorted(ps1_full['meta'][treat_col].astype(str).unique())}\n"
                )

            if matched_samples < 2:
                raise ValueError(f"Only {matched_samples} samples found. Insufficient sample size for analysis.")

            # Nur gültige Samples behalten
            valid_idx = ps1_full['meta'].index[mask]
            ps1 = {
                'asv': ps1_full['asv'].loc[:, valid_idx],
                'tax': ps1_full['tax'],
                'meta': ps1_full['meta'].loc[valid_idx]
            }

            logger.info(f"SUBSETTING SUCCESFULL → {len(valid_idx)} samples included in subset")
        else:
            logger.info("No subset requested")
        global_data['ps1'] = ps1

               # ---- Consistent group order (from metadata) ----
        group_order = _infer_group_order(ps1['meta'], treat_col)
        ps1['meta'][treat_col] = pd.Categorical(
            ps1['meta'][treat_col], categories=group_order, ordered=True
        )
        global_data['group_order'] = group_order
        
        # === LABEL/ORDER NORMALIZATION (build once) ===
        # use treat_col directly, but renamed to "Group"
        df_groups = ps1['meta'][[treat_col]].copy()
        df_groups.rename(columns={treat_col: "Group"}, inplace=True)
        
        _, present_order, label_map = _apply_labels_and_order(
            df=df_groups,
            col="Group",
            meta=df_groups,
            order_mode="custom",          # << use the known group_order
            custom_order=group_order,
            label_map=None,
            auto_compact=True
        )
        
        pretty = lambda s: label_map.get(str(s), str(s))
        color_map_groups = consistent_color_mapping(present_order, order=present_order)




        # --- Total reads per treatment group (sum) ---
        reads_per_sample = seqtab.sum(axis=1).rename("Reads")
        reads_by_group = (
     reads_per_sample
     .to_frame()
     .join(ps1['meta'][[treat_col]])
     .groupby(treat_col, dropna=True)["Reads"]
     .sum()
     .reset_index()
 )
 # keep same order as everywhere else
        reads_by_group['GroupPretty'] = reads_by_group[treat_col].astype(str).map(pretty)

        total_reads_fig = px.bar(
            reads_by_group,
            x='GroupPretty', y="Reads",
            category_orders={'GroupPretty': present_order},
            color='GroupPretty',
            color_discrete_map=color_map_groups,
            text="Reads",
            title="Total Reads per Treatment Group",
        )
        total_reads_fig.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        total_reads_fig.update_layout(
            xaxis_title=treat_col,
            yaxis_title="Total reads",
            showlegend=False,
            width=900, height=700,
            margin=dict(l=80, r=40, t=60, b=100),
        )
        total_reads_fig.update_yaxes(tickformat=",")
        apply_pub_style(total_reads_fig, portrait=True, base_font=16)


        # ---- Alpha Diversity (respect group order) ----
        ps1_meta = calculate_alpha_diversity(ps1, treat_col)
        ps1_meta[treat_col] = pd.Categorical(
            ps1_meta[treat_col], categories=group_order, ordered=True
        )
        ps1_meta['GroupPretty'] = ps1_meta[treat_col].astype(str).map(pretty)
        
        alpha_fig = px.violin(
            ps1_meta,
            x='GroupPretty',
            y='Shannon',
            box=True,
            points='all',
            title=f"Shannon Diversity by {treat_col}",
            category_orders={'GroupPretty': present_order},
            color='GroupPretty',
            color_discrete_map=color_map_groups,
        )
        alpha_fig.update_xaxes(categoryorder="array", categoryarray=present_order)
        apply_pub_style(alpha_fig, portrait=True, base_font=16)

        stats_df = perform_pairwise_alpha_tests(ps1_meta, treat_col)
        if not stats_df.empty:
            # Map group names to pretty labels
            stats_df_pretty = stats_df.copy()
            stats_df_pretty['group1'] = stats_df_pretty['group1'].map(pretty)
            stats_df_pretty['group2'] = stats_df_pretty['group2'].map(pretty)
        
            # Call with GroupPretty + pretty order
            try:
                alpha_fig = add_stat_annotations(
                    alpha_fig,
                    ps1_meta.assign(GroupPretty=ps1_meta[treat_col].astype(str).map(pretty)),
                    'GroupPretty',
                    stats_df_pretty,
                    group_order=present_order
                )
            except TypeError:
                alpha_fig = add_stat_annotations(
                    alpha_fig,
                    ps1_meta.assign(GroupPretty=ps1_meta[treat_col].astype(str).map(pretty)),
                    'GroupPretty',
                    stats_df_pretty
                )


                global_data['alpha_fig'] = alpha_fig

        # ---- Beta Diversity & Ordinations ----
        asv_rel, meta_rel = calculate_beta_diversity(ps1)
        # Make a copy to avoid SettingWithCopyWarning
        meta_rel = meta_rel.copy()

        pcoa_scores, dm, pcoa_var = perform_pcoa(asv_rel, meta_rel, treat_col)
        global_data['pcoa_scores'] = pcoa_scores

        # Ensure group order and categorical factors
        meta_rel[treat_col] = pd.Categorical(
            meta_rel[treat_col],
            categories=group_order,
            ordered=True,
        )
    # ---- (Always) build ordination figures ----
        if treat_col in pcoa_scores.columns:
    # ensure clean copy and consistent dtype
            pcoa_scores[treat_col] = pcoa_scores[treat_col].astype(str)
        else:
            pcoa_scores = pcoa_scores.join(meta_rel[[treat_col]].astype(str))
        pcoa_scores['GroupPretty'] = pcoa_scores[treat_col].astype(str).map(pretty)
        
        pcoa_fig = px.scatter(
            pcoa_scores,
            x='PC1', y='PC2',
            color='GroupPretty',
            title="PCoA (Bray-Curtis)",
            labels={
                "PC1": f"PC1 ({pcoa_var['PC1']*100:.2f}%)",
                "PC2": f"PC2 ({pcoa_var['PC2']*100:.2f}%)"
            },
            category_orders={'GroupPretty': present_order},
            color_discrete_map=color_map_groups,
        )
        apply_pub_style(pcoa_fig, portrait=True, base_font=16)

        global_data['pcoa_fig'] = pcoa_fig
    
        # Aitchison (CLR-Euclidean)
        pcoa_ait_scores, dm_ait, var_ait = perform_pcoa_aitchison(ps1, treat_col)
        var_vals = np.asarray(var_ait).ravel()
        pc1_lbl = f"PC1 ({(var_vals[0]*100):.2f}%)" if len(var_vals) > 0 else "PC1"
        pc2_lbl = f"PC2 ({(var_vals[1]*100):.2f}%)" if len(var_vals) > 1 else "PC2"
        if treat_col in pcoa_ait_scores.columns:
            pcoa_ait_scores[treat_col] = pcoa_ait_scores[treat_col].astype(str)
        else:
            pcoa_ait_scores = pcoa_ait_scores.join(meta_rel[[treat_col]].astype(str))
        pcoa_ait_scores['GroupPretty'] = pcoa_ait_scores[treat_col].astype(str).map(pretty)
        
        pcoa_ait_fig = px.scatter(
            pcoa_ait_scores,
            x=pcoa_ait_scores.columns[0], y=pcoa_ait_scores.columns[1],
            color='GroupPretty',
            title="PCoA (Aitchison / CLR-Euclidean)",
            labels={
                pcoa_ait_scores.columns[0]: pc1_lbl,
                pcoa_ait_scores.columns[1]: pc2_lbl
            },
            category_orders={'GroupPretty': present_order},
            color_discrete_map=color_map_groups,
        )
        apply_pub_style(pcoa_ait_fig, portrait=True, base_font=16)

        # ---- PERMANOVA (multifactor, marginal tests) ----
        permanova_res = html.Div("PERMANOVA lädt...", style={"color": "#666"})
        
        try:
            # 1. Faktoren (Hauptfaktoren)
            factors = permanova_selected_factors or _pick_multifactor_terms(meta_rel, extra=[treat_col])
            factors = [f for f in factors if f in meta_rel.columns and meta_rel[f].nunique(dropna=True) > 1]
        
            # 2. Interaktionen (z.B. "Cultivar:Media")
            interactions = []
            if permanova_interaction_input:
                for term in [t.strip() for t in permanova_interaction_input.split(",") if t.strip()]:
                    cols = [c.strip() for c in term.split(":")]
                    if len(cols) >= 2 and all(c in meta_rel.columns for c in cols):
                        interactions.append(term)
        
            # 3. Strata (Blocking-Faktor)
            strata_vec = meta_rel[permanova_selected_strata] if permanova_selected_strata in meta_rel.columns else None
        
            # 4. Wenn nichts → Warnung
            if not factors and not interactions:
                permanova_res = html.Div(
                    [
                        html.H4("PERMANOVA: Keine gültigen Faktoren", style={"color": "#e74c3c"}),
                        html.P("Wähle mindestens einen Faktor mit ≥2 Gruppen (z.B. Substrate, Cultivar)."),
                    ],
                    style={
                        "border": "3px solid #e74c3c",
                        "padding": "18px",
                        "borderRadius": "10px",
                        "background": "#fadbd8",
                        "fontSize": "1.1em",
                    },
                )
        
            else:
                logger.info(f"PERMANOVA: {factors} | {interactions} | Strata: {permanova_selected_strata}")
        
                # --- Multifaktor-PERMANOVA ---
                mf_res = permanova_marginal(
                    distance_matrix=pd.DataFrame(dm_ait.data, index=dm_ait.ids, columns=dm_ait.ids),
                    metadata_df=meta_rel,
                    factors=factors,
                    interactions=interactions,
                    permutations=permanova_num_permutations or 999,
                    strata=strata_vec,
                    random_state=42,
                )
        
                csv_path = os.path.join(out_dir, "permanova_results.csv")
                mf_res.to_csv(csv_path, index=False)
        
                # --- Fall 1: keine brauchbaren Ergebnisse ---
                if mf_res.empty or not {"term", "R2"}.issubset(mf_res.columns):
                    permanova_res = html.Div(
                        [
                            html.H4("Multifactor PERMANOVA"),
                            html.P("No testable terms (all factors had a single level or were dropped)."),
                        ],
                        style={"color": "#666"},
                    )
        
                # --- Fall 2: normale Ergebnisse ---
                else:
                    # ---------- R²-Barplot ----------
                    r2_data = mf_res[["term", "R2"]].copy()
                    r2_data["term"] = r2_data["term"].str.replace(":", " × ")
        
                    r2_fig = px.bar(
                        r2_data,
                        x="term",
                        y="R2",
                        title="explained variance (R²)",
                        color="R2",
                        color_continuous_scale="Blues",
                        text="R2",
                    )
                    r2_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                    r2_fig.update_layout(showlegend=False, height=380)
                    apply_pub_style(r2_fig)
        
                    # ---------- PERMDISP pro Faktor ----------
                    try:
                        from skbio.stats.distance import DistanceMatrix, permdisp
        
                        dm_sk = DistanceMatrix(dm_ait.data, ids=dm_ait.ids)
                        disp_texts = []
                        for factor in factors:
                            try:
                                if factor not in meta_rel.columns or meta_rel[factor].nunique(dropna=True) < 2:
                                    disp_texts.append(f"{factor}: not testable (only one level)")
                                    continue
        
                                disp = permdisp(dm_sk, meta_rel[factor].astype(str))
                                disp_texts.append(
                                    f"{factor}: F = {disp['test statistic']:.2f}, p = {disp['p-value']:.3f}"
                                )
                            except Exception as e_disp:
                                disp_texts.append(f"{factor}: failed ({e_disp})")
        
                        if disp_texts:
                            disp_text = " | ".join(disp_texts)
                        else:
                            disp_text = "PERMDISP not computed (no valid factors)"
        
                    except Exception:
                        disp_text = "PERMDISP nicht berechnet"
        
                    # ---------- Tabelle ----------
                    table = dash_table.DataTable(
                        data=mf_res[["term", "Df", "R2", "F", "p_value"]]
                        .rename(columns={"term": "Faktor", "R2": "R²", "p_value": "p"})
                        .round(4)
                        .to_dict("records"),
                        columns=[{"name": i, "id": i} for i in ["Faktor", "Df", "R²", "F", "p"]],
                        style_header={
                            "backgroundColor": "#2c3e50",
                            "color": "white",
                            "fontWeight": "bold",
                        },
                        style_cell={"padding": "10px"},
                        sort_action="native",
                    )
        
                    permanova_res = html.Div(
                        [
                            html.H4("Multifactor PERMANOVA", style={"color": "#27ae60"}),
                            html.P(f"factors: {', '.join(factors or ['–'])}"),
                            html.P(f"interactions: {', '.join(interactions) if interactions else '–'}"),
                            table,
                            html.Br(),
                            dcc.Graph(figure=r2_fig),
                            html.H5("PERMDISP (per factor)"),
                            html.P(
                                disp_text,
                                style={
                                    "background": "#ecf0f1",
                                    "padding": "8px",
                                    "fontFamily": "monospace",
                                },
                            ),
                            html.P(
                                "permanova_results.csv",
                                style={"color": "#27ae60", "fontWeight": "bold"},
                            ),
                        ]
                    )
        
        except Exception as e:
            logger.error(f"PERMANOVA Fehler: {e}", exc_info=True)
            permanova_res = html.Div(
                [
                    html.H4("PERMANOVA Fehler", style={"color": "#c0392b"}),
                    html.Pre(str(e)[:300]),
                ],
                style={
                    "border": "2px solid #c0392b",
                    "padding": "15px",
                    "background": "#fadbd8",
                },
            )

               # ---- NMDS ----
        nmds_scores, nmds_stress = (None, None)
        nmds_fig = go.Figure()

        try:
            nmds_scores, nmds_stress = perform_nmds(dm)
        except Exception as e_nmds:
            logger.warning(f"NMDS failed: {e_nmds}")

        if nmds_scores is not None:
            nmds_plot_df = nmds_scores.copy()
            if treat_col in nmds_plot_df.columns:
                nmds_plot_df[treat_col] = nmds_plot_df[treat_col].astype(str)
            else:
                nmds_plot_df = nmds_plot_df.join(ps1['meta'][[treat_col]].astype(str))

            nmds_plot_df[treat_col] = pd.Categorical(
                nmds_plot_df[treat_col], categories=group_order, ordered=True
            )
            nmds_plot_df['GroupPretty'] = nmds_plot_df[treat_col].astype(str).map(pretty)

            nmds_fig = px.scatter(
                nmds_plot_df,
                x='NMDS1', y='NMDS2',
                color='GroupPretty',
                title=f"NMDS (Stress: {nmds_stress:.4f})",
                category_orders={'GroupPretty': present_order},
                color_discrete_map=color_map_groups,
            )
            apply_pub_style(nmds_fig, portrait=True, base_font=16)
        # else: keep default empty nmds_fig



        # ---- PCA ----
        pca_res, pca_var = perform_pca(ps1['asv'], int(top_asvs))
        pca_df = pd.DataFrame(pca_res, columns=['PC1','PC2'], index=ps1['meta'].index)
        if treat_col in pca_df.columns:
            pca_df[treat_col] = pca_df[treat_col].astype(str)
        else:
            pca_df = pca_df.join(ps1['meta'][[treat_col]].astype(str))
        pca_df[treat_col] = pd.Categorical(pca_df[treat_col], categories=group_order, ordered=True)
        pca_df['GroupPretty'] = pca_df[treat_col].astype(str).map(pretty)

        pca_fig = px.scatter(
            pca_df, x='PC1', y='PC2',
            color='GroupPretty',
            title=f"PCA (Top {top_asvs} ASVs)",
            labels={
                "PC1": f"PC1 ({pca_var[0]*100:.2f}%)",
                "PC2": f"PC2 ({pca_var[1]*100:.2f}%)"
            },
            category_orders={'GroupPretty': present_order},
            color_discrete_map=color_map_groups,
        )
        apply_pub_style(pca_fig, portrait=True, base_font=16)




        # --- Plots & Stats ---
        seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth")
        apply_pub_style(seq_depth_fig, portrait=True, base_font=16)
        global_data['seq_depth_fig'] = seq_depth_fig
        # Make a shallow copy of ps1 with a pretty column
        # Make a shallow copy with pretty column
        ps1_pretty = {
            'asv': ps1['asv'],
            'tax': ps1['tax'],
            'meta': ps1['meta'].copy()
        }
        ps1_pretty['meta']['GroupPretty'] = ps1_pretty['meta'][treat_col].astype(str).map(pretty)
        
        # Order-level abundance
        try:
            abund_order_fig = plot_abundance_by_order(ps1_pretty, 'GroupPretty', group_order=present_order)
        except TypeError:
            abund_order_fig = plot_abundance_by_order(ps1_pretty, 'GroupPretty')
        global_data['abundance_order_plot'] = abund_order_fig
    
        # Genus-level abundance
        try:
            abund_genus_fig = plot_abundance_by_taxlevel(
                ps1_pretty, 'GroupPretty', tax_level="Genus", threshold=0.01, group_order=present_order
            )
        except TypeError:
            abund_genus_fig = plot_abundance_by_taxlevel(
                ps1_pretty, 'GroupPretty', tax_level="Genus", threshold=0.01
            )

        global_data['abundance_genus_plot'] = abund_genus_fig
    
        # ---- Differential abundance + ANCOM heatmap ----
        ancom_df = pd.DataFrame()
        
        # default empty heatmap (so the variable *always* exists)
        ancom_heatmap_fig = go.Figure()
        ancom_heatmap_fig.update_layout(
            title="ANCOM heatmap (no significant features or ANCOM not selected)",
            xaxis_title="Treatment",
            yaxis_title="Feature",
        )
        apply_pub_style(ancom_heatmap_fig, portrait=True, base_font=16)
        
        if da_method == 'ancom':
            try:
                ancom_df = run_ancom_skbio(ps1, treat_col, alpha=0.05)
        
                # robust: only filter if 'reject' exists and is True
                if 'reject' in ancom_df.columns:
                    sig = ancom_df[ancom_df['reject'] == True].copy()
                else:
                    sig = ancom_df.iloc[0:0].copy()
        
                cols = ['Feature_ID', 'W', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
                cols = [c for c in cols if c in sig.columns]
        
                diff_abund_res = html.Div([
                    html.H4("ANCOM (scikit-bio) significant features (reject = True)"),
                    html.P(f"Grouping: {treat_col} | alpha = 0.05 | correction = Holm–Bonferroni"),
                    html.Table(
                        [html.Thead(html.Tr([html.Th(c) for c in cols]))] +
                        [html.Tbody([
                            html.Tr([html.Td(sig.iloc[i][c]) for c in cols])
                            for i in range(min(50, len(sig)))
                        ])]
                    )
                ])
        
                out_path = os.path.join(out_dir, 'ancom_results.csv')
                ancom_df.to_csv(out_path, index=False)
        
                # ---- Heatmap: only if we actually have ANCOM results ----
                if not ancom_df.empty:
                    try:
                        # plotting.py signature:
                        # plot_ancom_clr_heatmap(ancom_df, top_k=None, clr_threshold=1.0, ...)
                        ancom_heatmap_fig = plot_ancom_clr_heatmap(
    ancom_df,
    top_k=None,
    clr_threshold=1.0,
    w_threshold=1,
    short_id_col='ASV',
    label_with_genus=True,
    group_order=group_order,   # raw group names
    label_map=label_map,       
)

                    except TypeError:
                        # fallback if your local version has no group_order argument
                        ancom_heatmap_fig = plot_ancom_clr_heatmap(
                            ancom_df,
                            top_k=None,
                            clr_threshold=1.0,
                            w_threshold=1,
                            short_id_col='ASV',
                            label_with_genus=True,
                            group_order=group_order,
    label_map=label_map,
                        )
                    apply_pub_style(ancom_heatmap_fig, portrait=True, base_font=16)
        
            except Exception as e_ancom:
                logger.error(f"ANCOM (skbio) failed: {e_ancom}", exc_info=True)
                diff_abund_res = html.Div([
                    html.H4("ANCOM (Python) failed"),
                    html.Pre(str(e_ancom), style={'whiteSpace': 'pre-wrap', 'color': '#b00'})
                ])
        
        else:
            # Kruskal–Wallis (current implementation)
            diff_abund_res = run_differential_abundance(ps1, treat_col)
            # ancom_heatmap_fig stays as the default empty figure defined above


    # ---------------- Indicator species & tree (robust) ----------------
        indic_spec_res = html.Div([html.H4("Indicator species"), html.P("No results computed.")])
        indic_df = pd.DataFrame()
        tree_img = None
    
        try:
                # Preflight checks for grouping
                if treat_col not in ps1['meta'].columns:
                    raise ValueError(f"Treatment column '{treat_col}' not found in metadata.")
            
                meta_ok = ps1['meta'].copy()
                meta_ok = meta_ok[meta_ok[treat_col].notna()]
                if meta_ok.empty:
                    raise ValueError(f"No non-NA values in '{treat_col}' after filtering/subsetting.")
    
        # Need ≥ 2 groups with ≥ 2 samples each
                counts = meta_ok[treat_col].value_counts()
                valid_groups = counts[counts >= 2].index.tolist()
                if len(valid_groups) < 2:
                    raise ValueError(
                        "Indicator species needs ≥ 2 groups with ≥ 2 samples each. "
                        f"Group sizes: {counts.to_dict()}"
                    )
    
        # Subset ps1 consistently to valid samples
                valid_samples = meta_ok.index.tolist()
                ps1_clean = {
                    'asv' : ps1['asv'].loc[:, valid_samples],
                    'tax' : ps1['tax'],
                    'meta': ps1['meta'].loc[valid_samples]
                }
    
        # Run indicator species on the clean object
                indic_spec_res, indic_df = run_indicator_species(ps1_clean, treat_col)
    
        except Exception as e_indic:
                logger.error(f"Indicator species failed: {e_indic}", exc_info=True)
                indic_spec_res = html.Div([
                    html.H4("Indicator species"),
                    html.P("Computation failed."),
                    html.Pre(str(e_indic), style={'whiteSpace': 'pre-wrap', 'color': '#b00'})
                ])
    
    # Tree: try to draw even if indicator step failed
        try:
                tree_img = plot_phylogenetic_tree(seqtab, taxa_filled, indic_df if not indic_df.empty else None)
                global_data['tree_img'] = tree_img
        except Exception as e_tree:
                logger.error(f"Tree plotting failed: {e_tree}", exc_info=True)
                tree_img = None
    
    
            # Mixed Models
        if model_type == 'pymc_zinb':
                lme_res_df = run_pymc_zinb_mixed_model(ps1, mem_treat, rand_eff, time_col, analysis_lvl, mem_top_n, mem_ref, force_feat)
        else:
                lme_res_df = run_mixed_effect_model(ps1, mem_treat, rand_eff, time_col, analysis_lvl, mem_top_n, mem_ref, force_feat)
    
        mix_model_res = format_lme_results_for_display(
                lme_res_df, ps1, mem_treat, time_col, mem_ref, mem_show_insig, force_feat
            )
        mix_model_plot = plot_lme_results(lme_res_df, mem_show_insig)
    
            # AI Interpretation & Outputs
        df_asv = ps1['asv'].T.rename_axis('SampleID').reset_index()
        global_data['ps1_melt'] = df_asv.melt(id_vars='SampleID', var_name='ASV', value_name='Abundance')
        global_data['pca_result'] = (pca_res, pca_var)
        ai_interp = ai_interpret_results(global_data, background, treat_col)
        out_files = html.Div([html.P(f) for f in os.listdir(out_dir) if f.endswith('.csv')])
    
        logger.info("Analysis completed successfully.")
    
            # SUCCESS: 17 outputs in declared order
        return (
                seq_depth_fig,        # 1  seq-depth-plot.figure
                total_reads_fig,
                alpha_fig,            # 2  alpha-diversity-plot.figure
                pcoa_fig,             # 3  pcoa-plot.figure
                pcoa_ait_fig,         # 4  pcoa-aitchison-plot.figure
                pca_fig,              # 5  pca-plot.figure
                nmds_fig,             # 6  nmds-plot.figure
                permanova_res,        # 7  permanova-results.children
                abund_order_fig,      # 8  abundance-order-plot.figure
                abund_genus_fig,      # 9  abundance-genus-plot.figure
                diff_abund_res,       # 10 differential-abundance-results.children
                ancom_heatmap_fig,    # 11 ancom-heatmap.figure
                indic_spec_res,       # 12 indicator-species-results.children
                (html.Img(src=tree_img, style={'width': '100%'})
                 if tree_img else html.P("Tree could not be generated.")),  # 13 phylogenetic-tree.children
                mix_model_res,        # 14 mixed-model-results.children
                mix_model_plot,       # 15 mixed-model-plot.figure
                ai_interp,            # 16 ai-interpretations.children
                out_files             # 17 output-files.children
            )
    
    except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            error_fig = go.Figure(layout_title_text=f"Error: {e}")
            error_msg = html.Div(
                [html.H4("Analysis Failed"), html.P(f"Details: {e}")],
                style={'color': 'red', 'fontWeight': 'bold'}
            )
            # ERROR: 17 outputs in declared order
            return (
                error_fig,  # 1  seq-depth-plot.figure
                error_fig,
                error_fig,  # 2  alpha-diversity-plot.figure
                error_fig,  # 3  pcoa-plot.figure
                error_fig,  # 4  pcoa-aitchison-plot.figure
                error_fig,  # 5  pca-plot.figure
                error_fig,  # 6  nmds-plot.figure
                error_msg,  # 7  permanova-results.children
                error_fig,  # 8  abundance-order-plot.figure
                error_fig,  # 9  abundance-genus-plot.figure
                error_msg,  # 10 differential-abundance-results.children
                error_fig,  # 11 ancom-heatmap.figure
                error_msg,  # 12 indicator-species-results.children
                error_msg,  # 13 phylogenetic-tree.children
                error_msg,  # 14 mixed-model-results.children
                error_fig,  # 15 mixed-model-plot.figure
                f"### Error\n{e}",  # 16 ai-interpretations.children
                error_msg   # 17 output-files.children
            )
    
@app.callback(
Output('download-report', 'data'),
Input('download-report-button', 'n_clicks'),
    State('output-dir', 'value'),
    State('ai-interpretations', 'children'),
    prevent_initial_call=True
)
def download_pdf_report(n_clicks, output_dir, interpretations_md):
        if n_clicks > 0:
            interpretations = interpretations_md or "No interpretation generated."
            report_path = generate_pdf_report(global_data, interpretations, output_dir)
            if report_path:
                return dcc.send_file(report_path)
        return None



@app.callback(
    [Output('data-folder-path', 'value'),
     Output('data-folder-path', 'disabled')],
    Input('load-sample-data', 'n_clicks'),
    prevent_initial_call=True
)
def load_sample_data(n_clicks):
    if n_clicks > 0 and ctx.triggered_id == 'load-sample-data':
        global_data['sample_data_mode'] = True
        logger.info("Sample data mode activated.")
        return "Using internal sample data", True
    return dash.no_update

