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
from .statistics import (perform_pairwise_alpha_tests, run_permanova, run_differential_abundance, 
                         run_indicator_species, run_mixed_effect_model, run_pymc_zinb_mixed_model, run_ancom_skbio)
from .plotting import (add_stat_annotations, plot_phylogenetic_tree, plot_abundance_by_order, plot_abundance_by_taxlevel, plot_ancom_clr_heatmap, 
                       plot_lme_results, format_lme_results_for_display)
from .reporting import generate_pdf_report
from .utils import fill_taxonomy_forward
from dash import no_update
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

# --- Callbacks ---

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
    ],
    prevent_initial_call=True,
)
def run_full_analysis(n_clicks, analysis_mode, data_path, out_dir, trunc_f, trunc_r, max_ee, treat_col,
                      subset, top_asvs, background, silva_content, model_type, mem_treat, time_col,
                      rand_eff, analysis_lvl, mem_top_n, mem_ref, mem_show_insig, force_feat, da_method):
    if n_clicks == 0:
        empty_fig = go.Figure()
        empty_div = html.Div()
        return [
            empty_fig,  # 1 seq-depth-plot
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
        ps1 = ps1_full
        if subset:
            meta_subset = ps1_full['meta'][ps1_full['meta'][treat_col].isin(subset)]
            ps1 = {'asv': ps1_full['asv'].loc[:, meta_subset.index], 'tax': ps1_full['tax'], 'meta': meta_subset}
        global_data['ps1'] = ps1

        # Alpha Diversity
        ps1_meta = calculate_alpha_diversity(ps1, treat_col)
        alpha_fig = px.violin(ps1_meta, x=treat_col, y='Shannon', box=True, points='all', title=f"Shannon Diversity by {treat_col}")
        stats_df = perform_pairwise_alpha_tests(ps1_meta, treat_col)
        if not stats_df.empty:
            alpha_fig = add_stat_annotations(alpha_fig, ps1_meta, treat_col, stats_df)
        global_data['alpha_fig'] = alpha_fig

        # Beta Diversity & Ordinations
        asv_rel, meta_rel = calculate_beta_diversity(ps1)
        pcoa_scores, dm, pcoa_var = perform_pcoa(asv_rel, meta_rel, treat_col)
        global_data['pcoa_scores'] = pcoa_scores
        permanova_res = run_permanova(dm, meta_rel, treat_col)
        pcoa_fig = px.scatter(
            pcoa_scores, x='PC1', y='PC2', color=treat_col,
            title="PCoA (Bray-Curtis)",
            labels={"PC1": f"PC1 ({pcoa_var['PC1']*100:.2f}%)",
                    "PC2": f"PC2 ({pcoa_var['PC2']*100:.2f}%)"}
        )
        global_data['pcoa_fig'] = pcoa_fig

        # Aitchison (CLR-Euclidean) PCoA
        pcoa_ait_scores, dm_ait, var_ait = perform_pcoa_aitchison(ps1, treat_col)
        var_vals = np.asarray(var_ait).ravel()  # robust to Series/array/list
        pc1_lbl = f"PC1 ({(var_vals[0]*100):.2f}%)" if len(var_vals) > 0 else "PC1"
        pc2_lbl = f"PC2 ({(var_vals[1]*100):.2f}%)" if len(var_vals) > 1 else "PC2"
        pcoa_ait_fig = px.scatter(
            pcoa_ait_scores,
            x=pcoa_ait_scores.columns[0],
            y=pcoa_ait_scores.columns[1],
            color=treat_col,
            title="PCoA (Aitchison / CLR-Euclidean)",
            labels={pcoa_ait_scores.columns[0]: pc1_lbl,
                    pcoa_ait_scores.columns[1]: pc2_lbl}
        )

        # NMDS
        nmds_scores, nmds_stress = perform_nmds(dm)
        nmds_fig = (
            px.scatter(
                nmds_scores.join(ps1['meta'][[treat_col]]),
                x='NMDS1', y='NMDS2', color=treat_col,
                title=f"NMDS (Stress: {nmds_stress:.4f})"
            )
            if nmds_scores is not None else go.Figure(layout_title_text="NMDS Failed")
        )

        # PCA
        pca_res, pca_var = perform_pca(ps1['asv'], int(top_asvs))
        pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'], index=ps1['meta'].index).join(ps1['meta'][treat_col])
        pca_fig = px.scatter(
            pca_df, x='PC1', y='PC2', color=treat_col,
            title=f"PCA (Top {top_asvs} ASVs)",
            labels={"PC1": f"PC1 ({pca_var[0]*100:.2f}%)",
                    "PC2": f"PC2 ({pca_var[1]*100:.2f}%)"}
        )

        # --- Plots & Stats ---
        seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth")
        global_data['seq_depth_fig'] = seq_depth_fig

        abund_order_fig = plot_abundance_by_order(ps1, treat_col)
        global_data['abundance_order_plot'] = abund_order_fig

        abund_genus_fig = plot_abundance_by_taxlevel(ps1, treat_col, tax_level="Genus", threshold=0.01)
        global_data['abundance_genus_plot'] = abund_genus_fig

        ancom_df = pd.DataFrame()  # default empty

        if da_method == 'ancom':
            try:
                ancom_df = run_ancom_skbio(ps1, treat_col, alpha=0.05)

                # robust: only filter if 'reject' exists and is True
                if 'reject' in ancom_df.columns:
                    sig = ancom_df[ancom_df['reject'] == True].copy()
                else:
                    sig = ancom_df.iloc[0:0].copy()  # empty

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

            except Exception as e_ancom:
                logger.error(f"ANCOM (skbio) failed: {e_ancom}", exc_info=True)
                diff_abund_res = html.Div([
                    html.H4("ANCOM (Python) failed"),
                    html.Pre(str(e_ancom), style={'whiteSpace': 'pre-wrap', 'color': '#b00'})
                ])
        else:
            diff_abund_res = run_differential_abundance(ps1, treat_col)

        # Heatmap (guarded)
        if not ancom_df.empty:
            ancom_heatmap_fig = plot_ancom_clr_heatmap(ancom_df, tax_level_cols=('Genus', 'Species'), top_k=30)
        else:
            ancom_heatmap_fig = go.Figure()
            ancom_heatmap_fig.update_layout(
                title="ANCOM heatmap (run with DA method = 'ANCOM' to populate)",
                xaxis_title="Treatment",
                yaxis_title="Feature"
            )

        # Indicator species & tree
        indic_spec_res, indic_df = run_indicator_species(ps1, treat_col)
        tree_img = plot_phylogenetic_tree(seqtab, taxa_filled, indic_df)
        global_data['tree_img'] = tree_img

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
