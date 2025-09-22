
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
from .analysis import (create_phyloseq_object, calculate_alpha_diversity, calculate_beta_diversity, 
                       perform_pcoa, perform_pca, perform_nmds)
from .statistics import (perform_pairwise_alpha_tests, run_permanova, run_differential_abundance, 
                         run_indicator_species, run_mixed_effect_model, run_pymc_zinb_mixed_model)
from .plotting import (add_stat_annotations, plot_phylogenetic_tree, plot_abundance_by_order, 
                       plot_lme_results, format_lme_results_for_display)
from .reporting import generate_pdf_report
from .utils import fill_taxonomy_forward

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
                    dcc.Graph(id='pca-plot'),
                    dcc.Graph(id='nmds-plot'),
                    html.H3("Taxonomic Composition"),
                    dcc.Graph(id='abundance-order-plot'),
                    html.H3("Statistical Comparisons"),
                    html.Div(id='differential-abundance-results'),
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
    [Output('seq-depth-plot', 'figure'),
     Output('alpha-diversity-plot', 'figure'),
     Output('pcoa-plot', 'figure'),
     Output('pca-plot', 'figure'),
     Output('nmds-plot', 'figure'),
     Output('permanova-results', 'children'),
     Output('abundance-order-plot', 'figure'),
     Output('differential-abundance-results', 'children'),
     Output('indicator-species-results', 'children'),
     Output('phylogenetic-tree', 'children'),
     Output('mixed-model-results', 'children'),
     Output('mixed-model-plot', 'figure'),
     Output('ai-interpretations', 'children'),
     Output('output-files', 'children')],
    Input('run-analysis', 'n_clicks'),
    [State('analysis-mode', 'value'), State('data-folder-path', 'value'), State('output-dir', 'value'),
     State('trunc-len-f', 'value'), State('trunc-len-r', 'value'), State('max-ee', 'value'),
     State('treatment-group', 'value'), State('subset-groups-dropdown', 'value'), State('top-asvs', 'value'),
     State('background-info', 'value'), State('upload-silva', 'contents'), State('model-type-dropdown', 'value'),
     State('mem-treatment-col', 'value'), State('time-col', 'value'), State('random-effect-cols', 'value'),
     State('analysis-level-dropdown', 'value'), State('mem-top-n-features', 'value'),
     State('mem-reference-group-input', 'value'), State('show-insignificant', 'value'), State('force_features', 'value')],
    prevent_initial_call=True
)
def run_full_analysis(n_clicks, analysis_mode, data_path, out_dir, trunc_f, trunc_r, max_ee, treat_col,
                      subset, top_asvs, background, silva_content, model_type, mem_treat, time_col,
                      rand_eff, analysis_lvl, mem_top_n, mem_ref, mem_show_insig, force_feat):

    if n_clicks == 0:
        return [go.Figure()] * 7 + [html.Div()] * 3 + [html.Div(), go.Figure()] + [dcc.Markdown(), html.Div()]

    try:
        global_data.clear() # Reset data for new run
        os.makedirs(out_dir, exist_ok=True)

        # --- DATA LOADING ---
        use_sample_data = 'sample_data_mode' in global_data

        if use_sample_data:
            logger.info("Using internal sample data for analysis.")
            seqtab, taxa_df, meta_df = sample_seqtab.copy(), sample_taxa.copy(), sample_metadata_df.copy()
        else:
            if not os.path.isdir(data_path): raise FileNotFoundError(f"Data folder '{data_path}' not found.")
            files = os.listdir(data_path)
            meta_file = next((f for f in files if f.lower().endswith(('.csv', '.tsv'))), None)
            if not meta_file: raise FileNotFoundError("Metadata file not found in data folder.")
            meta_df = pd.read_csv(os.path.join(data_path, meta_file), index_col=0)
            meta_df.index = meta_df.index.astype(str)

            if analysis_mode == 'fastq':
                logger.info("Starting analysis from FASTQ files.")
                fnFs = sorted([os.path.join(data_path, f) for f in files if '_R1' in f.upper()])
                fnRs = sorted([os.path.join(data_path, f) for f in files if '_R2' in f.upper()])
                s_names = [os.path.basename(f).split('_')[0] for f in fnFs]
                meta_df = meta_df.loc[s_names].copy()
                filtFs, filtRs = filter_and_trim_parallel(fnFs, fnRs, s_names, out_dir, trunc_f, trunc_r, max_ee)
                seqtab = denoise_and_create_asv_table_vsearch(filtFs, filtRs, s_names, out_dir)
                silva_path = os.path.join(data_path, 'silva.fasta')
                if silva_content:
                    _, content_string = silva_content.split(',')
                    silva_path = os.path.join(out_dir, 'uploaded_silva.fasta')
                    with open(silva_path, 'wb') as f: f.write(base64.b64decode(content_string))
                taxa_df = assign_taxonomy(list(seqtab.columns), os.path.join(out_dir, 'asvs.fa'), silva_path, out_dir)
            else: # ASV mode
                logger.info("Starting analysis from pre-processed tables.")
                asv_path, taxa_path = os.path.join(out_dir, 'microbiome_ai_16s_asv.csv'), os.path.join(out_dir, 'microbiome_ai_taxonomy.csv')
                if not (os.path.exists(asv_path) and os.path.exists(taxa_path)):
                    raise FileNotFoundError("Run in FASTQ mode first to generate ASV/Taxonomy tables in the output directory.")
                seqtab = pd.read_csv(asv_path, sep='\t', index_col=0, engine='python')
                taxa_df = pd.read_csv(taxa_path, index_col=0)
                meta_df = meta_df.loc[seqtab.index].copy()

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
        if not stats_df.empty: alpha_fig = add_stat_annotations(alpha_fig, ps1_meta, treat_col, stats_df)
        global_data['alpha_fig'] = alpha_fig

        # Beta Diversity & Ordinations
        asv_rel, meta_rel = calculate_beta_diversity(ps1)
        pcoa_scores, dm, pcoa_var = perform_pcoa(asv_rel, meta_rel, treat_col)
        global_data['pcoa_scores'] = pcoa_scores
        permanova_res = run_permanova(dm, meta_rel, treat_col)
        pcoa_fig = px.scatter(pcoa_scores, x='PC1', y='PC2', color=treat_col, title="PCoA (Bray-Curtis)", labels={"PC1": f"PC1 ({pcoa_var['PC1']*100:.2f}%)", "PC2": f"PC2 ({pcoa_var['PC2']*100:.2f}%)"})
        global_data['pcoa_fig'] = pcoa_fig

        nmds_scores, nmds_stress = perform_nmds(dm)
        nmds_fig = px.scatter(nmds_scores.join(ps1['meta'][[treat_col]]), x='NMDS1', y='NMDS2', color=treat_col, title=f"NMDS (Stress: {nmds_stress:.4f})") if nmds_scores is not None else go.Figure(layout_title_text="NMDS Failed")

        pca_res, pca_var = perform_pca(ps1['asv'], int(top_asvs))
        pca_df = pd.DataFrame(pca_res, columns=['PC1', 'PC2'], index=ps1['meta'].index).join(ps1['meta'][treat_col])
        pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color=treat_col, title=f"PCA (Top {top_asvs} ASVs)", labels={"PC1": f"PC1 ({pca_var[0]*100:.2f}%)", "PC2": f"PC2 ({pca_var[1]*100:.2f}%)"})

        # Plots & Stats
        seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth")
        global_data['seq_depth_fig'] = seq_depth_fig
        abund_order_fig = plot_abundance_by_order(ps1, treat_col)
        global_data['abundance_order_plot'] = abund_order_fig

        diff_abund_res = run_differential_abundance(ps1, treat_col)
        indic_spec_res, indic_df = run_indicator_species(ps1, treat_col)
        tree_img = plot_phylogenetic_tree(seqtab, taxa_filled, indic_df)
        global_data['tree_img'] = tree_img

        # Mixed Models
        if model_type == 'pymc_zinb':
            lme_res_df = run_pymc_zinb_mixed_model(ps1, mem_treat, rand_eff, time_col, analysis_lvl, mem_top_n, mem_ref, force_feat)
        else:
            lme_res_df = run_mixed_effect_model(ps1, mem_treat, rand_eff, time_col, analysis_lvl, mem_top_n, mem_ref, force_feat)
        mix_model_res = format_lme_results_for_display(lme_res_df, ps1, mem_treat, time_col, mem_ref, mem_show_insig, force_feat)
        mix_model_plot = plot_lme_results(lme_res_df, mem_show_insig)

        # AI Interpretation & Outputs
        global_data['ps1_melt'] = ps1['asv'].T.reset_index().melt(id_vars=['SampleID'], var_name='ASV', value_name='Abundance')
        global_data['pca_result'] = (pca_res, pca_var)
        ai_interp = ai_interpret_results(global_data, background, treat_col)
        out_files = html.Div([html.P(f) for f in os.listdir(out_dir) if f.endswith('.csv')])

        logger.info("Analysis completed successfully.")
        return (seq_depth_fig, alpha_fig, pcoa_fig, pca_fig, nmds_fig, permanova_res, abund_order_fig,
                diff_abund_res, indic_spec_res, html.Img(src=tree_img, style={'width': '100%'}) if tree_img else html.P("Tree could not be generated."),
                mix_model_res, mix_model_plot, dcc.Markdown(ai_interp), out_files)

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        error_fig = go.Figure(layout_title_text=f"Error: {e}")
        error_msg = html.Div([html.H4("Analysis Failed"), html.P(f"Details: {e}")], style={'color': 'red', 'fontWeight': 'bold'})
        empty_outputs = [go.Figure()] * 7 + [error_msg] * 3 + [error_msg, go.Figure()] + [dcc.Markdown(f"### Error\n{e}"), error_msg]
        return empty_outputs

@app.callback(
    Output('download-report', 'data'),
    Input('download-report-button', 'n_clicks'),
    State('output-dir', 'value'),
    State('ai-interpretations', 'children'),
    prevent_initial_call=True
)
def download_pdf_report(n_clicks, output_dir, interpretations_md):
    if n_clicks > 0:
        interpretations = interpretations_md['props']['children'] if interpretations_md and 'props' in interpretations_md else "No interpretation generated."
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
