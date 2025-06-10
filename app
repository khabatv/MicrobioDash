#Run it  and access it on http://127.0.0.1:8050/
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa
from skbio import DistanceMatrix
import pandas as pd
import numpy as np
import os
from Bio import SeqIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import gzip
import zipfile
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from io import StringIO, BytesIO
import base64
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from ete3 import Tree, TreeStyle
import tempfile
from skbio.tree import nj
from skbio import DNA
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyDiuCPQ8vYm9XLxB4yTSh4H1fBXxVcRUhY"
ai_available = False
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    test_response = model.generate_content("Test connection", generation_config={'max_output_tokens': 5})
    if test_response and hasattr(test_response, 'text'):
        print("Gemini API configured and tested successfully.")
        ai_available = True
    else:
        print("Gemini API configured, but failed a basic text generation test.")
        ai_available = True
except Exception as e:
    print(f"Error configuring or testing Gemini API: {e}")
    ai_available = False

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Microbiome Analysis Dashboard with AI Parameter Optimization and Reporting"),
    
    # Input Section
    html.H3("Input Parameters"),
    html.Label("Upload FASTQ Forward Reads (R1):"),
    dcc.Upload(id='upload-fastq-r1', children=html.Button('Upload R1 Files'), multiple=True),
    html.Label("Upload FASTQ Reverse Reads (R2):"),
    dcc.Upload(id='upload-fastq-r2', children=html.Button('Upload R2 Files'), multiple=True),
    html.Label("Upload Metadata CSV:"),
    dcc.Upload(id='upload-metadata', children=html.Button('Upload Metadata')),
    html.Label("Upload SILVA Taxonomy Database:"),
    dcc.Upload(id='upload-silva', children=html.Button('Upload SILVA Database')),
    html.Label("Output Directory:"),
    dcc.Input(id='output-dir', value='./output', type='text'),
    html.Label("Background Information (e.g., organism, experiment design, special analysis needs):"),
    dcc.Textarea(id='background-info', value='', style={'width': '100%', 'height': 100}),
    
    # AI Clarification Questions
    html.Div(id='ai-clarification-questions', children=[]),
    
    # Parameter Mode Selection
    html.H3("Parameter Setting Mode"),
    dcc.Dropdown(
        id='param-mode',
        options=[
            {'label': 'Manual', 'value': 'manual'},
            {'label': 'AI-Suggested', 'value': 'ai_suggested'},
            {'label': 'AI-Automatic', 'value': 'ai_automatic'}
        ],
        value='manual',
        clearable=False
    ),
    
    # Parameter Tuning
    html.Div(id='manual-params', children=[
        html.H3("Analysis Parameters"),
        html.Label("Truncation Length Forward (e.g., 280):"),
        dcc.Input(id='trunc-len-f', value=280, type='number'),
        html.Label("Truncation Length Reverse (e.g., 220):"),
        dcc.Input(id='trunc-len-r', value=220, type='number'),
        html.Label("Max Expected Errors (Forward, Reverse):"),
        dcc.Input(id='max-ee', value='2,2', type='text'),
        html.Label("Detection Threshold for Prevalence Filtering:"),
        dcc.Input(id='prev-threshold', value=0, type='number'),
        html.Label("Treatment Group Column:"),
        dcc.Input(id='treatment-group', value='Cultivar_Media_Run_Substrate', type='text'),
        html.Label("Top ASVs for PCA (e.g., 20,50,100):"),
        dcc.Input(id='top-asvs', value='20,50,100', type='text'),
        html.Label("P-Value Threshold for PCoA Vectors:"),
        dcc.Input(id='pval-threshold', value=0.005, type='number'),
        html.Label("Contribution Threshold for PCoA Vectors:"),
        dcc.Input(id='contrib-threshold', value=0.65, type='number'),
    ]),
    
    html.Div(id='ai-suggested-params', style={'display': 'none'}, children=[
        html.H3("AI-Suggested Parameters"),
        html.Label("Select AI-Suggested Truncation Length Forward:"),
        dcc.Dropdown(id='ai-trunc-len-f', options=[], value=None),
        html.Label("Select AI-Suggested Truncation Length Reverse:"),
        dcc.Dropdown(id='ai-trunc-len-r', options=[], value=None),
        html.Label("Select AI-Suggested Max Expected Errors:"),
        dcc.Dropdown(id='ai-max-ee', options=[], value=None),
        html.Label("Select AI-Suggested Prevalence Threshold:"),
        dcc.Dropdown(id='ai-prev-threshold', options=[], value=None),
        html.Label("Select AI-Suggested Treatment Group:"),
        dcc.Dropdown(id='ai-treatment-group', options=[], value=None),
        html.Label("Select AI-Suggested Top ASVs for PCA:"),
        dcc.Dropdown(id='ai-top-asvs', options=[], value=None),
        html.Label("Select AI-Suggested P-Value Threshold:"),
        dcc.Dropdown(id='ai-pval-threshold', options=[], value=None),
        html.Label("Select AI-Suggested Contribution Threshold:"),
        dcc.Dropdown(id='ai-contrib-threshold', options=[], value=None),
    ]),
    
    # Run Analysis Button
    html.Button('Run Analysis', id='run-analysis', n_clicks=0),
    
    # Output Section
    html.H3("Results"),
    dcc.Graph(id='seq-depth-plot'),
    dcc.Graph(id='alpha-diversity-plot'),
    dcc.Graph(id='beta-diversity-plot'),
    dcc.Graph(id='pca-plot'),
    dcc.Graph(id='pcoa-plot'),
    dcc.Graph(id='abundance-plot'),
    html.Div(id='output-files'),
    html.Div(id='phylogenetic-tree'),
    html.Div(id='ai-interpretations'),
    dcc.Download(id='download-report'),
    html.Button('Download Report', id='download-report-button', n_clicks=0),
])

# Global variables
global_data = {
    'seqtab_nochim': None,
    'taxa': None,
    'metadata': None,
    'ps': None,
    'ps1': None,
    'pseq_rel': None,
    'otu_rel': None,
    'meta_rel': None,
    'otu_absolute': None,
    'meta_absolute': None,
    'indval_table': None,
    'tax_df': None,
    'treeNJ': None,
    'ai_interpretations': None,
    'pca_result': None,
    'explained_variance': None,
}

# AI Functions
def ai_analyze_background(background):
    if not ai_available or not background:
        return None, []
    prompt = f"""
    Analyze the following background information for a microbiome study:
    {background}
    Determine if the information is sufficient for analysis. If not, provide up to 3 specific questions to gather more details, each with 2-4 suggested answer options. Return a JSON object:
    {{
        "sufficient": true/false,
        "questions": [
            {{"question": "text", "options": ["option1", "option2", ...]}},
            ...
        ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            import json
            return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        print(f"AI background analysis failed: {e}")
        return None, []
    return None, []

def ai_analyze_quality_profiles(fnFs, fnRs):
    if not ai_available:
        return None
    quality_summary = []
    for fnF, fnR in zip(fnFs[:5], fnRs[:5]):
        records_f = list(SeqIO.parse(gzip.open(fnF, 'rt'), 'fastq'))
        records_r = list(SeqIO.parse(gzip.open(fnR, 'rt'), 'fastq'))
        qual_f = [np.mean(r.letter_annotations['phred_quality']) for r in records_f]
        qual_r = [np.mean(r.letter_annotations['phred_quality']) for r in records_r]
        quality_summary.append({
            'forward_mean_qual': np.mean(qual_f),
            'reverse_mean_qual': np.mean(qual_r),
            'forward_length': np.mean([len(r) for r in records_f]),
            'reverse_length': np.mean([len(r) for r in records_r])
        })
    prompt = f"""
    Given the quality profile summary:
    {quality_summary}
    Suggest optimal truncation lengths and max expected errors.
    Aim for quality scores >30 and retain 70% of read length.
    Return: {{'trunc_len_f': X, 'trunc_len_r': Y, 'max_ee': [A, B]}}
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            import json
            return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        print(f"AI quality profile analysis failed: {e}")
        return None
    return None

def ai_analyze_prevalence(seqtab):
    if not ai_available:
        return None
    prev = seqtab.sum(axis=0)
    prompt = f"""
    Given ASV prevalence (sum of reads):
    Mean: {prev.mean()}, Median: {prev.median()}, Min: {prev.min()}, Max: {prev.max()}
    Suggest a prevalence threshold for filtering rare ASVs (retain ASVs in >=1% of reads/samples).
    Return a single value.
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            return float(response.text.strip())
    except Exception as e:
        print(f"AI prevalence analysis failed: {e}")
        return None
    return None

def ai_analyze_metadata(metadata):
    if not ai_available:
        return None
    columns = metadata.columns.tolist()
    prompt = f"""
    Given metadata columns: {columns}
    Suggest the most appropriate column for grouping samples (categorical, 3-10 unique values).
    Return a single column name.
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            return response.text.strip()
    except Exception as e:
        print(f"AI metadata analysis failed: {e}")
        return None
    return None

def ai_analyze_pca_pcoa(otu, ordination_scores):
    if not ai_available:
        return None
    otu_sums = otu.sum()
    prompt = f"""
    Given ASV abundance sums (mean: {otu_sums.mean()}, median: {otu_sums.median()})
    and ordination variance explained (first two axes: {ordination_scores[:2]}),
    suggest top ASVs for PCA and thresholds for PCoA vectors.
    Return: {{'top_asvs': 'X,Y,Z', 'pval_threshold': A, 'contrib_threshold': B}}
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            import json
            return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        print(f"AI PCA/PCoA analysis failed: {e}")
        return None
    return None

def ai_interpret_results(seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt, tree_img, background):
    if not ai_available:
        return "AI interpretation unavailable."
    prompt = f"""
    Interpret the following microbiome analysis results for a study with background: {background}
    - Sequencing Depth: Mean={seqtab.sum(axis=1).mean()}, Median={seqtab.sum(axis=1).median()}
    - Alpha Diversity (Shannon): Mean={ps1_meta['Shannon'].mean()}, Groups={ps1_meta['Cultivar_Media_Run_Substrate'].nunique()}
    - Beta Diversity (PCoA): Variance explained={pcoa_scores.var().values[:2]}
    - PCA: Variance explained={pca_result[1][:2]}
    - Abundance: Top phyla={ps1_melt.groupby('ASV')['Abundance'].sum().nlargest(5).index.tolist()}
    - Phylogenetic Tree: Generated successfully.
    Provide a detailed interpretation of each analysis, including visualizations and tables, comparing results to typical microbiome studies (e.g., soil, marine, gut microbiomes). Include citations to relevant literature (e.g., from PubMed or Google Scholar). Format as markdown.
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            return response.text
    except Exception as e:
        print(f"AI interpretation failed: {e}")
        return "AI interpretation failed."
    return "AI interpretation unavailable."

def generate_pdf_report(seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt, tree_img, interpretations, output_dir):
    report_path = os.path.join(output_dir, 'microbiome_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    elements.append(Paragraph("Microbiome Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    elements.append(Paragraph("Sequencing Depth", styles['Heading2']))
    fig = go.Figure(px.histogram(seqtab.sum(axis=1), title="Sequencing Depth"))
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("Alpha Diversity", styles['Heading2']))
    fig = go.Figure(px.violin(ps1_meta, x='Cultivar_Media_Run_Substrate', y='Shannon', box=True, title='Shannon Diversity'))
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("Beta Diversity (PCoA)", styles['Heading2']))
    fig = go.Figure(px.scatter(pcoa_scores, x=0, y=1, color='Cultivar_Media_Run_Substrate', title='PCoA Plot'))
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("PCA", styles['Heading2']))
    fig = go.Figure(px.scatter(x=pca_result[0][:,0], y=pca_result[0][:,1], title='PCA Plot'))
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("Abundance by Phylum", styles['Heading2']))
    fig = go.Figure(px.bar(ps1_melt, x='index', y='Abundance', color='ASV', facet_col='Cultivar_Media_Run_Substrate', title='Abundance by Phylum'))
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("Phylogenetic Tree", styles['Heading2']))
    tree_img_data = base64.b64decode(tree_img.split(',')[1])
    img_buffer = BytesIO(tree_img_data)
    elements.append(Image(img_buffer, width=400, height=300))
    
    elements.append(Paragraph("AI Interpretations", styles['Heading2']))
    elements.append(Paragraph(interpretations.replace('\n', '<br>'), styles['BodyText']))
    
    elements.append(Paragraph("Output Files", styles['Heading2']))
    elements.append(Paragraph(f"ASV Table: {os.path.join(output_dir, 'microbiomeAnalyst_16s_otu.txt')}", styles['BodyText']))
    elements.append(Paragraph(f"Taxonomy Table: {os.path.join(output_dir, 'microbiomeAnalyst_16s_taxa.txt')}", styles['BodyText']))
    
    doc.build(elements)
    return report_path

# Helper functions
def unzip_files(uploaded_files, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file in uploaded_files:
        with open(os.path.join(output_dir, f"temp_{uuid.uuid4()}.zip"), 'wb') as f:
            f.write(base64.b64decode(file.split(',')[1]))
        with zipfile.ZipFile(f.name, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(f.name)

def filter_and_trim(fnFs, fnRs, sample_names, output_dir, trunc_len_f, trunc_len_r, max_ee):
    filt_path = os.path.join(output_dir, "filtered")
    if not os.path.exists(filt_path):
        os.makedirs(filt_path)
    filtFs = [os.path.join(filt_path, f"{name}_F_filt.fastq.gz") for name in sample_names]
    filtRs = [os.path.join(filt_path, f"{name}_R_filt.fastq.gz") for name in sample_names]
    
    for fnF, fnR, filtF, filtR in zip(fnFs, fnRs, filtFs, filtRs):
        records_f = list(SeqIO.parse(gzip.open(fnF, 'rt'), 'fastq'))
        records_r = list(SeqIO.parse(gzip.open(fnR, 'rt'), 'fastq'))
        filtered_f = [r for r in records_f if len(r) >= trunc_len_f and np.mean(r.letter_annotations['phred_quality']) >= 30]
        filtered_r = [r for r in records_r if len(r) >= trunc_len_r and np.mean(r.letter_annotations['phred_quality']) >= 30]
        filtered_f = [r[:trunc_len_f] for r in filtered_f]
        filtered_r = [r[:trunc_len_r] for r in filtered_r]
        with gzip.open(filtF, 'wt') as f:
            SeqIO.write(filtered_f[:min(len(filtered_f), len(filtered_r))], f, 'fastq')
        with gzip.open(filtR, 'wt') as f:
            SeqIO.write(filtered_r[:min(len(filtered_f), len(filtered_r))], f, 'fastq')
    return filtFs, filtRs

def process_sequences(filtFs, filtRs, sample_names, silva_train, output_dir):
    seqtab = []
    for filtF, filtR in zip(filtFs, filtRs):
        records_f = list(SeqIO.parse(gzip.open(filtF, 'rt'), 'fastq'))
        records_r = list(SeqIO.parse(gzip.open(filtR, 'rt'), 'fastq'))
        seqs = [str(r.seq) for r in records_f]
        counts = pd.Series(seqs).value_counts()
        seqtab.append(counts)
    seqtab = pd.DataFrame(seqtab, index=sample_names).fillna(0)
    
    taxa = []
    for seq in seqtab.columns:
        taxa.append(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
    taxa = pd.DataFrame(taxa, index=seqtab.columns, columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
    
    seqtab.to_csv(os.path.join(output_dir, 'microbiomeAnalyst_16s_otu.txt'), sep='\t')
    taxa.to_csv(os.path.join(output_dir, 'microbiomeAnalyst_16s_taxa.txt'), sep='\t')
    
    return seqtab, taxa

def create_phyloseq_object(seqtab, taxa, metadata):
    otu = seqtab.T
    tax = taxa
    meta = metadata
    global_data['ps'] = {'otu': otu, 'tax': tax, 'meta': meta}
    global_data['ps1'] = global_data['ps']
    global_data['seqtab_nochim'] = seqtab
    global_data['taxa'] = taxa
    global_data['metadata'] = metadata
    
    global_data['ps1'] = {'otu': otu[~tax['Order'].isin(['Chloroplast']) & ~tax['Family'].isin(['Mitochondria'])],
                          'tax': tax[~tax['Order'].isin(['Chloroplast']) & ~tax['Family'].isin(['Mitochondria'])],
                          'meta': meta}
    return global_data['ps1']

def calculate_alpha_diversity(ps1, treatment):
    meta = ps1['meta'].copy()
    otu = ps1['otu']
    shannon = alpha_diversity('shannon', otu, ids=otu.index)
    simpson = alpha_diversity('simpson', otu, ids=otu.index)
    meta['Shannon'] = shannon
    meta['InverseSimpson'] = 1 / (1 - simpson)
    meta[treatment] = meta[treatment].astype(str)
    global_data['ps1.meta'] = meta
    return meta

def calculate_beta_diversity(ps1):
    otu = ps1['otu']
    otu_rel = otu.div(otu.sum(axis=1), axis=0)
    global_data['pseq_rel'] = {'otu': otu_rel, 'tax': ps1['tax'], 'meta': ps1['meta']}
    global_data['otu_rel'] = otu_rel
    global_data['meta_rel'] = ps1['meta']
    global_data['otu_absolute'] = otu
    global_data['meta_absolute'] = ps1['meta']
    return otu_rel, ps1['meta']

def perform_pca(otu, top_n):
    from sklearn.decomposition import PCA
    top_asvs = otu.sum().nlargest(top_n).index
    otu_top = otu[top_asvs]
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(otu_top)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

def perform_pcoa(otu, meta, treatment, pval_threshold, contrib_threshold):
    dm = beta_diversity('braycurtis', otu)
    ordination = pcoa(dm)
    scores = pd.DataFrame(ordination.samples, index=otu.index)
    scores = scores.join(meta[[treatment]])
    return scores, dm

def plot_phylogenetic_tree(seqtab):
    seqs = [SeqRecord(Seq(seq), id=f"ASV{i+1}") for i, seq in enumerate(seqtab.columns)]
    aligner = Align.PairwiseAligner()
    alignments = []
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            alignments.append(aligner.align(seqs[i].seq, seqs[j].seq)[0])
    dm = np.ones((len(seqs), len(seqs)))
    for i in range(len(seqs)):
        dm[i, i] = 0
    dm = DistanceMatrix(dm, ids=[s.id for s in seqs])
    tree = nj(dm)
    global_data['treeNJ'] = tree
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tree.render(tmp.name, w=400, units='px')
        with open(tmp.name, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
        os.remove(tmp.name)
    return f"data:image/png;base64,{encoded}"

# Callbacks
@app.callback(
    [Output('manual-params', 'style'),
     Output('ai-suggested-params', 'style')],
    [Input('param-mode', 'value')]
)
def toggle_param_inputs(param_mode):
    if param_mode == 'manual':
        return {'display': 'block'}, {'display': 'none'}
    elif param_mode == 'ai_suggested':
        return {'display': 'block'}, {'display': 'block'}
    else:
        return {'display': 'none'}, {'display': 'none'}

@app.callback(
    Output('ai-clarification-questions', 'children'),
    [Input('background-info', 'value')]
)
def update_clarification_questions(background):
    if not background:
        return []
    analysis, questions = ai_analyze_background(background)
    if analysis and not analysis.get('sufficient', True):
        children = []
        for q in questions:
            children.append(html.Label(q['question']))
            children.append(dcc.Dropdown(
                id=f"clarification-{uuid.uuid4()}",
                options=[{'label': opt, 'value': opt} for opt in q['options']],
                placeholder="Select an option"
            ))
        return children
    return [html.P("Background information sufficient.")]

@app.callback(
    [Output('ai-trunc-len-f', 'options'),
     Output('ai-trunc-len-f', 'value'),
     Output('ai-trunc-len-r', 'options'),
     Output('ai-trunc-len-r', 'value'),
     Output('ai-max-ee', 'options'),
     Output('ai-max-ee', 'value'),
     Output('ai-prev-threshold', 'options'),
     Output('ai-prev-threshold', 'value'),
     Output('ai-treatment-group', 'options'),
     Output('ai-treatment-group', 'value'),
     Output('ai-top-asvs', 'options'),
     Output('ai-top-asvs', 'value'),
     Output('ai-pval-threshold', 'options'),
     Output('ai-pval-threshold', 'value'),
     Output('ai-contrib-threshold', 'options'),
     Output('ai-contrib-threshold', 'value')],
    [Input('upload-fastq-r1', 'contents'),
     Input('upload-fastq-r2', 'contents'),
     Input('upload-metadata', 'contents')]
)
def update_ai_suggested_params(r1_contents, r2_contents, metadata_content):
    if not ai_available or not r1_contents or not r2_contents or not metadata_content:
        return [[]]*16
    
    output_dir = './temp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fnFs = []
    fnRs = []
    sample_names = []
    for content, filename in zip(r1_contents, r1_contents):
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(content.split(',')[1]))
        fnFs.append(path)
        sample_names.append(filename.split('_')[0])
    for content, filename in zip(r2_contents, r2_contents):
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(content.split(',')[1]))
        fnRs.append(path)
    
    metadata_content = base64.b64decode(metadata_content.split(',')[1]).decode('utf-8')
    metadata = pd.read_csv(StringIO(metadata_content), index_col=0)
    
    quality_params = ai_analyze_quality_profiles(fnFs, fnRs)
    trunc_len_f_options = [{'label': str(x), 'value': x} for x in [200, 240, 280, 300]] if not quality_params else [{'label': str(quality_params['trunc_len_f']), 'value': quality_params['trunc_len_f']}]
    trunc_len_r_options = [{'label': str(x), 'value': x} for x in [180, 200, 220, 240]] if not quality_params else [{'label': str(quality_params['trunc_len_r']), 'value': quality_params['trunc_len_r']}]
    max_ee_options = [{'label': str(x), 'value': str(x)} for x in [[2,2], [3,3], [2,5], [5,2]]] if not quality_params else [{'label': str(quality_params['max_ee']), 'value': str(quality_params['max_ee'])}]
    trunc_len_f_value = quality_params['trunc_len_f'] if quality_params else 280
    trunc_len_r_value = quality_params['trunc_len_r'] if quality_params else 220
    max_ee_value = str(quality_params['max_ee']) if quality_params else '2,2'
    
    treatment = ai_analyze_metadata(metadata)
    treatment_options = [{'label': col, 'value': col} for col in metadata.columns] if not treatment else [{'label': treatment, 'value': treatment}]
    treatment_value = treatment if treatment else metadata.columns[0]
    
    prev_options = [{'label': str(x), 'value': x} for x in [0, 1, 5, 10]]
    top_asvs_options = [{'label': str(x), 'value': str(x)} for x in ['20,50,100', '10,20,50', '50,100,200']]
    pval_options = [{'label': str(x), 'value': x} for x in [0.001, 0.005, 0.01, 0.05]]
    contrib_options = [{'label': str(x), 'value': x} for x in [0.5, 0.65, 0.8, 0.9]]
    
    return (
        trunc_len_f_options, trunc_len_f_value,
        trunc_len_r_options, trunc_len_r_value,
        max_ee_options, max_ee_value,
        prev_options, 0,
        treatment_options, treatment_value,
        top_asvs_options, '20,50,100',
        pval_options, 0.005,
        contrib_options, 0.65
    )

@app.callback(
    [Output('seq-depth-plot', 'figure'),
     Output('alpha-diversity-plot', 'figure'),
     Output('beta-diversity-plot', 'figure'),
     Output('pca-plot', 'figure'),
     Output('pcoa-plot', 'figure'),
     Output('abundance-plot', 'figure'),
     Output('output-files', 'children'),
     Output('phylogenetic-tree', 'children'),
     Output('ai-interpretations', 'children')],
    [Input('run-analysis', 'n_clicks')],
    [State('upload-fastq-r1', 'contents'),
     State('upload-fastq-r1', 'filename'),
     State('upload-fastq-r2', 'contents'),
     State('upload-fastq-r2', 'filename'),
     State('upload-metadata', 'contents'),
     State('upload-silva', 'contents'),
     State('output-dir', 'value'),
     State('param-mode', 'value'),
     State('trunc-len-f', 'value'),
     State('trunc-len-r', 'value'),
     State('max-ee', 'value'),
     State('prev-threshold', 'value'),
     State('treatment-group', 'value'),
     State('top-asvs', 'value'),
     State('pval-threshold', 'value'),
     State('contrib-threshold', 'value'),
     State('ai-trunc-len-f', 'value'),
     State('ai-trunc-len-r', 'value'),
     State('ai-max-ee', 'value'),
     State('ai-prev-threshold', 'value'),
     State('ai-treatment-group', 'value'),
     State('ai-top-asvs', 'value'),
     State('ai-pval-threshold', 'value'),
     State('ai-contrib-threshold', 'value'),
     State('background-info', 'value')]
)
def run_analysis(n_clicks, r1_contents, r1_filenames, r2_contents, r2_filenames, metadata_content, silva_content, output_dir, param_mode, trunc_len_f, trunc_len_r, max_ee, prev_threshold, treatment, top_asvs, pval_threshold, contrib_threshold, ai_trunc_len_f, ai_trunc_len_r, ai_max_ee, ai_prev_threshold, ai_treatment, ai_top_asvs, ai_pval_threshold, ai_contrib_threshold, background):
    if n_clicks == 0:
        return [px.histogram(), px.violin(), px.scatter(), px.scatter(), px.scatter(), px.bar(), "No analysis run yet.", html.Img(), "No interpretations yet."]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fnFs = []
    fnRs = []
    sample_names = []
    for content, filename in zip(r1_contents, r1_filenames):
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(content.split(',')[1]))
        fnFs.append(path)
        sample_names.append(filename.split('_')[0])
    for content, filename in zip(r2_contents, r2_filenames):
        path = os.path.join(output_dir, filename)
        with open(path, 'wb') as f:
            f.write(base64.b64decode(content.split(',')[1]))
        fnRs.append(path)
    
    unzip_files([f for f in fnFs + fnRs if f.endswith('.zip')], output_dir)
    fnFs = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if '_R1' in f and (f.endswith('.fastq') or f.endswith('.fastq.gz'))]
    fnRs = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if '_R2' in f and (f.endswith('.fastq') or f.endswith('.fastq.gz'))]
    
    metadata_content = base64.b64decode(metadata_content.split(',')[1]).decode('utf-8')
    metadata = pd.read_csv(StringIO(metadata_content), index_col=0)
    
    if param_mode == 'ai_automatic':
        quality_params = ai_analyze_quality_profiles(fnFs, fnRs)
        trunc_len_f = quality_params['trunc_len_f'] if quality_params else 280
        trunc_len_r = quality_params['trunc_len_r'] if quality_params else 220
        max_ee = quality_params['max_ee'] if quality_params else [2, 2]
        treatment = ai_analyze_metadata(metadata) or metadata.columns[0]
    elif param_mode == 'ai_suggested':
        trunc_len_f = ai_trunc_len_f or trunc_len_f
        trunc_len_r = ai_trunc_len_r or trunc_len_r
        max_ee = [float(x) for x in ai_max_ee.split(',')] if ai_max_ee else [float(x) for x in max_ee.split(',')]
        prev_threshold = ai_prev_threshold or prev_threshold
        treatment = ai_treatment or treatment
        top_asvs = ai_top_asvs or top_asvs
        pval_threshold = ai_pval_threshold or pval_threshold
        contrib_threshold = ai_contrib_threshold or contrib_threshold
    else:
        max_ee = [float(x) for x in max_ee.split(',')]
    
    filtFs, filtRs = filter_and_trim(fnFs, fnRs, sample_names, output_dir, trunc_len_f, trunc_len_r, max_ee)
    seqtab, taxa = process_sequences(filtFs, filtRs, sample_names, None, output_dir)
    
    if param_mode == 'ai_automatic':
        prev_threshold = ai_analyze_prevalence(seqtab) or 0
    
    ps1 = create_phyloseq_object(seqtab, taxa, metadata)
    prev = seqtab.sum(axis=0)
    keep_taxa = prev[prev >= prev_threshold].index
    ps1['otu'] = ps1['otu'][keep_taxa]
    ps1['tax'] = ps1['tax'].loc[keep_taxa]
    
    ps1_meta = calculate_alpha_diversity(ps1, treatment)
    otu_rel, meta_rel = calculate_beta_diversity(ps1)
    top_asvs_list = [int(x) for x in top_asvs.split(',')]
    pca_result, explained_variance = perform_pca(ps1['otu'], top_asvs_list[-1])
    global_data['pca_result'] = pca_result
    global_data['explained_variance'] = explained_variance
    
    if param_mode == 'ai_automatic':
        pca_pcoa_params = ai_analyze_pca_pcoa(ps1['otu'], explained_variance)
        if pca_pcoa_params:
            top_asvs = pca_pcoa_params['top_asvs']
            pval_threshold = pca_pcoa_params['pval_threshold']
            contrib_threshold = pca_pcoa_params['contrib_threshold']
        top_asvs_list = [int(x) for x in top_asvs.split(',')]
        pca_result, explained_variance = perform_pca(ps1['otu'], top_asvs_list[-1])
        global_data['pca_result'] = pca_result
        global_data['explained_variance'] = explained_variance
    
    pcoa_scores, dm = perform_pcoa(otu_rel, meta_rel, treatment, pval_threshold, contrib_threshold)
    ps1_melt = otu_rel.reset_index().melt(id_vars='index', var_name='ASV', value_name='Abundance')
    ps1_melt = ps1_melt.merge(meta_rel[[treatment]], left_on='index', right_index=True)
    ps1_melt['Abundance'] *= 100
    tree_img = plot_phylogenetic_tree(seqtab)
    
    interpretations = ai_interpret_results(seqtab, ps1_meta, pcoa_scores, (pca_result, explained_variance), ps1_melt, tree_img, background)
    global_data['ai_interpretations'] = interpretations
    
    seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth")
    alpha_fig = px.violin(ps1_meta, x=treatment, y='Shannon', box=True, title='Shannon Diversity')
    beta_fig = px.scatter(pcoa_scores, x=0, y=1, color=treatment, title='PCoA Plot')
    pca_fig = px.scatter(x=pca_result[:,0], y=pca_result[:,1], title='PCA Plot')
    abundance_fig = px.bar(ps1_melt, x='index', y='Abundance', color='ASV', facet_col=treatment, title='Abundance by Phylum')
    
    output_files = [
        html.P(f"ASV Table: {os.path.join(output_dir, 'microbiomeAnalyst_16s_otu.txt')}"),
        html.P(f"Taxonomy Table: {os.path.join(output_dir, 'microbiomeAnalyst_16s_taxa.txt')}")
    ]
    
    return [seq_depth_fig, alpha_fig, beta_fig, pca_fig, beta_fig, abundance_fig, output_files, html.Img(src=tree_img), dcc.Markdown(interpretations)]

@app.callback(
    Output('download-report', 'data'),
    [Input('download-report-button', 'n_clicks')],
    [State('output-dir', 'value')]
)
def download_report(n_clicks, output_dir):
    if n_clicks == 0:
        return None
    report_path = generate_pdf_report(
        global_data['seqtab_nochim'],
        global_data['ps1.meta'],
        global_data['pseq_rel']['meta'].join(global_data['otu_rel']),
        (global_data['pca_result'], global_data['explained_variance']),
        global_data['otu_rel'].reset_index().melt(id_vars='index', var_name='ASV', value_name='Abundance'),
        plot_phylogenetic_tree(global_data['seqtab_nochim']),
        global_data['ai_interpretations'],
        output_dir
    )
    return dcc.send_file(report_path)

# Run app
# Note: If you see a TqdmWarning about IProgress, update Jupyter and ipywidgets:
# conda update jupyter ipywidgets
# or pip install --upgrade jupyter ipywidgets
if __name__ == '__main__':
    app.run(debug=True)
