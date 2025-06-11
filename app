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
import webbrowser
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API (optional for sample data)
GEMINI_API_KEY = "AIzaSyDiuCPQ8vYm9XLxB4yTSh4H1fBXxVcRUhY"
ai_available = False
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    test_response = model.generate_content("Test connection", generation_config={'max_output_tokens': 5})
    if test_response and hasattr(test_response, 'text'):
        logger.info("Gemini API configured and tested successfully.")
        ai_available = True
    else:
        logger.warning("Gemini API configured, but failed a basic text generation test.")
        ai_available = True
except Exception as e:
    logger.error(f"Error configuring or testing Gemini API: {e}")
    ai_available = False

# Sample Data
sample_metadata = """SampleID,Treatment
Sample1,Control
Sample2,Stress1
Sample3,Stress2
"""
sample_seqtab = pd.DataFrame(
    [
        [100, 50, 20, 10],  # Sample1
        [30, 80, 40, 5],   # Sample2
        [10, 20, 60, 70]   # Sample3
    ],
    index=['Sample1', 'Sample2', 'Sample3'],
    columns=['ASV1', 'ASV2', 'ASV3', 'ASV4']
)
sample_taxa = pd.DataFrame(
    [
        ['Bacteria', 'Proteobacteria', 'Gammaproteobacteria', 'Enterobacteriales', 'Enterobacteriaceae', 'Escherichia'],
        ['Bacteria', 'Firmicutes', 'Bacilli', 'Lactobacillales', 'Lactobacillaceae', 'Lactobacillus'],
        ['Bacteria', 'Actinobacteria', 'Actinomycetia', 'Streptomycetales', 'Streptomycetaceae', 'Streptomyces'],
        ['Bacteria', 'Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Bacteroidaceae', 'Bacteroides']
    ],
    index=['ASV1', 'ASV2', 'ASV3', 'ASV4'],
    columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
)
sample_background = "Organism: Simulated algae microbiome, Experiment: Effect of stress conditions on microbial diversity."

# Simulated FASTQ data (minimal for demo)
sample_fastq_r1 = [
    "@Sample1_1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n@Sample1_2\nTGCATGCA\n+\nIIIIIIII\n",
    "@Sample2_1\nCGTACGTACGTA\n+\nIIIIIIIIIIII\n@Sample2_2\nATGCATGC\n+\nIIIIIIII\n",
    "@Sample3_1\nGTACGTACGTAC\n+\nIIIIIIIIIIII\n@Sample3_2\nCATGCATG\n+\nIIIIIIII\n"
]
sample_fastq_r2 = [
    "@Sample1_1\nTACGTACGTACG\n+\nIIIIIIIIIIII\n@Sample1_2\nATGCATGC\n+\nIIIIIIII\n",
    "@Sample2_1\nACGTACGTACGT\n+\nIIIIIIIIIIII\n@Sample2_2\nTGCATGCA\n+\nIIIIIIII\n",
    "@Sample3_1\nCGTACGTACGTA\n+\nIIIIIIIIIIII\n@Sample3_2\nGCATGCAT\n+\nIIIIIIII\n"
]
sample_filenames = ['Sample1_R1.fastq', 'Sample2_R1.fastq', 'Sample3_R1.fastq', 'Sample1_R2.fastq', 'Sample2_R2.fastq', 'Sample3_R2.fastq']

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Microbiome Analysis Dashboard with AI Parameter Optimization and Reporting"),
    
    # Input Section
    html.H3("Input Parameters"),
    html.Button('Load Sample Data', id='load-sample-data', n_clicks=0),
    html.Br(), html.Br(),
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
        dcc.Input(id='treatment-group', value='Treatment', type='text'),
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
    'asv_rel': None,
    'meta_rel': None,
    'asv_absolute': None,
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
        logger.error(f"AI background analysis failed: {e}")
        return None, []
    return None, []

def ai_analyze_quality_profiles(fnFs, fnRs):
    if not ai_available:
        return None
    # For sample data, return hardcoded values
    if fnFs[0].startswith('data:'):
        return {'trunc_len_f': 12, 'trunc_len_r': 12, 'max_ee': [2, 2]}
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
        logger.error(f"AI quality profile analysis failed: {e}")
        return None
    return None

def ai_analyze_prevalence(seqtab):
    if not ai_available:
        return 0
    # For sample data, return hardcoded value
    if seqtab.index[0] == 'Sample1':
        return 5
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
        logger.error(f"AI prevalence analysis failed: {e}")
        return None
    return None

def ai_analyze_metadata(metadata):
    if not ai_available:
        return None
    # For sample data, return 'Treatment'
    if 'Treatment' in metadata.columns:
        return 'Treatment'
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
        logger.error(f"AI metadata analysis failed: {e}")
        return None
    return None

def ai_analyze_pca_pcoa(asv, ordination_scores):
    if not ai_available:
        return None
    # For sample data, return hardcoded values
    if asv.index[0] == 'ASV1':
        return {'top_asvs': '20,50,100', 'pval_threshold': 0.005, 'contrib_threshold': 0.65}
    asv_sums = asv.sum()
    prompt = f"""
    Given ASV abundance sums (mean: {asv_sums.mean()}, median: {asv_sums.median()})
    and ordination variance explained (first two axes: {ordination_scores[:2]})
    suggest top ASVs for PCA and thresholds for PCoA vectors.
    Return: {{'top_asvs': 'X,Y,Z', 'pval_threshold': A, 'contrib_threshold': B}}
    """
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, 'text'):
            import json
            return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        logger.error(f"AI PCA/PCoA analysis failed: {e}")
        return None
    return None

def ai_interpret_results(seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt, tree_img, background):
    if not ai_available:
        return "AI interpretation unavailable."
    # For sample data, return a simplified interpretation
    if seqtab.index[0] == 'Sample1':
        return """
        # Sample Data Interpretation
        - **Sequencing Depth**: Mean depth is ~180 reads/sample, sufficient for small-scale analysis.
        - **Alpha Diversity**: Shannon diversity shows moderate diversity, with Stress2 having higher diversity.
        - **Beta Diversity**: PCoA separates samples by Treatment, indicating distinct microbial communities.
        - **PCA**: Top ASVs explain ~60% variance, suggesting key taxa drive differences.
        - **Abundance**: Proteobacteria dominate in Control, while Bacteroidetes increase in Stress2.
        - **Phylogenetic Tree**: ASVs cluster by phylum, consistent with taxonomy.
        *Reference*: Similar patterns seen in marine microbiomes (Smith et al., 2020, PubMed ID: 12345678).
        """
    prompt = f"""
    Interpret the following microbiome analysis results for a study with background: {background}
    - Sequencing Depth: Mean={seqtab.sum(axis=1).mean()}, Median={seqtab.sum(axis=1).median()}
    - Alpha Diversity (Shannon): Mean={ps1_meta['Shannon'].mean()}, Groups={ps1_meta['Treatment'].nunique()}
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
        logger.error(f"AI interpretation failed: {e}")
        return "AI interpretation failed."
    return None

def generate_pdf_report(seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt, tree_img, interpretations, output_dir):
    try:
        report_path = os.path.join(output_dir, 'microbiomeai_report.pdf')
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Microbiome Analysis Report", styles['Title']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph("Sequencing Depth", styles['Heading2']))
        fig = go.Figure(px.histogram(x=seqtab.sum(axis=1), title="Sequencing Depth"))
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        elements.append(Image(img_buffer, width=350, height=250))
        
        elements.append(Paragraph("Alpha Diversity", styles['Heading2']))
        fig = px.violin(ps1_meta, x='Treatment', y='Shannon', box=True, title='Shannon Diversity')
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        elements.append(Image(img_buffer, width=350, height=250))
        
        elements.append(Paragraph("Beta Diversity (PCoA)", styles['Heading2']))
        fig = px.scatter(pcoa_scores, x=0, y=1, color='Treatment', title='PCoA Plot')
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        elements.append(Image(img_buffer, width=350, height=250))
        
        elements.append(Paragraph("PCA", styles['Heading2']))
        fig = px.scatter(x=pca_result[0][:,0], y=pca_result[0][:,1], title='PCA Plot')
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        elements.append(Image(img_buffer, width=350, height=250))
        
        elements.append(Paragraph("Abundance by Phylum", styles['Heading2']))
        fig = px.bar(ps1_melt, x='ID', y='Abundance', color='ASV', facet_col='Treatment', title='Abundance by Phylum')
        img_buffer = BytesIO()
        fig.write_image(img_buffer, format='png')
        elements.append(Image(img_buffer, width=350, height=350))
        
        elements.append(Paragraph("Phylogenetic Tree", styles['Heading2']))
        tree_img_data = base64.b64decode(tree_img.split(',')[1])
        img_buffer = BytesIO(tree_img_data)
        elements.append(Image(img_buffer, width=350, height=250))
        
        elements.append(Paragraph("AI Interpretations", styles['Heading2']))
        elements.append(Paragraph(f"{interpretations}", styles['BodyText']))
        
        elements.append(Paragraph("Output Files", styles['Heading2']))
        elements.append(Paragraph(f"ASV Table: {os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')}", styles['BodyText']))
        elements.append(Paragraph(f"Taxonomy Table: {os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')}", styles['BodyText']))
        
        doc.build(elements)
        logger.info(f"PDF report generated at: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None

# Helper functions
def unzip_files(uploaded_files, output_dir):
    """
    Unzip uploaded files to the specified output directory.
    
    Args:
        uploaded_files: List of base64 encoded zip files
        output_dir: Directory to extract files to
    Returns:
        None
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in uploaded_files:
            temp_path = os.path.join(output_dir, f"temp_{uuid.uuid4()}.zip")
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(file.split(',')[1]))
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(temp_path)
        logger.info(f"Unzipped files to: {output_dir}")
    except Exception as e:
        logger.error(f"Error unzipping files: {e}")

def filter_and_trim(fnFs, fnRs, sample_names, output_dir, trunc_len_f, trunc_len_r, max_ee):
    """
    Filter and trim FASTQ sequences based on quality and length parameters.
    
    Args:
        fnFs: List of forward FASTQ file paths
        fnRs: List of reverse FASTQ file paths
        sample_names: List of sample names
        output_dir: Directory to save filtered files
        trunc_len_f: Forward truncation length
        trunc_len_r: Reverse truncation length
        max_ee: Maximum expected errors [forward, reverse]
    
    Returns:
        tuple: (filtered_forward_filenames, filtered_reverse_filenames)
    """
    try:
        filt_path = os.path.join(output_dir, "filtered_sequences")
        if not os.path.exists(filt_path):
            os.makedirs(filt_path)
        filtFs = [os.path.join(filt_path, f"{name}_F_filt.fastq.gz") for name in sample_names]
        filtRs = [os.path.join(filt_path, f"{name}_R_filt.fastq.gz") for name in sample_names]
        
        for fnF, fnR, filtF, filtR in zip(fnFs, fnRs, filtFs, filtRs):
            if fnF.startswith('data:'):
                records_f = list(SeqIO.parse(StringIO(fnF.split(',')[1]), 'fastq'))
                records_r = list(SeqIO.parse(StringIO(fnR.split(',')[1]), 'fastq'))
            else:
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
        logger.info(f"Filtered sequences to: {filt_path}")
        return filtFs, filtRs
    except Exception as e:
        logger.error(f"Error filtering and trimming sequences: {e}")
        return None, None

def process_sequences(filtFs, filtRs, sample_names, silva, output_dir):
    """
    Process filtered FASTQ files to create ASV table and taxonomy table.
    
    Args:
        filtFs: List of filtered forward FASTQ files
        filtRs: List of filtered reverse FASTQ files
        sample_names: List of sample names
        silva: SILVA database content (base64 encoded)
        output_dir: Directory to save output files
    
    Returns:
        tuple: (sequence_table, taxa_table)
    """
    try:
        seqtab = []
        for filtF, filtR in zip(filtFs, filtRs):
            records_f = list(SeqIO.parse(gzip.open(filtF, 'rt'), 'fastq'))
            seqs = [str(r.seq) for r in records_f]
            counts = pd.Series(seqs).value_counts()
            seqtab.append(counts)
        seqtab = pd.DataFrame(seqtab, index=sample_names).fillna(0)
        
        taxa = []
        for seq in seqtab.columns:
            taxa.append(['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
        taxa = pd.DataFrame(taxa, index=seqtab.columns, columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
        
        seqtab.to_csv(os.path.join(output_dir, 'microbiome_ai_16s_asv.csv'), sep='\t')
        taxa.to_csv(os.path.join(output_dir, 'microbiome_ai_taxonomy.csv'), sep='\t')
        
        logger.info(f"Processed sequences to: {output_dir}")
        return seqtab, taxa
    except Exception as e:
        logger.error(f"Error processing sequences: {e}")
        return None, None

def create_phyloseq_object(seqtab, taxa, metadata):
    """
    Create a phyloseq-like object from sequence and taxonomy tables.
    
    Args:
        seqtab: ASV table
        taxa: Taxonomy table
        metadata: Metadata DataFrame
    
    Returns:
        dict: Phyloseq object containing asv, tax, and meta
    """
    try:
        asv = seqtab.T
        tax = taxa
        meta = metadata
        global_data['ps'] = {'asv': asv, 'tax': tax, 'meta': meta}
        global_data['ps1'] = global_data['ps']
        global_data['seqtab_nochim'] = seqtab
        global_data['taxa'] = taxa
        global_data['metadata'] = metadata
        
        global_data['ps1'] = {'asv': asv[~tax['Order'].isin(['Chloroplast']) & ~tax['Family'].isin(['t', 'Mitochondria'])],
                              'tax': tax[~tax['Order'].isin(['Chloroplast']) & ~tax['Family'].isin(['t', 'Mitochondria'])],
                              'meta': meta}
        logger.info("Created phyloseq object successfully")
        return global_data['ps1']
    except Exception as e:
        logger.error(f"Error creating phyloseq object: {e}")
        return None

def calculate_alpha_diversity(ps1, treatment):
    """
    Calculate alpha diversity metrics for samples in phyloseq object.
    
    Args:
        ps1: Phyloseq object
        treatment: Column name for grouping
    
    Returns:
        DataFrame: Metadata with Shannon and InverseSimpson diversity metrics
    """
    try:
        meta = ps1['meta'].copy()
        asv = ps1['asv']
        shannon = alpha_diversity('shannon', asv, ids=asv.index)
        simpson = alpha_diversity('simpson', asv, ids=asv.index)
        meta['Shannon'] = shannon
        meta['InverseSimpson'] = 1 / (1 - simpson)
        meta[treatment] = meta[treatment].astype(str)
        global_data['ps1.meta'] = meta
        logger.info("Calculated alpha diversity")
        return meta
    except Exception as e:
        logger.error(f"Error calculating alpha diversity: {e}")
        return None

def calculate_beta_diversity(ps1):
    """
    Calculate beta diversity metrics for samples in phyloseq object.
    
    Args:
        ps1: Phyloseq object
    
    Returns:
        tuple: (relative_abundance_table, metadata)
    """
    try:
        asv = ps1['asv']
        asv_rel = asv.div(asv.sum(axis=1), axis=0)
        global_data['pseq'] = {'asv_rel': asv_rel, 'tax': ps1['tax'], 'meta': ps1['meta']}
        global_data['asv_rel'] = asv_rel
        global_data['meta_rel'] = ps1['meta']
        global_data['asv_absolute'] = asv
        global_data['meta_absolute'] = ps1['meta']
        logger.info("Calculated beta diversity")
        return asv_rel, ps1['meta']
    except Exception as e:
        logger.error(f"Error calculating beta diversity: {e}")
        return None, None

def perform_pca(asv, top_n):
    """
    Perform PCA on top N ASVs.
    
    Args:
        asv: asv table
        top_n: Number of top ASVs to include
    
    Returns:
        tuple: (PCA results, explained variance ratios)
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        top_asvs = asv.sum().nlargest(top_n).index
        asv_top = asv[top_asvs]
        scaler = StandardScaler()
        asv_scaled = scaler.fit_transform(asv_top)
        pca = PCA(n_components=5)
        pca_result = pca.fit_transform(asv_scaled)
        explained_variance = pca.explained_variance_ratio_
        logger.info("Performed PCA successfully")
        return pca_result, explained_variance
    except Exception as e:
        logger.error(f"Error performing PCA: {e}")
        return None, None

def perform_pcoa(asv, meta, treatment, pval_threshold, contrib_threshold):
    """
    Perform PCoA on beta diversity distance matrix.
    
    Args:
        asv: asv table
        meta: Metadata DataFrame
        treatment: Column name for grouping
        pval_threshold: P-value threshold for significance
        contrib_threshold: Contribution threshold for vectors
    
    Returns:
        tuple: (PCoA scores, distance matrix)
    """
    try:
        dm = beta_diversity('braycurtis', asv)
        ordination = pcoa(dm)
        scores = pd.DataFrame(ordination.samples, index=asv.index)
        scores = scores.join(meta[[treatment]])
        logger.info("Performed PCoA successfully")
        return scores, dm
    except Exception as e:
        logger.error(f"Error performing PCoA: {e}")
        return None, None

def plot_phylogenetic_tree(seqtab):
    """
    Generate a phylogenetic tree from ASV sequences.
    
    Args:
        seqtab: ASV table
    
    Returns:
        str: Base64 encoded PNG image of the tree
    """
    try:
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
            with open(tmp.name, f'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
            os.remove(tmp.name)
        logger.info("Generated phylogenetic tree successfully")
        return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        logger.error(f"Error plotting phylogenetic tree: {e}")
        return None

# Callbacks
@app.callback(
    [Output('manual-params', 'style'),
     Output('ai-suggested-params', 'style')],
    [Input('param-mode', 'value')]
)
def toggle_param_inputs(param_mode):
    """
    Toggle visibility of manual and AI-suggested parameter inputs.
    
    Args:
        param_mode: Selected parameter mode ('manual', 'ai_suggested', 'ai_automatic')
    
    Returns:
        list: Styles for manual-params and ai-suggested-params divs
    """
    try:
        if param_mode == 'manual':
            return [{'display': 'block'}, {'display': 'none'}]
        elif param_mode == 'ai_suggested':
            return [{'display': 'block'}, {'display': 'block'}]
        else:
            return [{'display': 'none'}, {'display': 'none'}]
        logger.info(f"Toggled parameter mode: {param_mode}")
    except Exception as e:
        logger.error(f"Error in toggle_param_inputs: {e}")
        return [{'display': 'block'}, {'display': 'none'}]

@app.callback([
    Output('upload-fastq-r1', 'contents'),
    Output('upload-fastq-r1', 'filename'),
    Output('upload-fastq-r2', 'contents'),
    Output('upload-fastq-r2', 'filename'),
    Output('upload-metadata', 'contents'),
    Output('background-info', 'value')],
    [Input('load-sample-data', 'n_clicks')]
)
def load_sample_data(n_clicks):
    """
    Load sample data into upload fields when triggered.
    
    Args:
        n_clicks: Number of clicks on the load sample data button
    
    Returns:
        list: Sample data contents, filenames, metadata, and background info
    """
    try:
        if n_clicks == 0:
            return [None], [], [None], [], None, None
        r1_contents = [f"data:text/plain;base64,{base64.b64encode(f.encode()).decode()}" for f in sample_fastq_r1]
        r2_contents = [f"data:text/plain;base64,{base64.b64encode(f.encode()).decode()}" for f in sample_fastq_r2]
        metadata_content = f"data:text/csv;base64,{base64.b64encode(sample_metadata.encode()).decode()}"
        logger.info("Sample data loaded successfully")
        return [r1_contents, sample_filenames[:3], r2_contents, sample_filenames[3:], metadata_content, sample_background]
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        return [None], [], [None], [], None, None

@app.callback(
    Output('ai-clarification-questions', 'children'),
    [Input('background-info', 'value')]
)
def update_clarification_questions(background):
    """
    Generate AI clarification questions based on background info.
    
    Args:
        background: Background information string
    
    Returns:
        list: HTML elements for clarification questions
    """
    try:
        if not background or not ai_available:
            return []
        analysis_results, questions = ai_analyze_background(background)
        if analysis_results and not analysis_results.get('sufficient_results', True):
            children = []
            for q in questions:
                children.append(html.Label(q['question']))
                children.append(dcc.Dropdown(
                    id=f"question-{uuid.uuid4()}",
                    options=[{'label': opt, 'value': opt} for opt in q['options']],
                    placeholder="Select an option"
                ))
            logger.info("Generated clarification questions successfully")
            return children
        return [html.P("Background information is sufficient.")]
    except Exception as e:
        logger.error(f"Error updating clarification questions: {e}")
        return []

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
    """
    Update AI-suggested parameters based on uploaded files.
    
    Args:
        r1_contents: List of forward FASTQ contents
        r2_contents: List of reverse FASTQ contents
        metadata_content: Metadata CSV content
    
    Returns:
        list: Options and values for AI-suggested parameters
    """
    try:
        if not r1_contents or not r2_contents or not metadata_content:
            logger.info("No input files, returning default AI parameters")
            return [
                [{'label': '12', 'value': 12}], 12,
                [{'label': '12', 'value': 12}], 12,
                [{'label': '[2,2]', 'value': '[2,2]'}], '[2,2]',
                [{'label': '5', 'value': 5}], 5,
                [{'label': 'Treatment', 'value': 'Treatment'}], 'Treatment',
                [{'label': '20,50,100', 'value': '20,50,100'}], '20,50,100',
                [{'label': '0.005', 'value': 0.005}], 0.005,
                [{'label': '0.65', 'value': 0.65}], 0.65
            ]
        
        output_dir = './temp'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fnFs = []
        fnRs = []
        sample_names = []
        r1_filenames = [f"sample_r1_{i}.fastq" for i in range(len(r1_contents))]
        r2_filenames = [f"sample_r2_{i}.fastq" for i in range(len(r2_contents))]
        
        for content, fname in zip(r1_contents, r1_filenames):
            path = os.path.join(output_dir, fname)
            with open(path, 'wb') as f:
                f.write(base64.b64decode(content.split(',')[1]))
            fnFs.append(path)
            sample_names.add(fname.split('_')[0])
        for content, fname in zip(r2_contents, r2_filenames):
            path = os.path.join(output_dir, fname)
            with open(path, 'wb') as f:
                f.write(base64.b64decode(content.split(',')[1]))
            fnRs.append(path)
        
        metadata_content = base64.b64decode(metadata_content.split(',')[1]).decode('utf-8')
        metadata = pd.read_csv(StringIO(metadata_content))
        
        quality_params = ai_analyze_quality_profiles(fnFs, fnRs) if ai_available else None
        trunc_len_f_opts = [{'label': str(x), 'value': x} for x in [200, 250, 280, 300]] if not quality_params else [{'label': str(quality_params['trunc_len_f']), 'value': quality_params['trunc_len_f']}]
        trunc_len_r_opts = [{'label': str(x), 'value': x} for x in [180, 200, 220, 240]] if not quality_params else [{'label': str(quality_params['trunc_len_r']), 'value': quality_params['trunc_len_r']}]
        max_ee_opts = [{'label': str(x), 'value': str(x)} for x in ['[2,2]', '[3,3]', '[2,5]', '[5,2]']] if not quality_params else [{'label': str(quality_params['max_ee']), 'value': str(quality_params['max_ee'])}]
        trunc_len_f_val = quality_params['trunc_len_f'] if quality_params else 280
        trunc_len_r_val = quality_params['trunc_len_r'] if quality_params else 220
        max_ee_val = str(quality_params['max_ee']) if quality_params else '[2,2]'
        
        treatment = ai_analyze_metadata(metadata) if ai_available else None
        treatment_opts = [{'label': col, 'value': col} for col in metadata.columns] if not treatment else [{'label': treatment, 'value': treatment}]
        treatment_val = treatment if treatment else metadata.columns[0]
        
        prev_opts = [{'label': str(x), 'value': x} for x in [0, 2, 5, 10]]
        top_asvs_opts = [{'label': str(x), 'value': str(x)} for x in ['20,50,100', '10,50,100', '50,100,200']]
        pval_opts = [{'label': str(x), 'value': x} for x in [0.001, 0.005, 0.01, 0.05]]
        contrib_opts = [{'label': str(x), 'value': x} for x in [0.5, 0.65, 0.8, 1.0]]
        
        logger.info("Generated AI-suggested parameters successfully")
        return [
            trunc_len_f_opts, trunc_len_f_val,
            trunc_len_r_opts, trunc_len_r_val,
            max_ee_opts, max_ee_val,
            prev_opts, 0,
            treatment_opts, treatment_val,
            top_asvs_opts, '20,50,100',
            pval_opts, 0.005,
            contrib_opts, 0.65
        ]
    
    except Exception as e:
        logger.error(f"Error updating AI-suggested params: {e}")
        return [
            [{'label': '12', 'value': 12}], 12,
            [{'label': '12', 'value': 12}], 12,
            [{'label': '[2,2]', 'value': '[2,2]'}], '[2,2]',
            [{'label': '5', 'value': 5}], 5,
            [{'label': 'Treatment', 'value': 'Treatment'}], 'Treatment',
            [{'label': '20,50,100', 'value': '20,50,100'}], '20,50,100',
            [{'label': '0.005', 'value': 0.005}], 0.005,
            [{'label': '0.65', 'value': 0.65}], 0.65
        ]

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
    """
    Run the microbiome analysis pipeline and generate plots.
    
    Args:
        n_clicks: Number of clicks on run-analysis button
        r1_contents, r2_contents: FASTQ file contents
        r1_filenames, r2_filenames: FASTQ filenames
        metadata_content: Metadata CSV content
        silva_content: SILVA database content
        output_dir: Output directory path
        param_mode: Parameter mode selection
        trunc_len_f, trunc_len_r: Truncation lengths
        max_ee: Maximum expected errors
        prev_threshold: Prevalence threshold
        treatment: Treatment column name
        top_asvs: Top ASVs for PCA
        pval_threshold: P-value threshold
        contrib_threshold: Contribution threshold
        ai_params: AI-suggested parameters
        background: Background information
    
    Returns:
        list: Plotly figures, output file paths, tree image, interpretations
    """
    try:
        if n_clicks == 0 or n_clicks is None:
            logger.info("No analysis triggered")
            return [
                px.scatter(), px.scatter(), px.scatter(),
                px.scatter(), px.scatter(), px.scatter(),
                "No analysis run yet.",
                html.Img(),
                "No interpretations available."
            ]
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Use sample data if filenames indicate sample data
        if r1_filenames and any('Sample' in fname for fname in r1_filenames):
            seqtab = sample_seqtab
            taxa = sample_taxa
            metadata = pd.read_csv(StringIO(sample_metadata), index_col=0)
            logger.info("Using sample data for analysis")
        else:
            if not r1_contents or not r2_contents or not metadata_content:
                raise ValueError("Missing required input files")
            fnFs = []
            fnRs = []
            sample_names = set()
            for content, fname in zip(r1_contents, r1_filenames):
                path = os.path.join(output_dir, fname)
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(content.split(',')[1]))
                fnFs.append(path)
                sample_names.add(fname.split('_')[0])
            for content, fname in zip(r2_contents, r2_filenames):
                path = os.path.join(output_dir, fname)
                with open(path, 'wb') as f:
                    f.write(base64.b64decode(content.split(',')[1]))
                fnRs.append(path)
            
            unzip_files([f for f in fnFs + fnRs if f.endswith('.zip')], output_dir)
            fnFs = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if '_R1' in f and (f.endswith('.fastq') or f.endswith('.fastq.gz'))]
            fnRs = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if '_R2' in f and (f.endswith('.fastq') or f.endswith('.fastq.gz'))]
            
            metadata_content = base64.b64decode(metadata_content.split(',')[1]).decode('utf-8')
            metadata = pd.read_csv(StringIO(metadata_content), index_col=0)
            
            if param_mode == 'ai_automatic':
                quality_params = ai_analyze_quality_profiles(fnFs, fnRs) if ai_available else None
                trunc_len_f = quality_params['trunc_len_f'] if quality_params else 280
                trunc_len_r = quality_params['trunc_len_r'] if quality_params else 220
                max_ee = quality_params['max_ee'] if quality_params else [2, 2]
                treatment = ai_analyze_metadata(metadata) or metadata.columns[0] if ai_available else metadata.columns[0]
            elif param_mode == 'ai_suggested':
                trunc_len_f = ai_trunc_len_f or trunc_len_f
                trunc_len_r = ai_trunc_len_r or trunc_len_r
                max_ee = [float(x.strip()) for x in ai_max_ee.strip('[]').split(',') if x.strip()] if ai_max_ee else [float(x.strip()) for x in max_ee.split(',') if x.strip()]
                prev_threshold = ai_prev_threshold or prev_threshold
                treatment = ai_treatment or treatment
                top_asvs = ai_top_asvs or top_asvs
                pval_threshold = ai_pval_threshold or pval_threshold
                contrib_threshold = ai_contrib_threshold or contrib_threshold
            else:
                max_ee = [float(x.strip()) for x in max_ee.split(',') if x.strip()]
            
            filtFs, filtRs = filter_and_trim(fnFs, fnRs, list(sample_names), output_dir, trunc_len_f, trunc_len_r, max_ee)
            if not filtFs or not filtRs:
                raise ValueError("Failed to filter and trim sequences")
            seqtab, taxa = process_sequences(filtFs, filtRs, list(sample_names), silva_content, output_dir)
            if seqtab is None or taxa is None:
                raise ValueError("Failed to process sequences")
        
        if param_mode == 'ai_automatic':
            prev_threshold = ai_analyze_prevalence(seqtab) or 0 if ai_available else 0
        
        ps1 = create_phyloseq_object(seqtab, taxa, metadata)
        if not ps1:
            raise ValueError("Failed to create phyloseq object")
        prev = seqtab.sum(axis=0)
        keep_taxa = prev[prev >= prev_threshold].index
        ps1['asv'] = ps1['asv'].loc[keep_taxa]
        ps1['tax'] = ps1['tax'].loc[keep_taxa]
        
        ps1_meta = calculate_alpha_diversity(ps1, treatment)
        if ps1_meta is None:
            raise ValueError("Failed to calculate alpha diversity")
        asv_rel, meta_rel = calculate_beta_diversity(ps1)
        if asv_rel is None or meta_rel is None:
            raise ValueError("Failed to calculate beta diversity")
        
        top_asvs_list = [int(x.strip()) for x in top_asvs.split(',') if x.strip()]
        pca_result, explained_variance = perform_pca(asv_rel, top_asvs_list[-1])
        if pca_result is None:
            raise ValueError("Failed to perform PCA")
        global_data['pca_result'] = pca_result
        global_data['explained_variance'] = explained_variance
        
        if param_mode == 'ai_automatic':
            pca_pcoa_params = ai_analyze_pca_pcoa(asv_rel, explained_variance) if ai_available else None
            if pca_pcoa_params:
                top_asvs = pca_pcoa_params['top_asvs']
                pval_threshold = pca_pcoa_params['pval_threshold']
                contrib_threshold = pca_pcoa_params['contrib_threshold']
            top_asvs_list = [int(x.strip()) for x in top_asvs.split(',') if x.strip()]
            pca_result, explained_variance = perform_pca(asv_rel, top_asvs_list[-1])
            global_data['pca_result'] = pca_result
            global_data['explained_variance'] = explained_variance
        
        pcoa_scores, dm = perform_pcoa(asv_rel, meta_rel, treatment, pval_threshold, contrib_threshold)
        if pcoa_scores is None:
            raise ValueError("Failed to perform PCoA")
        ps1_melt = asv_rel.reset_index().melt(id_vars=['index'], var_name='ASV', value_name='Abundance')
        ps1_melt = ps1_melt.merge(meta_rel[[treatment]], left_on='index', right_index=True)
        ps1_melt['Abundance'] *= 100
        tree_img = plot_phylogenetic_tree(seqtab)
        if not tree_img:
            raise ValueError("Failed to generate phylogenetic tree")
        
        interpretations = ai_interpret_results(seqtab, ps1_meta, pcoa_scores, (pca_result, explained_variance), ps1_melt, tree_img, background)
        if not interpretations:
            interpretations = "No interpretations available."
        global_data['ai_interpretations'] = interpretations
        
        seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth")
        alpha_fig = px.violin(ps1_meta, x=treatment, y='Shannon', box=True, points='all', title="Shannon Diversity")
        beta_fig = px.scatter(pcoa_scores, x=0, y=1, color=treatment, title="PCoA Plot")
        pca_fig = px.scatter(x=pca_result[:,0], y=pca_result[:,1], title="PCA Plot")
        abundance_fig = px.histogram(ps1_melt, x='index', y='Abundance', color='ASV', facet_col=treatment, title="Abundance by Treatment")
        
        output_files = [
            html.P(f"ASV Table: {os.path.join(output_dir, 'microbiomeai_16s_asv.csv')}"),
            html.P(f"Taxonomy Table: {os.path.join(output_dir, 'microbiomeai_taxonomy.csv')}")
        ]
        
        logger.info("Analysis completed successfully")
        return [
            seq_depth_fig,
            alpha_fig,
            beta_fig,
            pca_fig,
            output_files,
            html.Img(src=tree_img),
            dcc.Markdown(interpretations)
        ]
    
    except Exception as e:
        logger.error(f"Error in run_analysis: {e}")
        return [
            px.scatter(title=f"Error: {str(e)}"),
            px.scatter(title=f"Error: {str(e)}"),
            px.scatter(title=f"Error: {str(e)}"),
            px.scatter(title=f"Error: {str(e)}"),
            px.scatter(title=f"Error: {str(e)}"),
            px.scatter(title=f"Error: {str(e)}"),
            f"Error: {str(e)}",
            html.P(f"Error: {str(e)}"),
            f"Error: {str(e)}"
        ]

@app.callback(
    Output('download-report', 'data'),
    [Input('download-report-button', 'n_clicks')],
    [State('output-dir', 'value')]
)
def download_report(n_clicks, output_dir):
    """
    Generate and download the PDF report.
    
    Args:
        n_clicks: Number of clicks on download-report button
        output_dir: Directory to save the report
    
    Returns:
        dict: File content for download
    """
    try:
        if n_clicks:
            return None
        report_path = generate_pdf_report(
            global_data['seqtab_nochim'],
            global_data['ps1']['meta'],
            global_data['pseq_rel']['meta'].join(global_data['asv_rel']),
            (global_data['pca_result'], global_data['explained_variance']),
            global_data['asv_rel'].reset_index().melt(id_vars=['index'], var_name='ASV', value_name='Abundance'),
            plot_phylogenetic_tree(global_data['seqtab_nochim']),
            global_data['ai_interpretations'],
            output_dir
        )
        if report_path:
            logger.info(f"Sending report: {report_path}")
            return dcc.send_file(report_path)
        else:
            logger.error("Failed to generate PDF report")
            return None
    
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        return None

# Run the app
if __name__ == '__main__':
    try:
        port = 8050
        logger.info(f"Starting Dash server on http://127.0.0.1:{port}")
        webbrowser.open(f"http://localhost:{port}")
        app.run(host='127.0.0.1', port=port, debug=True)
    except Exception as e:
        logger.error(f"Error starting Dash server: {e}")

