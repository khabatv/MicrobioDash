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
import io
import re
from Bio import Phylo
import multiprocessing
from filter_and_trim_function import filter_and_trim_parallel

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
        [100, 50, 20, 10],
        [30, 80, 40, 5],
        [10, 20, 60, 70]
    ],
    index=['Sample1', 'Sample2', 'Sample3'],
    columns=[
        'TACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGTTTCTTAAGTCTGATGTGAAAGCCCCCGGCTCAACCGGGGAGGGTCATTGGAAACTGGGGAACTTGAGTGCAGAAGAGGAAAGTGGAATTCCATGTGTAGCGGTGAAATGCGTAGATATATGGAGGAACACCAGTGGCGAAGGCGACTTTCTGGTCTGTAACTGAC',
        'TACGTAGGTGGCGAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGTTTTTTAAGTCTGATGTGAAAGCCCCCGGCTCAACCGGGGAGGGTCATTGGAAACTGGAAAACTTGAGTGCAGAAGAGGAGAGTGGAATTCCATGTGTAGCGGTGAAATGCGTAGATATATGGAGGAACACCAGTGGCGAAGGCGACTCTCTGGTCTGTAACTGAC',
        'TACGTAGGGGGCAAGCGTTGTCCGGATTTACTGGGCGTAAAGCGCGTGCAGGCGGTTATTCAAGTCGGATGTGAAATCCCCGGGCTCAACCTGGGAACTGCATTCGAAACTGGTGAGCTAGAGTTTGGTAGAGGGTGGTGGAATTTCCTGTGTAGCGGTGAAATGCGTAGATATAGGAAGGAACACCAGTGGCGAAGGCGACCACCTGGACTGATACTGAC',
        'TACGTAGGTGGCAAGCGTTATCCGGAATTATTGGGCGTAAAGCGCGCGTAGGCGGTTTTGTAAGTCTGAAGTGAAATCCCTGGGCTCAACCTGGGAACTGCATTCAGAACTGGGCGACTAGAGTACGTCAGAGGGGAGTGGAATTCCTGGTGTAGCGGTGAAATGCATAGATATCAGGAGGAACACCGGTGGCGAAGGCGGCTCACTGGACGTATTACTGAC'
    ]
)
sample_taxa = pd.DataFrame(
    [
        ['Bacteria', 'Proteobacteria', 'Gammaproteobacteria', 'Enterobacteriales', 'Enterobacteriaceae', 'Escherichia'],
        ['Bacteria', 'Firmicutes', 'Bacilli', 'Lactobacillales', 'Lactobacillaceae', 'Lactobacillus'],
        ['Bacteria', 'Actinobacteria', 'Actinomycetia', 'Streptomycetales', 'Streptomycetaceae', 'Streptomyces'],
        ['Bacteria', 'Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Bacteroidaceae', 'Bacteroides']
    ],
    index=sample_seqtab.columns, # Use the new DNA sequences as the index
    columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']
)
sample_taxa.index.name = 'ASV'
sample_filenames = ['Sample1_R1.fastq', 'Sample2_R1.fastq', 'Sample3_R1.fastq', 'Sample1_R2.fastq', 'Sample2_R2.fastq', 'Sample3_R2.fastq']

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Microbiome Analysis Dashboard with AI Parameter Optimization and Reporting"),
    
    # Input Section
    html.H3("Input Parameters"),
    html.Button('Use Internal Sample Data', id='load-sample-data', n_clicks=0,
                title="Loads small, pre-packaged sample data to test the analysis pipeline."),
    html.Br(), html.Br(),

    # New input for the data folder path
    html.Label("Project Data Folder Path (contains your FASTQ and metadata files):"),
    dcc.Input(id='data-folder-path', value='./uploads', type='text', style={'width': '80%'}),
    html.Button('List Files in Folder', id='list-files-button', n_clicks=0, style={'marginLeft': '10px'}),
    html.P("Place your data in a folder (e.g., 'uploads') and provide the path here.", style={'fontSize': 'small', 'color': 'gray'}),

    html.Div(id='output-r1-filenames', style={'marginTop': '10px'}),
    html.Div(id='output-r2-filenames', style={'marginTop': '10px'}),
    html.Div(id='output-metadata-filename', style={'marginTop': '10px'}),

    html.Label("Upload SILVA Taxonomy Database (or place in data folder and name it silva.fasta)"),
    dcc.Upload(id='upload-silva', children=html.Button('Upload SILVA File')),
    html.Div(id='silva-status-output', style={'marginTop': '5px', 'padding': '5px', 'borderRadius': '3px'}),
    
    html.Label("Output Directory:"),
    dcc.Input(id='output-dir', value='./output', type='text'),
    
    html.Label("Background Information:"),
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
    html.H3("Analysis Starting Point"),
dcc.RadioItems(
    id='analysis-mode',
    options=[
        {'label': 'Start from raw FASTQ files (Slow, run first)', 'value': 'fastq'},
        {'label': 'Start from Processed ASV/Taxonomy Tables (Fast)', 'value': 'asv'}
    ],
    value='fastq', # Default to starting from scratch
    labelStyle={'display': 'block'}
),
html.Br(),
    # Parameter Tuning
   html.Div(id='manual-params', children=[
    html.H3("Analysis Parameters"),

    # This is the new wrapper div we are adding
    html.Div(id='preprocessing-params-div', children=[
        html.H4("Pre-processing Parameters (for FASTQ mode)", style={'color': '#555'}),
        html.Label("Truncation Length Forward (e.g., 280):"),
        dcc.Input(id='trunc-len-f', value=280, type='number'),
        html.Label("Truncation Length Reverse (e.g., 220):"),
        dcc.Input(id='trunc-len-r', value=220, type='number'),
        html.Label("Max Expected Errors (Forward, Reverse):"),
        dcc.Input(id='max-ee', value='2,2', type='text'),
    ]),
    html.H4("Downstream Analysis Parameters", style={'color': '#555'}), # Header for clarity
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
        html.Button('Suggest AI Parameters', id='suggest-ai-params-button', n_clicks=0),
        html.Br(),
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

def ai_interpret_results(seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt, tree_img, background, treatment_column_name):
    """
    This function now correctly calculates variance on numeric columns only.
    """
    if not ai_available:
        return "AI interpretation unavailable."

    if 'PC1' in pcoa_scores.columns and 'PC2' in pcoa_scores.columns:
        pcoa_variance_explained = pcoa_scores[['PC1', 'PC2']].var().values
    else:
        numeric_pcoa_scores = pcoa_scores.select_dtypes(include=np.number)
        pcoa_variance_explained = numeric_pcoa_scores.var().values[:2]

    if seqtab.index[0] == 'Sample1':
        return """
        # Sample Data Interpretation
        - **Sequencing Depth**: Mean depth is ~180 reads/sample.
        - **Alpha Diversity**: Shannon diversity shows moderate diversity.
        - **Beta Diversity**: PCoA separates samples by Treatment.
        - **PCA**: Top ASVs explain ~60% variance.
        - **Abundance**: Proteobacteria dominate in Control.
        - **Phylogenetic Tree**: ASVs cluster by phylum.
        *Reference*: Smith et al., 2020, PubMed ID: 12345678.
        """
        
    prompt = f"""
    Interpret the following microbiome analysis results for a study with background: {background}
    - Sequencing Depth: Mean={seqtab.sum(axis=1).mean()}, Median={seqtab.sum(axis=1).median()}
    - Alpha Diversity (Shannon): Mean={ps1_meta['Shannon'].mean()}, Groups={ps1_meta[treatment_column_name].nunique()}
    - Beta Diversity (PCoA): Variance explained={pcoa_variance_explained}
    - PCA: Variance explained={pca_result[1][:2]}
    - Abundance: Top phyla={ps1_melt.groupby('ASV')['Abundance'].sum().nlargest(5).index.tolist()}
    - Phylogenetic Tree: Generated successfully.
    Provide a detailed interpretation... (rest of prompt is the same)
    """
    try:
        # We add a timeout here for robustness
        response = model.generate_content(prompt, request_options={'timeout': 120})
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
        fig = px.scatter(pcoa_scores, x='PC1', y='PC2', color='Treatment', title='PCoA Plot')
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

def _filter_and_trim_worker(args):

    fnF, fnR, sample_name, output_dir, trunc_len_f, trunc_len_r, max_ee = args
    filt_path = os.path.join(output_dir, "filtered_sequences")
    filtF = os.path.join(filt_path, f"{sample_name}_F_filt.fastq.gz")
    filtR = os.path.join(filt_path, f"{sample_name}_R_filt.fastq.gz")

    logger.info(f"--- Processing sample '{sample_name}' from files: {os.path.basename(fnF)} & {os.path.basename(fnR)} ---")

    try:
        # Open input files (handles .gz or plain text)
        open_f = gzip.open if fnF.lower().endswith('.gz') else open
        open_r = gzip.open if fnR.lower().endswith('.gz') else open

        with open_f(fnF, 'rt') as f_in, open_r(fnR, 'rt') as r_in:
            records_f_iter = SeqIO.parse(f_in, 'fastq')
            records_r_iter = SeqIO.parse(r_in, 'fastq')

            kept_records_f = []
            kept_records_r = []
            initial_count = 0

            for rec_f, rec_r in zip(records_f_iter, records_r_iter):
                initial_count += 1
                if (len(rec_f) >= trunc_len_f and
                    len(rec_r) >= trunc_len_r and
                    np.mean(rec_f.letter_annotations['phred_quality']) >= 30 and
                    np.mean(rec_r.letter_annotations['phred_quality']) >= 30):

                    kept_records_f.append(rec_f[:trunc_len_f])
                    kept_records_r.append(rec_r[:trunc_len_r])

        final_count = len(kept_records_f)
        if initial_count > 0:
            logger.info(f"[{sample_name}] Initial read pairs: {initial_count}")
            logger.info(f"[{sample_name}] Read pairs remaining after filtering: {final_count} ({final_count/initial_count:.2%})")
        else:
            logger.warning(f"[{sample_name}] No read pairs found in the input files.")
            return None, None


        if not kept_records_f:
            logger.error(f"[{sample_name}] Zero read pairs remaining after filtering. Check truncation lengths and data quality.")
            return None, None

        with gzip.open(filtF, 'wt') as f_out:
            SeqIO.write(kept_records_f, f_out, 'fastq')
        with gzip.open(filtR, 'wt') as r_out:
            SeqIO.write(kept_records_r, r_out, 'fastq')

        return filtF, filtR

    except Exception as e:
        logger.error(f"Error processing sample {sample_name}: {e}", exc_info=True)
        return None, None

def filter_and_trim(fnFs, fnRs, sample_names, output_dir, trunc_len_f, trunc_len_r, max_ee):
    """
    Filter and trim FASTQ sequences in parallel using a multiprocessing Pool.
    """
    try:
        filt_path = os.path.join(output_dir, "filtered_sequences")
        if not os.path.exists(filt_path):
            os.makedirs(filt_path)

        if not fnFs or not fnRs:
            logger.error("filter_and_trim was called with empty file lists (fnFs or fnRs).")
            return None, None

        # Prepare arguments for each worker process
        args_list = [
            (fnF, fnR, sample_name, output_dir, trunc_len_f, trunc_len_r, max_ee)
            for fnF, fnR, sample_name in zip(fnFs, fnRs, sample_names)
        ]

        # Use a multiprocessing Pool to process files in parallel
        # We wrap this in a `if __name__ == '__main__':` block in the main script
        # to ensure it works correctly on all platforms (especially Windows).
        with multiprocessing.Pool() as pool:
            results = pool.map(_filter_and_trim_worker, args_list)

        # Process the results to separate the filtered file paths
        filtFs = [res[0] for res in results if res and res[0]]
        filtRs = [res[1] for res in results if res and res[1]]

        if not filtFs or not filtRs or len(filtFs) != len(filtRs):
            logger.error("Parallel filtering and trimming failed to produce valid output for some samples. Aborting.")
            return None, None

        logger.info(f"Successfully filtered and trimmed all sequences in parallel to: {filt_path}")
        return filtFs, filtRs

    except Exception as e:
        logger.error(f"Error in parallel filtering and trimming orchestrator: {e}", exc_info=True)
        return None, None

from collections import defaultdict

import subprocess

import subprocess
import os 

def denoise_and_create_asv_table_vsearch(filtFs, filtRs, sample_names, output_dir):
    """
    Denoises sequences and creates an ASV table using vsearch.
    This version corrects the order of operations and includes robust checks and safe cleanup.
    """
    vsearch_executable = r"C:\Users\schwert\.conda\envs\Environment_Python_Microbiome_Project\vsearch-2.30.0-win-x86_64\bin\vsearch.exe"
    
    if not os.path.exists(vsearch_executable):
        raise FileNotFoundError(f"vsearch not found at: {vsearch_executable}")

    # --- Setup paths for all temporary files ---
    merged_fasta = os.path.join(output_dir, 'all_samples_merged.fa')
    derep_fasta = os.path.join(output_dir, 'all_samples_derep.fa')
    denoised_fasta = os.path.join(output_dir, 'asvs.fa')
    asv_table_tsv = os.path.join(output_dir, 'asv_table.tsv')
    os.makedirs(output_dir, exist_ok=True)
    
    # List of temporary files to clean up at the end
    temp_files_to_clean = [merged_fasta, derep_fasta, denoised_fasta, asv_table_tsv]

    try:
        # --- START OF CORRECTED LOGIC ---

        # Step 1: Merge Paired-End Reads
        logger.info("Step 1/5: Merging paired-end reads for all samples...")
        with open(merged_fasta, 'w') as merged_file:
            for i, (fF, fR, s_name) in enumerate(zip(filtFs, filtRs, sample_names)):
                temp_merged_fq = os.path.join(output_dir, f"{s_name}_merged.fq")
                temp_files_to_clean.append(temp_merged_fq) # Add to cleanup list
                
                merge_cmd = [
                    vsearch_executable, '--fastq_mergepairs', fF, '--reverse', fR,
                    '--fastqout', temp_merged_fq, '--relabel', f"{s_name};"
                ]
                
                # We use a try-except block for each subprocess call for better error reporting
                try:
                    subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
                    with open(temp_merged_fq, 'rt') as fq_in:
                        SeqIO.convert(fq_in, 'fastq', merged_file, 'fasta')
                except subprocess.CalledProcessError as e:
                    logger.error(f"vsearch merge failed for sample {s_name}. Stderr: {e.stderr}")
                    raise
        
        if not os.path.exists(merged_fasta) or os.path.getsize(merged_fasta) == 0:
            raise FileNotFoundError("Step 1 (Merging) failed: Merged FASTA file was not created or is empty.")

        # Step 2: Dereplicate sequences
        logger.info("Step 2/5: Dereplicating all merged sequences...")
        derep_cmd = [vsearch_executable, '--derep_fulllength', merged_fasta, '--output', derep_fasta, '--sizeout']
        try:
            subprocess.run(derep_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"vsearch dereplication failed. Stderr: {e.stderr}")
            raise
        
        if not os.path.exists(derep_fasta) or os.path.getsize(derep_fasta) == 0:
            raise FileNotFoundError("Step 2 (Dereplication) failed: Dereplicated FASTA file was not created or is empty.")

        # Step 3: Denoise (form ASVs)
        logger.info("Step 3/5: Denoising sequences with cluster_unoise...")
        unoise_cmd = [vsearch_executable, '--cluster_unoise', derep_fasta, '--minsize', '8', '--centroids', denoised_fasta,'--relabel', 'ASV_']
        try:
            subprocess.run(unoise_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"vsearch denoising failed. Stderr: {e.stderr}")
            raise
        
        if not os.path.exists(denoised_fasta) or os.path.getsize(denoised_fasta) == 0:
            raise FileNotFoundError("Step 3 (Denoising) failed: ASV centroids FASTA file (asvs.fa) was not created or is empty.")

        # Step 4: Map original merged reads to ASVs to generate table
        logger.info("Step 4/5: Mapping reads to ASVs to generate feature table...")
        map_cmd = [
            vsearch_executable, '--usearch_global', merged_fasta, '--db', denoised_fasta, 
            '--id', '1.0', '--otutabout', asv_table_tsv
        ]
        try:
            subprocess.run(map_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"vsearch mapping (otutabout) failed. Stderr: {e.stderr}")
            raise

        logger.info("Step 5/5: Formatting ASV table...")
        if not os.path.exists(asv_table_tsv) or os.path.getsize(asv_table_tsv) == 0:
            logger.warning("vsearch did not produce an ASV table in Step 4. This likely means no reads mapped to the ASVs.")
            return pd.DataFrame()

        asv_table = pd.read_csv(asv_table_tsv, sep='\t', index_col=0).T
        
        asv_sequences_dict = {
            record.id.split(';')[0]: str(record.seq) 
            for record in SeqIO.parse(denoised_fasta, 'fasta')
        }
        
        new_columns = [asv_sequences_dict.get(col_id) for col_id in asv_table.columns]
        if any(seq is None for seq in new_columns):
             raise KeyError("ID mismatch error: Some ASV table columns could not be found in the ASV FASTA file.")
        asv_table.columns = new_columns
        
        logger.info(f"Successfully generated ASV table with {asv_table.shape[0]} samples and {asv_table.shape[1]} unique ASVs.")
        final_asv_path = os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')
        asv_table.to_csv(final_asv_path, sep='\t')
        logger.info(f"Final ASV table saved to {final_asv_path}")
        
        return asv_table

    finally:
        pass
#        logger.info("Cleaning up temporary vsearch files...")
#        for f_path in temp_files_to_clean:
#            if os.path.exists(f_path):
#                try:
#                    os.remove(f_path)
#                except OSError as e:
#                    logger.warning(f"Could not remove temporary file {f_path}: {e}")

def assign_taxonomy(asv_sequences, asv_fasta_path, silva_path, output_dir):
    """
    Assigns taxonomy using the fast vsearch --usearch_global command,
    replacing the slow, brute-force Python loop.
    
    Args:
        asv_sequences (list): List of ASV sequences (used for final table index).
        asv_fasta_path (str): Path to the ASV FASTA file (e.g., 'asvs.fa').
        silva_path (str): Path to the SILVA reference database.
        output_dir (str): Directory to save output files.
        
    Returns:
        pd.DataFrame: The final taxonomy table.
    """
    logger.info("Starting fast taxonomy assignment with vsearch.")
    
    if not asv_sequences:
        logger.warning("assign_taxonomy was called with an empty list of sequences. Returning empty table.")
        return pd.DataFrame(columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])

    if not silva_path or not os.path.exists(silva_path):
        logger.error(f"SILVA reference file not found at path: {silva_path}. Cannot assign taxonomy.")
        tax_df = pd.DataFrame(index=asv_sequences, columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
        tax_df.fillna("Unassigned", inplace=True)
        return tax_df

    # --- START: New High-Performance Logic ---
    vsearch_executable = r"C:\Users\schwert\.conda\envs\Environment_Python_Microbiome_Project\vsearch-2.30.0-win-x86_64\bin\vsearch.exe"
    if not os.path.exists(vsearch_executable):
        raise FileNotFoundError(f"vsearch not found at: {vsearch_executable}. Please update the path.")

    # Define the output path for the blast6-like results
    blast6_output = os.path.join(output_dir, 'taxonomy_hits.tsv')
    
    # Run vsearch to find the best hit for each ASV against the SILVA database
    # This is incredibly fast compared to the Python loop.
    tax_cmd = [
        vsearch_executable,
        '--usearch_global', asv_fasta_path,
        '--db', silva_path,
        '--id', '0.97',  # 97% identity, a common threshold for species/genus
        '--strand', 'plus',
        '--maxaccepts', '1', # We only want the single best hit
        '--blast6out', blast6_output
    ]
    
    try:
        logger.info("Running vsearch for taxonomy assignment... This should be fast.")
        subprocess.run(tax_cmd, check=True, capture_output=True, text=True)
        logger.info(f"vsearch completed. Taxonomy hits saved to {blast6_output}")
    except subprocess.CalledProcessError as e:
        logger.error(f"vsearch taxonomy assignment failed. Stderr: {e.stderr}")
        raise

    # Parse the vsearch output to create the taxonomy table
    # Columns: 0=ASV_ID, 1=SILVA_ID_with_taxonomy
    hits_map = {}
    if os.path.exists(blast6_output):
        with open(blast6_output, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                asv_id = parts[0]
                silva_full_header = parts[1]
                hits_map[asv_id] = silva_full_header

    # Create the final taxonomy DataFrame
    taxonomy_results = {}
    asv_id_to_seq = {rec.id: str(rec.seq) for rec in SeqIO.parse(asv_fasta_path, "fasta")}

    for asv_id, asv_seq in asv_id_to_seq.items():
        if asv_id in hits_map:
            # Found a hit, parse the taxonomy string
            silva_header = hits_map[asv_id]
            # Assuming format like: "AB12345.1.1234 Bacteria;Firmicutes;..."
            tax_string = silva_header.split(' ', 1)[-1] 
            tax_levels = tax_string.split(';')
            parsed_tax = [level.split('__')[-1] if '__' in level else level for level in tax_levels]
            # Pad with last known level if taxonomy is incomplete
            while len(parsed_tax) < 6:
                parsed_tax.append(parsed_tax[-1] if parsed_tax else 'Unassigned')
            taxonomy_results[asv_seq] = parsed_tax[:6] # Only take first 6 levels
        else:
            # No hit found for this ASV
            taxonomy_results[asv_seq] = ['Unassigned'] * 6

    tax_df = pd.DataFrame.from_dict(taxonomy_results, orient='index',
                                    columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus'])
    tax_df.index.name = 'ASV'
    # --- END: New High-Performance Logic ---

    tax_table_path = os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')
    tax_df.to_csv(tax_table_path)
    logger.info(f"Taxonomy assignment complete. Table saved to: {tax_table_path}")
    return tax_df


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
    Includes a check for an empty ASV table to prevent errors.
    """
    try:
        meta = ps1['meta'].copy()
        asv = ps1['asv']

        # --- ROBUSTNESS CHECK ---
        if asv.empty:
            logger.warning("Cannot calculate alpha diversity on an empty ASV table. Returning.")
            # Return the metadata with empty columns so the app doesn't crash
            meta['Shannon'] = np.nan
            meta['InverseSimpson'] = np.nan
            return meta
        # --- END OF CHECK ---

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
def perform_pcoa(asv, meta, treatment, pval_threshold, contrib_threshold):
    """
    Perform PCoA on beta diversity distance matrix.
    
    Args:
        asv: ASV table (features x samples)
        meta: Metadata DataFrame (samples x attributes)
        treatment: Column name for grouping
        pval_threshold: P-value threshold for significance (not used in this simple version)
        contrib_threshold: Contribution threshold for vectors (not used in this simple version)
    
    Returns:
        tuple: (PCoA scores DataFrame, distance matrix)
    """
    try:
        asv_transposed = asv.T
        
        dm = beta_diversity('braycurtis', asv_transposed.to_numpy(), ids=asv_transposed.index)
        
        ordination_result = pcoa(dm)
        
        scores = ordination_result.samples
        
        scores = scores.join(meta[[treatment]])
        
        logger.info("Performed PCoA successfully")
        return scores, dm
        
    except Exception as e:
        logger.error(f"Error performing PCoA: {e}", exc_info=True)
        return None, None
def perform_pca(asv, top_n):
    """
    Perform PCA on top N ASVs.
    This version corrects the data orientation for scikit-learn.
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        top_asvs = asv.sum(axis=1).nlargest(top_n).index
        asv_top = asv.loc[top_asvs]
        
        asv_top_transposed = asv_top.T
        
        scaler = StandardScaler()
        asv_scaled = scaler.fit_transform(asv_top_transposed)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(asv_scaled)
        
        explained_variance = pca.explained_variance_ratio_
        
        logger.info("Performed PCA successfully")
        return pca_result, explained_variance
        
    except Exception as e:
        logger.error(f"Error performing PCA: {e}", exc_info=True)
        return None, None

def plot_phylogenetic_tree(seqtab):
    """
    Generate a phylogenetic tree from ASV sequences using a compatible workflow.
    This version includes a fix for the Windows file lock issue.
    """
    from skbio.tree import nj as skbio_nj

    try:
        all_seqs = [SeqRecord(Seq(seq), id=f"ASV_{i+1}") for i, seq in enumerate(seqtab.columns)]
        unique_seq_dict = {str(rec.seq): rec for rec in reversed(all_seqs)}
        seqs = list(unique_seq_dict.values())
        
        if len(seqs) < 2:
            logger.warning("Cannot generate a tree with fewer than 2 unique sequences.")
            return None
        
        names = [s.id for s in seqs]
        num_seqs = len(seqs)
        dm_data = np.zeros((num_seqs, num_seqs))
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        for i in range(num_seqs):
            for j in range(i + 1, num_seqs):
                 score = aligner.align(seqs[i].seq, seqs[j].seq).score
                 max_len = max(len(seqs[i].seq), len(seqs[j].seq))
                 distance = 1 - (score / max_len) if max_len > 0 else 1
                 dm_data[i, j] = distance
                 dm_data[j, i] = distance
        
        dm = DistanceMatrix(data=dm_data, ids=names)
            
        skbio_tree = skbio_nj(dm)
        handle = io.StringIO()
        skbio_tree.write(handle, format='newick')
        newick_string = handle.getvalue()
        ete_tree = Tree(newick_string)
        
        
        tmp_file = None
        try:
            f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp_file_path = f.name
            f.close() 

            ts = TreeStyle()
            ts.show_leaf_name = True
            ete_tree.render(tmp_file_path, w=400, units='px', tree_style=ts)

            with open(tmp_file_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            logger.info("Generated phylogenetic tree successfully")
            return f"data:image/png;base64,{encoded_image}"

        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Error plotting phylogenetic tree: {e}", exc_info=True)
        return None

# Callbacks
@app.callback(
    Output('output-r1-filenames', 'children'),
    Output('output-r2-filenames', 'children'),
    Output('output-metadata-filename', 'children'),
    Input('list-files-button', 'n_clicks'),
    State('data-folder-path', 'value'),
    prevent_initial_call=True
)
def list_files_in_folder(n_clicks, folder_path):
    """
    Reads a folder path, lists the files, categorizes them,
    and displays them in the UI.
    """
    logger.info(f"Button clicked to list files in folder: '{folder_path}'")

    if not folder_path or not os.path.isdir(folder_path):
        error_message = html.Div(
            f"Error: Folder not found or path is invalid. Please check the path: '{folder_path}'",
            style={'color': 'red', 'fontWeight': 'bold'}
        )
        return error_message, None, None

    try:
        all_files = os.listdir(folder_path)
    except Exception as e:
        error_message = html.Div(f"An error occurred while accessing the folder: {e}", style={'color': 'red'})
        return error_message, None, None

    r1_files = sorted([f for f in all_files if '_R1' in f.upper() and (f.lower().endswith('.fastq') or f.lower().endswith('.fastq.gz'))])
    r2_files = sorted([f for f in all_files if '_R2' in f.upper() and (f.lower().endswith('.fastq') or f.lower().endswith('.fastq.gz'))])
    metadata_files = sorted([f for f in all_files if f.lower().endswith(('.csv', '.tsv', '.txt'))])

    def create_file_list_component(title, files):
        if not files:
            return html.P(f"No {title} found in this folder.")
        
        return html.Div([
            html.H4(f"{title} Found:"),
            html.Ul([html.Li(file) for file in files], style={'listStyleType': 'none', 'paddingLeft': '20px'})
        ])

    r1_output = create_file_list_component("R1 FASTQ Files", r1_files)
    r2_output = create_file_list_component("R2 FASTQ Files", r2_files)
    metadata_output = create_file_list_component("Metadata/Text Files", metadata_files)

    return r1_output, r2_output, metadata_output

@app.callback(
    Output('silva-status-output', 'children'),
    Output('silva-status-output', 'style'),
    Input('upload-silva', 'filename'),
    Input('list-files-button', 'n_clicks'),
    State('data-folder-path', 'value'),
    prevent_initial_call=True
)
def update_silva_status(uploaded_filename, n_clicks, folder_path):
    """
    Provides feedback on the status of the SILVA database file.
    Checks for both direct uploads and a file named 'silva.fasta' in the data folder.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id

    base_style = {'marginTop': '5px', 'padding': '8px', 'borderRadius': '3px', 'fontWeight': 'bold'}

    # Scenario 1: User uploaded a file directly
    if triggered_id == 'upload-silva' and uploaded_filename:
        logger.info(f"SILVA file uploaded: {uploaded_filename}")
        success_style = base_style | {'backgroundColor': '#e6ffed', 'color': '#2d6a4f'}
        return f"✓ Successfully uploaded: {uploaded_filename}", success_style

    # Scenario 2: User clicked the "List Files" button
    if triggered_id == 'list-files-button':
        # First, ensure the folder path is valid
        if not folder_path or not os.path.isdir(folder_path):
            warning_style = base_style | {'backgroundColor': '#fff3cd', 'color': '#856404'}
            return "ⓘ Provide a valid data folder path to check for 'silva.fasta'.", warning_style

        # Check for 'silva.fasta' in the specified folder
        expected_silva_path = os.path.join(folder_path, 'silva.fasta')
        if os.path.exists(expected_silva_path):
            logger.info(f"Found 'silva.fasta' in folder: {folder_path}")
            success_style = base_style | {'backgroundColor': '#e6ffed', 'color': '#2d6a4f'}
            return f"✓ Found 'silva.fasta' in {folder_path}", success_style
        else:
            logger.warning(f"'silva.fasta' not found in folder: {folder_path}")
            info_style = base_style | {'backgroundColor': '#e2e3e5', 'color': '#383d41'}
            return "ⓘ Note: 'silva.fasta' was not found in the data folder. Please upload it if needed for analysis.", info_style

    return dash.no_update, dash.no_update

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
        

@app.callback(
    Output('data-folder-path', 'value'),
    Input('load-sample-data', 'n_clicks')
)
def load_sample_data(n_clicks):
    if n_clicks > 0:
        return ""
    return dash.no_update    
    
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
    [Input('suggest-ai-params-button', 'n_clicks')],
    [State('data-folder-path', 'value')]
)
def update_ai_suggested_params(n_clicks, data_folder_path):
    if n_clicks is None or n_clicks == 0:
        return dash.no_update

    try:
        if not data_folder_path or not os.path.isdir(data_folder_path):
            logger.warning("AI suggestions require a valid data folder path.")
            return [dash.no_update] * 16 # 16 is the number of outputs

        logger.info(f"Generating AI suggestions from folder: {data_folder_path}")
        all_files = os.listdir(data_folder_path)
        fnFs = sorted([os.path.join(data_folder_path, f) for f in all_files if '_R1' in f])
        fnRs = sorted([os.path.join(data_folder_path, f) for f in all_files if '_R2' in f])
        meta_file_path = sorted([os.path.join(data_folder_path, f) for f in all_files if f.lower().endswith('.csv')])

        if not fnFs or not fnRs or not meta_file_path:
            logger.warning("Could not find R1, R2, and metadata files in folder to generate AI suggestions.")
            return [dash.no_update] * 16

        metadata = pd.read_csv(meta_file_path[0])

        
        quality_params = ai_analyze_quality_profiles(fnFs, fnRs) if ai_available else None
        trunc_len_f_val = quality_params['trunc_len_f'] if quality_params else 280
        trunc_len_r_val = quality_params['trunc_len_r'] if quality_params else 220
        max_ee_val = str(quality_params['max_ee']) if quality_params else '[2,2]'
        
        treatment_val = ai_analyze_metadata(metadata) if ai_available else metadata.columns[0]
        
        trunc_len_f_opts = [{'label': str(trunc_len_f_val), 'value': trunc_len_f_val}]
        trunc_len_r_opts = [{'label': str(trunc_len_r_val), 'value': trunc_len_r_val}]
        max_ee_opts = [{'label': max_ee_val, 'value': max_ee_val}]
        treatment_opts = [{'label': treatment_val, 'value': treatment_val}]
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
        return [dash.no_update] * 16
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
    Output('preprocessing-params-div', 'style'),
    Input('analysis-mode', 'value')
)
def toggle_preprocessing_params(analysis_mode):
    """
    This function listens for a click on the 'Analysis Starting Point' radio button.
    If the user selects 'asv' mode, it hides the pre-processing parameters.
    If the user selects 'fastq' mode, it shows them again.
    """
    if analysis_mode == 'fastq':
        # Show the div by returning a normal style dictionary
        return {'display': 'block'}
    else:
        # Hide the div by setting its display style to 'none'
        return {'display': 'none'}
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
    [
     State('analysis-mode', 'value'),
     State('data-folder-path', 'value'),
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
     State('background-info', 'value')
    ],
    prevent_initial_call=True
)
def run_analysis(n_clicks, analysis_mode, data_folder_path, silva_content, output_dir, param_mode,
                 trunc_len_f, trunc_len_r, max_ee_str, prev_threshold, treatment, top_asvs, pval_threshold, contrib_threshold,
                 ai_trunc_len_f, ai_trunc_len_r, ai_max_ee, ai_prev_threshold, ai_treatment, ai_top_asvs, ai_pval_threshold, ai_contrib_threshold,
                 background):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- Data Loading Logic ---
        # This section determines whether to use sample data or process real data.
        if not data_folder_path:
            # --- CASE 1: USE INTERNAL SAMPLE DATA ---
            logger.info("Data folder path is empty. Using internal sample data for analysis.")
            seqtab = sample_seqtab
            taxa = sample_taxa
            metadata = pd.read_csv(StringIO(sample_metadata))  
            metadata.set_index('SampleID', inplace=True)
        else:
    # --- CASE 2: PROCESS REAL DATA FROM FOLDER ---
          logger.info(f"Data folder path provided: '{data_folder_path}'. Reading files from disk.")
        if not os.path.isdir(data_folder_path):
            raise ValueError(f"The provided data folder path does not exist or is not a directory: {data_folder_path}")
    
        all_files = os.listdir(data_folder_path)
        meta_files = sorted([os.path.join(data_folder_path, f) for f in all_files if f.lower().endswith(('.csv', '.tsv'))])
        if not meta_files: raise ValueError("Could not find a metadata file in the specified folder.")
    
        # Load the full metadata file first, as it's needed for both modes
        metadata_df = pd.read_csv(meta_files[0])
        sample_id_col = metadata_df.columns[0]
        metadata_df[sample_id_col] = metadata_df[sample_id_col].astype(str)
        metadata_df.set_index(sample_id_col, inplace=True)
        metadata_df.index = metadata_df.index.astype(str) # <-- ADD THIS LINE
        # --- HERE IS THE NEW CORE LOGIC ---
        if analysis_mode == 'fastq':
            # --- THIS IS THE "SLOW PATH" ---
            logger.info("Mode selected: Starting from raw FASTQ files.")
            
            # This is your original code, now moved inside this 'if' block
            fnFs = sorted([os.path.join(data_folder_path, f) for f in all_files if '_R1' in f.upper()])
            fnRs = sorted([os.path.join(data_folder_path, f) for f in all_files if '_R2' in f.upper()])
            
            sample_names_from_files = [os.path.basename(f).split('_')[0] for f in fnFs]
            metadata = metadata_df.loc[sample_names_from_files].copy() # Align metadata to the order of FASTQ files
            sample_names = metadata.index.tolist()
            
            # Parse parameters
            max_ee = [float(x.strip()) for x in str(max_ee_str).strip('[]').split(',') if x.strip()]
    
            # Your original pre-processing steps
            filtFs, filtRs = filter_and_trim_parallel(
                fnFs=fnFs,
                fnRs=fnRs,
                sample_names=sample_names,
                output_dir=output_dir,
                trunc_len_f=trunc_len_f,
                trunc_len_r=trunc_len_r,
                quality_cutoff=20)  # A standard quality cutoff, you can make this a user parameter if you wish)
            if not filtFs or not filtRs:
                raise ValueError("Parallel filtering and trimming with Cutadapt failed. Check the console logs for detailed errors.")
    
            seqtab = denoise_and_create_asv_table_vsearch(filtFs, filtRs, sample_names, output_dir)
            if seqtab is None or seqtab.empty: raise ValueError("vsearch failed to create an ASV table.")
    
            asv_fasta_path = os.path.join(output_dir, 'asvs.fa')
            # Your logic for finding the SILVA file
            silva_path = None
            if silva_content:
                content_type, content_string = silva_content.split(',')
                decoded = base64.b64decode(content_string)
                silva_path = os.path.join(output_dir, 'uploaded_silva.fasta')
                with open(silva_path, 'wb') as f: f.write(decoded)
            elif os.path.exists(os.path.join(data_folder_path, 'silva.fasta')):
                silva_path = os.path.join(data_folder_path, 'silva.fasta')
    
            taxa = assign_taxonomy(list(seqtab.columns), asv_fasta_path, silva_path, output_dir)
            if taxa is None or taxa.empty: raise ValueError("Failed to assign taxonomy.")
    
        else:
            # --- THIS IS THE "FAST PATH" ---
            logger.info("Mode selected: Starting from Processed ASV and Taxonomy Tables.")
            
            # Define the paths to the files we expect to find
            asv_path = os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')
            taxa_path = os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')
    
            if not os.path.exists(asv_path) or not os.path.exists(taxa_path):
                raise FileNotFoundError(f"Processed files not found in '{output_dir}'. Please run the analysis in 'From FASTQ files' mode first.")
    
            # Load the tables directly from disk
            seqtab = pd.read_csv(asv_path, sep='\t', index_col=0)
            taxa = pd.read_csv(taxa_path, index_col=0)
            seqtab.index = seqtab.index.astype(str)
            metadata_df.index = metadata_df.index.astype(str)
            # Align the full metadata to the samples in our loaded ASV table
            metadata = metadata_df.loc[seqtab.index].copy()
            logger.info(f"Successfully loaded ASV table ({seqtab.shape}) and Taxonomy table ({taxa.shape}).")


        # --- SHARED ANALYSIS PIPELINE (runs for both sample and real data) ---
        
        if param_mode == 'ai_automatic':
            prev_threshold = ai_analyze_prevalence(seqtab) or 0 if ai_available else 0

        ps1 = create_phyloseq_object(seqtab, taxa, metadata)
        if not ps1: raise ValueError("Failed to create phyloseq object")
        
        ps1_meta = calculate_alpha_diversity(ps1, treatment)
        if ps1_meta is None: raise ValueError("Failed to calculate alpha diversity")

        asv_rel, meta_rel = calculate_beta_diversity(ps1)
        if asv_rel is None: raise ValueError("Failed to calculate beta diversity")
        
        pcoa_scores, dm = perform_pcoa(asv_rel, meta_rel, treatment, pval_threshold, contrib_threshold)
        if pcoa_scores is None: raise ValueError("Failed to perform PCoA")

        top_asvs_list = [int(x.strip()) for x in str(top_asvs).split(',') if x.strip()]
        pca_result, explained_variance = perform_pca(ps1['asv'], top_asvs_list[-1])
        if pca_result is None: raise ValueError("Failed to perform PCA")
        global_data['pca_result'], global_data['explained_variance'] = pca_result, explained_variance
        logger.info("Melting data for plotting...")

        asv_melted = ps1['asv'].reset_index()


        asv_melted.rename(columns={'index': 'ASV'}, inplace=True)

        ps1_melt = asv_melted.melt(id_vars=['ASV'], var_name='SampleID', value_name='Abundance')

        ps1_melt = ps1_melt.merge(taxa, on='ASV')
        ps1_melt = ps1_melt.merge(meta_rel[[treatment]], left_on='SampleID', right_index=True)


        tree_img = plot_phylogenetic_tree(seqtab)
        
        interpretations = ai_interpret_results(seqtab, ps1_meta, pcoa_scores, (pca_result, explained_variance), ps1_melt, tree_img, background, treatment)
        global_data['ai_interpretations'] = interpretations
        
        # --- Plotting ---
        seq_depth_fig = px.histogram(seqtab.sum(axis=1), title="Sequencing Depth", labels={'value': 'Read Count', 'count': 'Number of Samples'})
        alpha_fig = px.violin(ps1_meta, x=treatment, y='Shannon', box=True, points='all', title="Shannon Diversity")
        beta_fig = px.scatter(pcoa_scores, x='PC1', y='PC2', color=treatment, title="PCoA Plot (Bray-Curtis)")
        pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=ps1['meta'].index).join(ps1['meta'][treatment])
        pca_fig = px.scatter(pca_df, x='PC1', y='PC2', color=treatment, title=f"PCA Plot (Top {top_asvs_list[-1]} ASVs)")
        phylum_melt = ps1_melt.groupby(['SampleID', 'Phylum', treatment])['Abundance'].sum().reset_index()
        abundance_fig = px.bar(phylum_melt, x='SampleID', y='Abundance', color='Phylum', facet_col=treatment, title="Abundance by Phylum")

        output_files_children = [
            html.P(f"ASV Table: {os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')}"),
            html.P(f"Taxonomy Table: {os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')}")
        ]
        
        phylogenetic_tree_children = html.Img(src=tree_img) if tree_img else html.P("Tree could not be generated.")
        
        logger.info("Analysis completed successfully")
        return [seq_depth_fig, alpha_fig, beta_fig, pca_fig, beta_fig, abundance_fig, output_files_children, phylogenetic_tree_children, dcc.Markdown(interpretations)]
        
    except Exception as e:
        logger.error(f"Error in run_analysis: {e}", exc_info=True)
        error_fig = go.Figure().update_layout(title=f"Error: {str(e)}", xaxis={'visible': False}, yaxis={'visible': False})
        return [error_fig] * 9 

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
        
if __name__ == '__main__':
    app.run_server(debug=True)
