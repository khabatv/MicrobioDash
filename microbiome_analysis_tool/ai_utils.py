
import json
import numpy as np
import gzip
from Bio import SeqIO
import google.generativeai as genai
from .config import logger, GEMINI_API_KEY

# Attempt to configure the AI model
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    # Test connection with a very small request
    test_response = model.generate_content("Test", generation_config={'max_output_tokens': 5})
    if test_response and hasattr(test_response, 'text'):
        logger.info("Gemini AI API configured and tested successfully.")
        ai_available = True
    else:
        raise ValueError("Gemini API test failed.")
except Exception as e:
    logger.error(f"Could not configure or test Gemini AI API. AI features will be disabled. Error: {e}")
    ai_available = False
    model = None

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
        return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        logger.error(f"AI background analysis failed: {e}")
        return None, []

def ai_analyze_quality_profiles(fnFs, fnRs):
    if not ai_available: return None
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
        return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        logger.error(f"AI quality profile analysis failed: {e}")
        return None

def ai_analyze_prevalence(seqtab):
    if not ai_available: return 0
    prev = seqtab.sum(axis=0)
    prompt = f"""
    Given ASV prevalence (sum of reads):
    Mean: {prev.mean()}, Median: {prev.median()}, Min: {prev.min()}, Max: {prev.max()}
    Suggest a prevalence threshold for filtering rare ASVs (retain ASVs in >=1% of reads/samples).
    Return a single value.
    """
    try:
        response = model.generate_content(prompt)
        return float(response.text.strip())
    except Exception as e:
        logger.error(f"AI prevalence analysis failed: {e}")
        return None

def ai_analyze_metadata(metadata):
    if not ai_available: return None
    columns = metadata.columns.tolist()
    prompt = f"""
    Given metadata columns: {columns}
    Suggest the most appropriate column for grouping samples (categorical, 3-10 unique values).
    Return a single column name.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"AI metadata analysis failed: {e}")
        return None

def ai_analyze_pca_pcoa(asv, ordination_scores):
    if not ai_available: return None
    asv_sums = asv.sum()
    prompt = f"""
    Given ASV abundance sums (mean: {asv_sums.mean()}, median: {asv_sums.median()})
    and ordination variance explained (first two axes: {ordination_scores[:2]})
    suggest top ASVs for PCA and thresholds for PCoA vectors.
    Return: {{'top_asvs': 'X,Y,Z', 'pval_threshold': A, 'contrib_threshold': B}}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text.replace("'", '"'))
    except Exception as e:
        logger.error(f"AI PCA/PCoA analysis failed: {e}")
        return None

def ai_interpret_results(global_data, background, treatment_column_name):
    if not ai_available:
        return "AI interpretation unavailable. Please configure your Gemini API key in `config.py`."

    try:
        seqtab = global_data.get('seqtab_nochim')
        ps1_meta = global_data.get('ps1', {}).get('meta')
        pcoa_scores = global_data.get('pcoa_scores')
        pca_result = global_data.get('pca_result')
        ps1_melt = global_data.get('ps1_melt')

        if any(x is None for x in [seqtab, ps1_meta, pcoa_scores, pca_result, ps1_melt]):
            return "AI interpretation failed: Not all required data was available."

        if 'PC1' in pcoa_scores.columns and 'PC2' in pcoa_scores.columns:
            pcoa_variance_explained = pcoa_scores[['PC1', 'PC2']].var().values
        else:
            numeric_pcoa_scores = pcoa_scores.select_dtypes(include=np.number)
            pcoa_variance_explained = numeric_pcoa_scores.var().values[:2]

        ps1_melt_with_taxa = ps1_melt.merge(global_data['ps1']['tax']['Phylum'], left_on='ASV', right_index=True)
        top_phyla_list = ps1_melt_with_taxa.groupby('Phylum')['Abundance'].sum().nlargest(5).index.tolist()

        prompt = f"""
        As an expert microbiome data scientist, interpret the following results for a study with the background: "{background}".
        Provide a detailed, structured interpretation in Markdown format. Cover each section (Sequencing Depth, Diversity, Key Taxa, etc.) and conclude with a summary.

        RESULTS DATA:
        - Sequencing Depth: Mean={seqtab.sum(axis=1).mean():.0f}, Median={seqtab.sum(axis=1).median():.0f} reads per sample.
        - Alpha Diversity (Shannon): Mean={ps1_meta['Shannon'].mean():.2f}. The dataset has {ps1_meta[treatment_column_name].nunique()} groups.
        - Beta Diversity (PCoA): Variance explained by first two axes: {pcoa_variance_explained[0]*100:.1f}% and {pcoa_variance_explained[1]*100:.1f}%.
        - PCA: Variance explained by first two axes: {pca_result[1][0]*100:.1f}% and {pca_result[1][1]*100:.1f}%.
        - Abundance: The top 5 most abundant phyla are: {', '.join(top_phyla_list)}.
        - A phylogenetic tree was successfully generated.

        Based on this, generate a comprehensive report.
        """
        response = model.generate_content(prompt, request_options={'timeout': 120})
        return response.text if response and hasattr(response, 'text') else "AI interpretation failed to generate text."

    except Exception as e:
        logger.error(f"AI interpretation failed: {e}", exc_info=True)
        return f"AI interpretation failed due to an error: {e}"

