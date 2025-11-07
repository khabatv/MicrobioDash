
import os
import base64
import uuid
import zipfile
import pandas as pd
import numpy as np
from .config import logger

def unzip_files(uploaded_files, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        for file_content in uploaded_files:
            content_type, content_string = file_content.split(',')
            decoded = base64.b64decode(content_string)
            temp_path = os.path.join(output_dir, f"temp_{uuid.uuid4()}.zip")
            with open(temp_path, 'wb') as f:
                f.write(decoded)
            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(temp_path)
        logger.info(f"Successfully unzipped files to: {output_dir}")
    except Exception as e:
        logger.error(f"Error unzipping files: {e}")

def fill_taxonomy_forward(taxa_df):
    logger.info("Performing forward-fill on taxonomy table to handle unclassified ranks.")
    ranks = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    filled_taxa = taxa_df.copy()
    missing_markers = {'', None, np.nan, 'unassigned', 'unclassified'}

    for index, row in filled_taxa.iterrows():
        last_known_taxon = None
        for rank in ranks:
            current_taxon = row[rank]
            is_missing = pd.isna(current_taxon) or str(current_taxon).strip().lower() in missing_markers
            if is_missing:
                if last_known_taxon:
                    placeholder = f"{last_known_taxon}_unclassified_{rank.lower()}"
                    filled_taxa.at[index, rank] = placeholder
            else:
                last_known_taxon = current_taxon
    return filled_taxa

def run_diagnostic_checks(ps1_object, step_name):
    logger.info(f"--- [DIAGNOSTICS] Running checks for: {step_name} ---")
    asv, tax = ps1_object['asv'], ps1_object['tax']
    logger.info(f"Shape of ASV table: {asv.shape}, Shape of TAX table: {tax.shape}")
    if asv.index.equals(tax.index):
        logger.info("✅ SUCCESS: ASV table index and TAX table index are identical.")
        return True
    else:
        logger.error("❌ CRITICAL FAILURE: ASV table index and TAX table index DO NOT MATCH.")
        asv_only = asv.index.difference(tax.index)
        tax_only = tax.index.difference(asv.index)
        if len(asv_only) > 0: logger.error(f"Found {len(asv_only)} features in ASV but not TAX table.")
        if len(tax_only) > 0: logger.error(f"Found {len(tax_only)} features in TAX but not ASV table.")
        return False
