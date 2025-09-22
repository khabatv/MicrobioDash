
import os
import gzip
import multiprocessing
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from .config import logger, CPU_CORES

# --- VSEARCH WORKER FUNCTIONS (for parallel processing) ---

def merge_worker(args):
    fnF, fnR, sample_name, vsearch_executable, output_dir = args
    merged_output_path = os.path.join(output_dir, f"{sample_name}_merged.fastq")
    try:
        merge_cmd = [
            vsearch_executable, '--fastq_mergepairs', fnF, '--reverse', fnR,
            '--fastq_maxdiffs', '10', '--fastq_pctid', '90',
            '--fastqout', merged_output_path
        ]
        result = subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully merged reads for sample: {sample_name}")
        return merged_output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"VSEARCH merge failed for {sample_name}. Stderr: {e.stderr}")
        return None

def calculate_distance_pair_worker(args):
    # This is a placeholder as it's not used in the main analysis flow,
    # but was in the original imports.
    logger.warning("`calculate_distance_pair_worker` is a placeholder and was not called.")
    return None

# --- MAIN PIPELINE STEPS ---

def _filter_and_trim_worker(args):
    fnF, fnR, sample_name, output_dir, trunc_len_f, trunc_len_r, max_ee = args
    filt_path = os.path.join(output_dir, "filtered_sequences")
    filtF = os.path.join(filt_path, f"{sample_name}_F_filt.fastq.gz")
    filtR = os.path.join(filt_path, f"{sample_name}_R_filt.fastq.gz")

    logger.info(f"Processing sample '{sample_name}': {os.path.basename(fnF)} & {os.path.basename(fnR)}")
    try:
        open_f = gzip.open if fnF.lower().endswith('.gz') else open
        open_r = gzip.open if fnR.lower().endswith('.gz') else open
        with open_f(fnF, 'rt') as f_in, open_r(fnR, 'rt') as r_in:
            records_f_iter, records_r_iter = SeqIO.parse(f_in, 'fastq'), SeqIO.parse(r_in, 'fastq')
            kept_records_f, kept_records_r = [], []
            initial_count = 0
            for rec_f, rec_r in zip(records_f_iter, records_r_iter):
                initial_count += 1
                if (len(rec_f) >= trunc_len_f and len(rec_r) >= trunc_len_r and
                    np.mean(rec_f.letter_annotations['phred_quality']) >= 30 and
                    np.mean(rec_r.letter_annotations['phred_quality']) >= 30):
                    kept_records_f.append(rec_f[:trunc_len_f])
                    kept_records_r.append(rec_r[:trunc_len_r])

        final_count = len(kept_records_f)
        if initial_count > 0:
            logger.info(f"[{sample_name}] Kept {final_count}/{initial_count} read pairs ({final_count/initial_count:.2%})")
        else:
            logger.warning(f"[{sample_name}] No read pairs found.")
            return None, None

        with gzip.open(filtF, 'wt') as f_out: SeqIO.write(kept_records_f, f_out, 'fastq')
        with gzip.open(filtR, 'wt') as r_out: SeqIO.write(kept_records_r, r_out, 'fastq')
        return filtF, filtR
    except Exception as e:
        logger.error(f"Error processing sample {sample_name}: {e}", exc_info=True)
        return None, None

def filter_and_trim_parallel(fnFs, fnRs, sample_names, output_dir, trunc_len_f, trunc_len_r, max_ee):
    filt_path = os.path.join(output_dir, "filtered_sequences")
    os.makedirs(filt_path, exist_ok=True)
    args_list = [(fnF, fnR, s_name, output_dir, trunc_len_f, trunc_len_r, max_ee) for fnF, fnR, s_name in zip(fnFs, fnRs, sample_names)]
    with multiprocessing.Pool(processes=CPU_CORES) as pool:
        results = pool.map(_filter_and_trim_worker, args_list)
    filtFs = [res[0] for res in results if res and res[0]]
    filtRs = [res[1] for res in results if res and res[1]]
    if not filtFs or not filtRs: raise RuntimeError("Filtering step failed to produce output files.")
    logger.info(f"Successfully filtered and trimmed sequences in parallel to: {filt_path}")
    return filtFs, filtRs

def denoise_and_create_asv_table_vsearch(filtFs, filtRs, sample_names, output_dir):
    # IMPORTANT: Update this path to your vsearch executable if it's not in your system's PATH
    vsearch_executable = "vsearch" 

    merged_fasta = os.path.join(output_dir, 'all_samples_merged.fa')
    derep_fasta = os.path.join(output_dir, 'all_samples_derep.fa')
    denoised_fasta = os.path.join(output_dir, 'asvs.fa')
    asv_table_tsv = os.path.join(output_dir, 'asv_table.tsv')

    try:
        # Step 1: Merge Paired-End Reads
        logger.info("Step 1/5: Merging paired-end reads in parallel...")
        args_list = [(fF, fR, s_name, vsearch_executable, output_dir) for fF, fR, s_name in zip(filtFs, filtRs, sample_names)]
        with multiprocessing.Pool(processes=CPU_CORES) as pool:
            merged_fastq_files = pool.map(merge_worker, args_list)

        with open(merged_fasta, 'w') as merged_output:
            for fq_file in merged_fastq_files:
                if fq_file and os.path.exists(fq_file):
                    with open(fq_file, 'rt') as fq_in: SeqIO.convert(fq_in, 'fastq', merged_output, 'fasta')
                    os.remove(fq_file)

        # Step 2: Dereplicate
        logger.info("Step 2/5: Dereplicating all merged sequences...")
        derep_cmd = [vsearch_executable, '--derep_fulllength', merged_fasta, '--output', derep_fasta, '--sizeout', '--threads', str(CPU_CORES)]
        subprocess.run(derep_cmd, check=True, capture_output=True, text=True)

        # Step 3: Denoise (form ASVs)
        logger.info("Step 3/5: Denoising sequences with cluster_unoise...")
        unoise_cmd = [vsearch_executable, '--cluster_unoise', derep_fasta, '--minsize', '8', '--centroids', denoised_fasta, '--relabel', 'ASV_', '--threads', str(CPU_CORES)]
        subprocess.run(unoise_cmd, check=True, capture_output=True, text=True)

        # Step 4: Map reads to ASVs
        logger.info("Step 4/5: Mapping reads to ASVs to generate feature table...")
        map_cmd = [vsearch_executable, '--usearch_global', merged_fasta, '--db', denoised_fasta, '--id', '1.0', '--otutabout', asv_table_tsv, '--threads', str(CPU_CORES)]
        subprocess.run(map_cmd, check=True, capture_output=True, text=True)

        # Step 5: Format ASV table
        logger.info("Step 5/5: Formatting ASV table...")
        asv_table = pd.read_csv(asv_table_tsv, sep='\t', index_col=0, engine='python').T
        asv_sequences_dict = {rec.id.split(';')[0]: str(rec.seq) for rec in SeqIO.parse(denoised_fasta, 'fasta')}
        asv_table.columns = [asv_sequences_dict.get(col_id) for col_id in asv_table.columns]

        final_asv_path = os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')
        asv_table.to_csv(final_asv_path, sep='\t')
        logger.info(f"Final ASV table ({asv_table.shape}) saved to {final_asv_path}")
        return asv_table
    except subprocess.CalledProcessError as e:
        logger.error(f"A VSEARCH command failed. Stderr: {e.stderr}. Ensure 'vsearch' is in your system's PATH or update the path in pipeline_steps.py.")
        raise
    except Exception as e:
        logger.error(f"Error during denoising pipeline: {e}", exc_info=True)
        raise

def assign_taxonomy(asv_sequences, asv_fasta_path, silva_path, output_dir):
    logger.info("Starting taxonomy assignment with vsearch...")
    if not asv_sequences: return pd.DataFrame(columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    if not silva_path or not os.path.exists(silva_path):
        logger.error(f"SILVA reference file not found at: {silva_path}. Returning unassigned taxonomy.")
        return pd.DataFrame('Unassigned', index=asv_sequences, columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])

    vsearch_executable = "vsearch"
    blast6_output = os.path.join(output_dir, 'taxonomy_hits.tsv')
    tax_cmd = [vsearch_executable, '--usearch_global', asv_fasta_path, '--db', silva_path, '--id', '0.97', '--strand', 'plus', '--maxaccepts', '1', '--blast6out', blast6_output, '--threads', str(CPU_CORES)]

    try:
        subprocess.run(tax_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"vsearch taxonomy assignment failed. Stderr: {e.stderr}. Ensure 'vsearch' is in your system's PATH or update the path in pipeline_steps.py.")
        raise

    hits_map = {}
    if os.path.exists(blast6_output):
        with open(blast6_output, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    hits_map[parts[0]] = parts[1]

    taxonomy_results = {}
    asv_id_to_seq = {rec.id: str(rec.seq) for rec in SeqIO.parse(asv_fasta_path, "fasta")}
    for asv_id, asv_seq in asv_id_to_seq.items():
        if asv_id in hits_map:
            tax_string = hits_map[asv_id].split(' ', 1)[-1]
            parsed_tax = [level.split('__')[-1] for level in tax_string.split(';')]
            while len(parsed_tax) < 7: parsed_tax.append(parsed_tax[-1] if parsed_tax else 'Unassigned')
            taxonomy_results[asv_seq] = parsed_tax[:7]
        else:
            taxonomy_results[asv_seq] = ['Unassigned'] * 7

    tax_df = pd.DataFrame.from_dict(taxonomy_results, orient='index', columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    tax_df.index.name = 'ASV'
    tax_table_path = os.path.join(output_dir, 'microbiome_ai_taxonomy.csv')
    tax_df.to_csv(tax_table_path)
    logger.info(f"Taxonomy assignment complete. Table saved to: {tax_table_path}")
    return tax_df
