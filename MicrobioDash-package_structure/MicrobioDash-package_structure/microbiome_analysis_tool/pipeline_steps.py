import os
import gzip
import multiprocessing
import subprocess
import pandas as pd
import numpy as np
from Bio import SeqIO
from .config import logger, CPU_CORES

# --- VSEARCH WORKER FUNCTIONS (for parallel processing) ---
def _norm_id(rec_id: str) -> str:
    # Use the cluster id up to first whitespace; strip trailing /1 or /2 if present
    core = rec_id.split()[0]
    if core.endswith('/1') or core.endswith('/2'):
        core = core[:-2]
    return core

def sync_pairs_to_temp(fnF: str, fnR: str, out_dir: str, sample_name: str):
    """Create synchronized paired FASTQs (same IDs, same order). Returns (synced_F, synced_R)."""
    synced_F = os.path.join(out_dir, f"{sample_name}_F_sync.fastq")
    synced_R = os.path.join(out_dir, f"{sample_name}_R_sync.fastq")

    # index reverse reads in memory: id -> SeqRecord
    rmap = {}
    with _open_auto(fnR, 'rt') as rh:
        for rec in SeqIO.parse(rh, 'fastq'):
            rmap[_norm_id(rec.id)] = rec

    kept = 0
    with _open_auto(fnF, 'rt') as fh, \
         open(synced_F, 'w') as outF, \
         open(synced_R, 'w') as outR:
        for frec in SeqIO.parse(fh, 'fastq'):
            fid = _norm_id(frec.id)
            rrec = rmap.get(fid)
            if rrec is None:
                continue
            SeqIO.write(frec, outF, 'fastq')
            SeqIO.write(rrec, outR, 'fastq')
            kept += 1

    if kept == 0:
        raise RuntimeError(f"No overlapping read IDs between {os.path.basename(fnF)} and {os.path.basename(fnR)}.")
    logger.info(f"[{sample_name}] Pair sync kept {kept} read pairs.")
    return synced_F, synced_R

def merge_worker(args):
    fnF, fnR, sample_name, vsearch_executable, output_dir = args
    merged_output_path = os.path.join(output_dir, f"{sample_name}_merged.fastq")
    try:
        # NEW: hard-sync pairs to avoid VSEARCH count/order errors
        syncF, syncR = sync_pairs_to_temp(fnF, fnR, output_dir, sample_name)

        merge_cmd = [
    vsearch_executable, '--fastq_mergepairs', syncF, '--reverse', syncR,
    '--fastq_minovlen', '20',
    '--fastq_maxdiffs', '10',
    '--fastq_minmergelen', '390',
    '--fastq_maxmergelen', '480',
    '--fastqout', merged_output_path
]
        subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully merged reads for sample: {sample_name}")

        # clean synced intermediates
        try:
            os.remove(syncF); os.remove(syncR)
        except Exception:
            pass

        return merged_output_path
    except subprocess.CalledProcessError as e:
        logger.error(f"VSEARCH merge failed for {sample_name}. Stderr: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Pair sync or merge error for {sample_name}: {e}")
        return None

def _fastq_to_fasta_with_label(in_fastq, out_fasta, sample_name):
    with open(in_fastq, 'rt') as fh_in, open(out_fasta, 'w') as fh_out:
        for rec in SeqIO.parse(fh_in, 'fastq'):
            rec.id = f"{rec.id};barcodelabel={sample_name}"
            rec.description = ""  # keep header clean
            SeqIO.write(rec, fh_out, 'fasta')
def calculate_distance_pair_worker(args):
    # This is a placeholder as it's not used in the main analysis flow,
    # but was in the original imports.
    logger.warning("`calculate_distance_pair_worker` is a placeholder and was not called.")
    return None

# --- MAIN PIPELINE STEPS ---

def _filter_and_trim_worker(args):
    fnF, fnR, sample_name, output_dir, trunc_len_f, trunc_len_r, max_ee = args
    filt_path = os.path.join(output_dir, "filtered_sequences")
    os.makedirs(filt_path, exist_ok=True)

    vsearch_executable = r"C:\Users\denisova\Desktop\Renamed files\vsearch\vsearch-2.30.0-win-x86_64\bin\vsearch.exe"
    truncF = os.path.join(filt_path, f"{sample_name}_F_trunc.fastq")
    truncR = os.path.join(filt_path, f"{sample_name}_R_trunc.fastq")

    try:
        # Truncate only
        cmd_F = [vsearch_executable, '--fastq_filter', fnF,
                 '--fastq_trunclen', str(trunc_len_f),
                 '--fastq_qmax', '41', '--fastqout', truncF]
        cmd_R = [vsearch_executable, '--fastq_filter', fnR,
                 '--fastq_trunclen', str(trunc_len_r),
                 '--fastq_qmax', '41', '--fastqout', truncR]
        subprocess.run(cmd_F, check=True, capture_output=True, text=True)
        subprocess.run(cmd_R, check=True, capture_output=True, text=True)

        # Pre-merge per-read EE (DADA2-like)
        filtF = truncF.replace("_trunc.fastq", "_trunc_filt.fastq")
        filtR = truncR.replace("_trunc.fastq", "_trunc_filt.fastq")
        cmd_F2 = [vsearch_executable, '--fastq_filter', truncF,
                  '--fastq_maxee', '2.2', '--fastq_qmax', '41',
                  '--fastqout', filtF]
        cmd_R2 = [vsearch_executable, '--fastq_filter', truncR,
                  '--fastq_maxee', '3.8', '--fastq_qmax', '41',
                  '--fastqout', filtR]
        subprocess.run(cmd_F2, check=True, capture_output=True, text=True)
        subprocess.run(cmd_R2, check=True, capture_output=True, text=True)

        logger.info(f"[{sample_name}] Truncation+preEE OK → {os.path.basename(filtF)}, {os.path.basename(filtR)}")
        return filtF, filtR

    except subprocess.CalledProcessError as e:
        logger.error(f"Trunc/EE failed for {sample_name}. Stderr: {e.stderr}")
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

def _open_auto(path, mode='rt'):
    # Optional: lets you pass .fastq or .fastq.gz transparently
    return gzip.open(path, mode) if path.endswith('.gz') else open(path, mode)


def _count_fasta_records(path: str) -> int:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return 0
    return sum(1 for _ in SeqIO.parse(path, 'fasta'))


def denoise_and_create_asv_table_vsearch(filtFs, filtRs, sample_names, output_dir):
    vsearch_executable = r"C:\Users\denisova\Desktop\Renamed files\vsearch\vsearch-2.30.0-win-x86_64\bin\vsearch.exe"

    merged_fasta   = os.path.join(output_dir, 'all_samples_merged.fa')
    derep_fasta    = os.path.join(output_dir, 'all_samples_derep.fa')
    sorted_derep   = os.path.join(output_dir, 'all_samples_derep_sorted.fa')
    denoised_fa    = os.path.join(output_dir, 'asvs.fa')
    asv_table_tsv  = os.path.join(output_dir, 'asv_table.tsv')

    # (A) Pre-merge diagnostics
    for fF, fR, s in zip(filtFs, filtRs, sample_names):
        with _open_auto(fF, 'rt') as fhF, _open_auto(fR, 'rt') as fhR:
            nF = sum(1 for _ in SeqIO.parse(fhF, 'fastq'))
            nR = sum(1 for _ in SeqIO.parse(fhR, 'fastq'))
        if nF != nR:
            logger.warning(f"[{s}] Pre-merge counts differ: F={nF}, R={nR}. Sync will fix this.")

    try:
        # (B) Merge (parallel)
        logger.info("Step 1/5: Merging paired-end reads in parallel...")
        args_list = [(fF, fR, s_name, vsearch_executable, output_dir)
                     for fF, fR, s_name in zip(filtFs, filtRs, sample_names)]
        with multiprocessing.Pool(processes=CPU_CORES) as pool:
            merged_fastq_files = pool.map(merge_worker, args_list)

        merged_fastq = [(p, s) for p, s in zip(merged_fastq_files, sample_names) if p and os.path.exists(p)]
        if not merged_fastq:
            raise RuntimeError("No merged read files were produced. Relax merge params or check input pairing.")

        # (C) Post-merge EE filter → per-sample FASTA
        per_sample_fastas = []
        total_queries = 0
        for fq_file, s_name in merged_fastq:
            fq_filt = os.path.join(output_dir, f"{s_name}_merged_filt.fastq")
            qf_cmd = [
                vsearch_executable, '--fastq_filter', fq_file,
                '--fastq_maxee', '5.0',
                '--fastq_minlen', '390',
                '--fastq_maxlen', '480',
                '--fastq_qmax', '41',
                '--fastqout', fq_filt
            ]
            subprocess.run(qf_cmd, check=True, capture_output=True, text=True)

            kept_fastq = sum(1 for _ in SeqIO.parse(_open_auto(fq_filt, 'rt'), 'fastq'))
            logger.info(f"[{s_name}] Post-merge EE kept {kept_fastq} sequences.")

            out_fa = os.path.join(output_dir, f"{s_name}_merged_labeled.fa")
            _fastq_to_fasta_with_label(fq_filt, out_fa, s_name)

            kept_fa = _count_fasta_records(out_fa)
            total_queries += kept_fa
            if kept_fa > 0:
                per_sample_fastas.append(out_fa)

            # cleanup intermediates
            try:
                os.remove(fq_file)
                os.remove(fq_filt)
            except Exception:
                pass

        if not per_sample_fastas or total_queries == 0:
            raise RuntimeError("All merged reads were removed by post-merge EE filtering. "
                               "Relax --fastq_maxee (e.g., 3.0–5.0) or merge params.")

        # (D) Concatenate all per-sample FASTAs
        with open(merged_fasta, 'w') as out_all:
            for fa in per_sample_fastas:
                with open(fa, 'r') as f_in:
                    out_all.write(f_in.read())
                try:
                    os.remove(fa)
                except Exception:
                    pass

        q_total = _count_fasta_records(merged_fasta)
        logger.info(f"Total query sequences (merged_fasta): {q_total}")

        # (E) Dereplicate → sortbysize
        logger.info("Step 2/5: Dereplicating all merged sequences...")
        derep_cmd = [
            vsearch_executable, '--derep_fulllength', merged_fasta,
            '--output', derep_fasta, '--sizeout',
            '--threads', str(CPU_CORES)
        ]
        subprocess.run(derep_cmd, check=True, capture_output=True, text=True)

        sort_cmd = [
            vsearch_executable, '--sortbysize', derep_fasta,
            '--output', sorted_derep,
            '--threads', str(CPU_CORES)
        ]
        subprocess.run(sort_cmd, check=True, capture_output=True, text=True)

        # (F) UNOISE
        logger.info("Step 3/5: Denoising sequences with cluster_unoise...")
        unoise_cmd = [
            vsearch_executable, '--cluster_unoise', sorted_derep,
            '--sizein', '--minsize', '3', '--unoise_alpha', '1.3', '--centroids', denoised_fa,
            '--relabel', 'ASV_', '--threads', str(CPU_CORES)
        ]
        subprocess.run(unoise_cmd, check=True, capture_output=True, text=True)

        pre_nochim = _count_fasta_records(denoised_fa)
        logger.info(f"UNOISE centroids (pre-chimera): {pre_nochim}")
        if pre_nochim == 0:
            raise RuntimeError("UNOISE produced 0 centroids. Relax --minsize and/or EE.")

        # (G) Chimera removal (fallback if all removed)
        nochim_fa = os.path.join(output_dir, 'asvs_nochim.fa')
        uchime_cmd = [
            vsearch_executable, '--uchime3_denovo', denoised_fa,
            '--nonchimeras', nochim_fa,
            '--threads', str(CPU_CORES)
        ]
        subprocess.run(uchime_cmd, check=True, capture_output=True, text=True)

        post_nochim = _count_fasta_records(nochim_fa)
        logger.info(f"ASVs after chimera removal: {post_nochim}")
        if post_nochim == 0:
            logger.warning("All ASVs removed by de novo chimera. Falling back to pre-chimera centroids for mapping.")
            asv_centroids_fa = denoised_fa
        else:
            asv_centroids_fa = nochim_fa

        # (H) Map reads → OTU table
        logger.info("Step 4/5: Mapping reads to ASVs to generate feature table...")
        map_cmd = [
            vsearch_executable, '--search_exact', merged_fasta,
            '--db', asv_centroids_fa, '--otutabout', asv_table_tsv,
            '--threads', str(CPU_CORES)
        ]
        subprocess.run(map_cmd, check=True, capture_output=True, text=True)

        if not os.path.exists(asv_table_tsv) or os.path.getsize(asv_table_tsv) == 0:
            raise RuntimeError("vsearch produced an empty OTU table (no matches). Try relaxing filters/UNOISE.")

        # (I) Load and relabel columns with sequences
        logger.info("Step 5/5: Formatting ASV table...")
        asv_table = pd.read_csv(asv_table_tsv, sep='\t', index_col=0, engine='python').T
        if asv_table.shape[1] == 0:
            raise RuntimeError("OTU table has zero ASV columns after transpose. No reads matched centroids.")

        asv_id_to_seq = {
            rec.id.split(';')[0]: str(rec.seq)
            for rec in SeqIO.parse(_open_auto(asv_centroids_fa, 'rt'), 'fasta')
        }
        asv_table.columns = [asv_id_to_seq.get(col_id, col_id) for col_id in asv_table.columns]

               
        final_asv_path = os.path.join(output_dir, 'microbiome_ai_16s_asv.csv')
        asv_table.to_csv(final_asv_path, sep='\t')
        logger.info(f"Final ASV table ({asv_table.shape}) saved to {final_asv_path}")
        return asv_table

    except subprocess.CalledProcessError as e:
        logger.error(f"A VSEARCH command failed. Stderr: {e.stderr}. Ensure 'vsearch' is in PATH or update the path.", exc_info=False)
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

    vsearch_executable = r"C:\Users\denisova\Desktop\Renamed files\vsearch\vsearch-2.30.0-win-x86_64\bin\vsearch.exe"
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

