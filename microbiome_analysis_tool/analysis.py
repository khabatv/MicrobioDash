
import pandas as pd
from skbio.diversity import alpha_diversity, beta_diversity
from skbio.stats.ordination import pcoa
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .config import logger
from .utils import run_diagnostic_checks

def create_phyloseq_object(seqtab, taxa, metadata):
    try:
        asv, tax, meta = seqtab.T, taxa, metadata
        is_contaminant = tax['Order'].str.contains('Chloroplast', na=False) | tax['Family'].str.contains('Mitochondria', na=False)
        tax_filtered = tax[~is_contaminant]
        asvs_to_keep = tax_filtered.index
        asv_filtered = asv.loc[asvs_to_keep]

        # Ensure samples align
        shared_samples = list(set(asv_filtered.columns) & set(meta.index))
        asv_aligned = asv_filtered[shared_samples]
        meta_aligned = meta.loc[shared_samples]

        ps1 = {'asv': asv_aligned, 'tax': tax_filtered, 'meta': meta_aligned}

        run_diagnostic_checks(ps1, "Post-Filtering")
        logger.info("Created phyloseq-like object successfully.")
        return ps1
    except Exception as e:
        logger.error(f"Error creating phyloseq object: {e}", exc_info=True)
        return None

def calculate_alpha_diversity(ps1, treatment):
    try:
        logger.info("Calculating Alpha Diversity...")
        meta, asv_table = ps1['meta'].copy(), ps1['asv']
        asv_table_transposed = asv_table.T
        if asv_table_transposed.empty:
            logger.warning("ASV table is empty. Cannot calculate alpha diversity.")
            meta['Shannon'] = 0.0
            return meta

        shannon_values = alpha_diversity('shannon', asv_table_transposed.astype(int).values, ids=asv_table_transposed.index)
        meta['Shannon'] = shannon_values
        logger.info("Alpha diversity calculation successful.")
        return meta
    except Exception as e:
        logger.error(f"Error in calculate_alpha_diversity: {e}", exc_info=True)
        ps1['meta']['Shannon'] = 0.0
        return ps1['meta']

def calculate_beta_diversity(ps1):
    try:
        asv = ps1['asv'] # ASVs x Samples
        asv_sums = asv.sum(axis=1) # Sum reads for each ASV
        asvs_to_keep = asv_sums[asv_sums > 0].index
        asv_filtered = asv.loc[asvs_to_keep]

        # Normalize by sample total (column sum) to get relative abundance
        asv_rel = asv_filtered.div(asv_filtered.sum(axis=0), axis=1)
        logger.info("Calculated beta diversity (relative abundance).")
        return asv_rel, ps1['meta']
    except Exception as e:
        logger.error(f"Error calculating beta diversity: {e}", exc_info=True)
        return None, None

def perform_pcoa(asv, meta, treatment):
    try:
        asv_transposed = asv.T
        dm = beta_diversity('braycurtis', asv_transposed.to_numpy(), ids=asv_transposed.index)
        ordination_result = pcoa(dm)
        scores = ordination_result.samples.join(meta[[treatment]])
        variance_explained = ordination_result.proportion_explained
        logger.info("Performed PCoA successfully.")
        return scores, dm, variance_explained
    except Exception as e:
        logger.error(f"Error performing PCoA: {e}", exc_info=True)
        return None, None, None

def perform_pca(asv, top_n):
    try:
        top_asvs = asv.sum(axis=1).nlargest(top_n).index
        asv_top_transposed = asv.loc[top_asvs].T

        scaler = StandardScaler()
        asv_scaled = scaler.fit_transform(asv_top_transposed)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(asv_scaled)
        explained_variance = pca.explained_variance_ratio_
        logger.info("Performed PCA successfully.")
        return pca_result, explained_variance
    except Exception as e:
        logger.error(f"Error performing PCA: {e}", exc_info=True)
        return None, None

def perform_nmds(distance_matrix):
    try:
        logger.info("Performing NMDS...")
        dm_array = distance_matrix.to_data_frame().values
        nmds_model = MDS(n_components=2, metric=False, n_init=20, dissimilarity='precomputed', random_state=42, max_iter=500)
        nmds_scores = nmds_model.fit_transform(dm_array)
        nmds_stress = nmds_model.stress_
        logger.info(f"NMDS completed with stress: {nmds_stress:.4f}")
        return pd.DataFrame(nmds_scores, index=distance_matrix.ids, columns=['NMDS1', 'NMDS2']), nmds_stress
    except Exception as e:
        logger.error(f"Error performing NMDS: {e}", exc_info=True)
        return None, None
