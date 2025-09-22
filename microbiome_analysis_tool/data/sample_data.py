
import pandas as pd
import numpy as np

# Sample Metadata: Describes the samples.
sample_metadata_df = pd.DataFrame({
    'Treatment': ['Control', 'Stress1', 'Stress2'],
    'Substrate': ['Cellulose', 'Cellulose', 'Lignin']
}, index=pd.Index(['Sample1', 'Sample2', 'Sample3'], name='SampleID'))

# Sample ASV Table (Feature Table): Shows counts of each ASV in each sample.
# Columns are DNA sequences (ASVs), Index is Sample IDs.
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
sample_seqtab.index.name = 'SampleID'

# Sample Taxonomy Table: Provides taxonomic classification for each ASV.
# Index is the DNA sequence (ASV).
sample_taxa = pd.DataFrame(
    [
        ['Bacteria', 'Proteobacteria', 'Gammaproteobacteria', 'Enterobacteriales', 'Enterobacteriaceae', 'Escherichia', np.nan],
        ['Bacteria', 'Firmicutes', 'Bacilli', 'Lactobacillales', 'Lactobacillaceae', 'Lactobacillus', np.nan],
        ['Bacteria', 'Actinobacteria', 'Actinomycetia', 'Streptomycetales', 'Streptomycetaceae', 'Streptomyces', np.nan],
        ['Bacteria', 'Bacteroidetes', 'Bacteroidia', 'Bacteroidales', 'Bacteroidaceae', 'Bacteroides', np.nan]
    ],
    index=sample_seqtab.columns,
    columns=['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus','Species']
)
sample_taxa.index.name = 'ASV'

# Sample filenames to simulate file discovery.
sample_filenames = ['Sample1_R1.fastq', 'Sample2_R1.fastq', 'Sample3_R1.fastq', 'Sample1_R2.fastq', 'Sample2_R2.fastq', 'Sample3_R2.fastq']
