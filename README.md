
# Microbiome Analysis Dashboard

This is a comprehensive tool for 16S rRNA microbiome data analysis, from raw FASTQ files to advanced statistical modeling and reporting, featuring an interactive Dash-based user interface.

## Features

-   End-to-end pipeline: FASTQ quality filtering, ASV table generation (via VSEARCH), and taxonomy assignment.
-   Core microbiome metrics: Alpha and Beta diversity, PCoA, PCA, and NMDS.
-   Advanced statistics: PERMANOVA, differential abundance testing, indicator species analysis, and mixed-effect models (GEE & Bayesian ZINB).
-   AI-powered assistance for parameter suggestion and results interpretation (requires Google Gemini API key).
-   Automated PDF report generation.
-   Interactive data visualization with Plotly and Dash.

## Installation

1.  Clone or download this repository.
2.  It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4.  You will also need to have `vsearch` installed and accessible in your system's PATH, or you must update the path to the executable in `microbiome_analysis_tool/pipeline_steps.py`.

## Usage

1.  **Prepare your data:**
    -   Create a folder (e.g., `my_project_data`).
    -   Place your paired-end FASTQ files (e.g., `Sample1_R1.fastq.gz`, `Sample1_R2.fastq.gz`) in this folder.
    -   Place your metadata file (CSV or TSV format) in the same folder. The first column must be the sample IDs.
    -   If you have a SILVA database for taxonomy assignment, name it `silva.fasta` and place it in this folder.

2.  **Run the application:**
    ```bash
    python run.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:8050`.

4.  **In the dashboard:**
    -   Enter the path to your data folder (e.g., `./my_project_data`).
    -   Click "List Files in Folder" to confirm the files are detected.
    -   Set the analysis parameters manually or use the AI suggestion features.
    -   Click "Run Analysis" to start the pipeline.

## Project Structure

-   `run.py`: The main entry point to start the Dash application.
-   `requirements.txt`: A list of all Python dependencies.
-   `setup.py`: Package setup script.
-   `microbiome_analysis_tool/`: The main package directory.
    -   `app.py`: Contains the Dash app layout, callbacks, and main analysis workflow logic.
    -   `analysis.py`: Functions for core diversity and ordination analyses.
    -   `pipeline_steps.py`: Functions for the bioinformatics pipeline (filtering, denoising, taxonomy).
    -   `statistics.py`: Functions for advanced statistical modeling.
    -   `plotting.py`: Functions for generating complex plots like the phylogenetic tree.
    -   `ai_utils.py`: Functions for interacting with the Gemini AI API.
    -   `reporting.py`: Function for generating the PDF report.
    -   `config.py`: Configuration variables and logging setup.
    -   `utils.py`: General utility and helper functions.
    -   `data/sample_data.py`: Contains pre-packaged sample data for testing.
