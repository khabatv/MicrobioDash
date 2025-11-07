
from setuptools import setup, find_packages

setup(
    name='microbiome_analysis_tool',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A comprehensive microbiome analysis dashboard with AI features.',
    packages=find_packages(),
    install_requires=[
        'dash',
        'plotly',
        'scikit-bio',
        'scikit-learn',
        'pandas',
        'numpy',
        'statsmodels',
        'biopython',
        'ete3',
        'google-generativeai',
        'reportlab',
        'matplotlib',
        'seaborn',
        'pymc',
        'arviz',
        'bambi',
        'pytensor',
        'patsy',
        'scikit-posthocs',
        'joblib',
        'gunicorn' # For deployment
    ],
    python_requires='>=3.9',
)
