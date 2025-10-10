
import os
import logging

# --- Basic Configuration ---

# Set the number of CPU cores to use for parallel processing.
# Leaves one core free for system processes.
CPU_CORES = max(1, os.cpu_count() - 1)

# --- AI Configuration ---

# IMPORTANT: Replace "YOUR_API_KEY_HERE" with your actual Google Gemini API Key.
# You can obtain a key from https://aistudio.google.com/
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE") # Placeholder Key, get from environment

# --- Logging Setup ---

# Configure a logger to provide informative output to the console.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("MicrobiomeTool")
