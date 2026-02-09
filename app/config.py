# Configuration for Invoice Parser

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# Paths
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# =============================================================================
# LLM Configuration
# =============================================================================
MODEL_NAME = os.getenv("MODEL_NAME", "google/flan-t5-base")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

# Device configuration - auto-detect GPU
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Azure / SharePoint Configuration
# =============================================================================
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")

SHAREPOINT_SITE_ID = os.getenv("SHAREPOINT_SITE_ID", "")
SHAREPOINT_LIST_ID = os.getenv("SHAREPOINT_LIST_ID", "")

# Graph API endpoints
GRAPH_API_BASE = "https://graph.microsoft.com/v1.0"

# =============================================================================
# OCR Configuration
# =============================================================================
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH = os.getenv("POPPLER_PATH", None)  # Set if not in PATH

# =============================================================================
# Allowed file types
# =============================================================================
ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
