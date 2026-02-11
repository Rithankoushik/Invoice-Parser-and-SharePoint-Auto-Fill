# Invoice Parser + SharePoint Auto-Fill

A production-grade, local-first Invoice Parsing System that extracts structured data from invoices using OCR and LLM, then auto-fills Microsoft SharePoint Lists.

## Features

- **OCR Layer**: Tesseract + OpenCV preprocessing (grayscale, adaptive thresholding)
- **LLM Parsing**: Local Hugging Face model (google/flan-t5-base) with GPU acceleration
- **SharePoint Integration**: Auto-fill SharePoint Lists via Microsoft Graph API
- **REST API**: FastAPI endpoint for easy integration
- **Windows + NVIDIA GPU**: Optimized for RTX 3050 (4GB VRAM)

## Prerequisites

### 1. Install Tesseract OCR
Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Install Poppler (for PDF support)
Download from: https://github.com/osborn/wnd/releases

Extract and add `bin` folder to PATH, or set `POPPLER_PATH` in `.env`

### 3. Python 3.11+
```bash
python --version  # Should be 3.11+
```

## Installation

```bash
# Clone/navigate to project
cd d:\invoice_parser

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
copy .env.example .env
# Edit .env with your Azure/SharePoint credentials
```

## Azure App Registration (for SharePoint)

1. Go to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Click **New registration**
3. Name: `invoice-parser-app`
4. Supported account types: Single tenant
5. Click **Register**

### Get Credentials
- **Client ID**: Overview → Application (client) ID
- **Tenant ID**: Overview → Directory (tenant) ID
- **Client Secret**: Certificates & secrets → New client secret

### Add API Permissions
1. API permissions → Add a permission
2. Microsoft Graph → Application permissions
3. Add: `Sites.ReadWrite.All`
4. Click **Grant admin consent**

### Get SharePoint IDs

**Site ID:**
```
GET https://graph.microsoft.com/v1.0/sites/{your-tenant}.sharepoint.com:/sites/{site-name}
```

**List ID:**
```
GET https://graph.microsoft.com/v1.0/sites/{site-id}/lists
```

## SharePoint List Setup

Create a SharePoint List with these columns (Single line of text):

| Column Name | Type |
|-------------|------|
| InvoiceNumber | Single line of text |
| InvoiceDate | Single line of text |
| VendorName | Single line of text |
| VendorGST | Single line of text |
| CustomerName | Single line of text |
| Subtotal | Single line of text |
| Tax | Single line of text |
| TotalAmount | Single line of text |
| Currency | Single line of text |
| PaymentTerms | Single line of text |

## Running the Application

This system consists of a backend (FastAPI) and a frontend (Streamlit).

### 1. Start the FastAPI Server
```bash
# Activate virtual environment
.venv\Scripts\activate

# Start server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Wait for the console to show "Model loaded. Ready to process invoices."

### 2. Start the Streamlit UI
Open a **new terminal** window:
```bash
# Navigate to project and activate venv
cd d:\invoice_parser
.venv\Scripts\activate

# Start UI
streamlit run app/ui.py
```
The browser will automatically open at http://localhost:8501

## API Usage

### Parse Invoice
```bash
curl -X POST "http://localhost:8000/parse-invoice" \
  -H "accept: application/json" \
  -F "file=@path/to/invoice.pdf"
```

### Response Format
```json
{
  "parsed_data": {
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "vendor_name": "ABC Corp",
    "vendor_gst": "29ABCDE1234F1Z5",
    "customer_name": "XYZ Ltd",
    "subtotal": "10000.00",
    "tax": "1800.00",
    "total_amount": "11800.00",
    "currency": "INR",
    "payment_terms": "Net 30"
  },
  "sharepoint_status": "success",
  "error": ""
}
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Project Structure

```
invoice_parser/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── ocr.py           # OCR processing
│   ├── llm_parser.py    # LLM parsing
│   ├── sharepoint.py    # SharePoint integration
│   ├── schemas.py       # Pydantic models
│   ├── config.py        # Configuration
│   └── utils.py         # Utilities
├── temp/                # Uploaded files (auto-cleaned)
├── requirements.txt
├── .env.example
└── README.md
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_TENANT_ID` | Azure AD Tenant ID |
| `AZURE_CLIENT_ID` | Azure App Client ID |
| `AZURE_CLIENT_SECRET` | Azure App Secret |
| `SHAREPOINT_SITE_ID` | SharePoint Site ID |
| `SHAREPOINT_LIST_ID` | SharePoint List ID |
| `MODEL_NAME` | HuggingFace model (default: google/flan-t5-base) |
| `TESSERACT_CMD` | Path to Tesseract executable |
| `POPPLER_PATH` | Path to Poppler bin folder |

## Troubleshooting

### OCR not working
- Verify Tesseract is installed: `tesseract --version`
- Check `TESSERACT_CMD` path in `.env`

### PDF conversion fails
- Install Poppler and add to PATH
- Or set `POPPLER_PATH` in `.env`

### CUDA not detected
- Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### SharePoint auth fails
- Verify admin consent is granted
- Check all credentials in `.env`

## License

MIT
