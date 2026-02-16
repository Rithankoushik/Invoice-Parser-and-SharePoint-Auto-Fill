"""
Vision Model Core — Invoice text extraction using Groq LLaMA Vision.

Converts PDF/image invoices → raw text (via vision model) → structured JSON (via LLM).
Based on the logic from invoice-optimised.ipynb.
"""

import json
import os
import base64
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

# ── Load .env from the project root ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Models ───────────────────────────────────────────────────────────────────
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
JSON_MODEL = "llama-3.1-8b-instant"


def get_groq_client() -> Groq:
    """Create and return a Groq client using the API key from .env."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Check your .env file.")
    return Groq(api_key=api_key)


# ── Image helpers ────────────────────────────────────────────────────────────

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF bytes into a list of PIL images (one per page)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a JPEG base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Groq vision extraction ──────────────────────────────────────────────────

VISION_PROMPT = """
Extract all readable text from this document.
Preserve layout and line breaks.
Do NOT summarize.
Return only raw extracted text.
"""


def extract_text_with_vision(client: Groq, image: Image.Image) -> str:
    """Use the Groq LLaMA vision model to OCR a single image."""
    b64 = image_to_base64(image)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ── Structured JSON extraction ───────────────────────────────────────────────

INVOICE_JSON_PROMPT = """
Extract structured invoice data from the text below. Do not explain or hallucinate.

Use semantic reasoning to match fields even if formatting is inconsistent.

Return exactly the following JSON structure with values or null if missing:

{{
"Vendor":{{"BusinessName":null,"Address":null,"GSTIN":null,"PAN":null,"Phone":null,"Email":null,"CIN":null}},
"Buyer":{{"Name":null,"BillingAddress":null,"ShippingAddress":null,"GSTIN":null,"Phone":null,"Email":null}},
"Items":[{{"Description":null,"Quantity":null,"Unit":null,"RatePerUnit":null,"Discount":null,"TaxableValue":null,
"GSTRatePercent":null,"CGSTAmount":null,"SGSTAmount":null,"IGSTAmount":null,"Cess":null,"TotalItemAmount":null}}],
"Totals":{{"Subtotal":null,"TotalTaxableValue":null,"TotalCGST":null,"TotalSGST":null,"TotalIGST":null,"TotalCess":null,
"RoundOff":null,"GrandTotal":null,"AmountInWords":null}},
"PaymentDetails":{{"ModeOfPayment":null,"UPIID":null,"BankName":null,"AccountNumber":null,"IFSCCode":null,
"TransactionReferenceID":null}}
}}

TEXT:
{text}
"""


def extract_invoice_fields(client: Groq, text: str) -> str:
    """Call the LLM to extract structured JSON from raw invoice text."""
    prompt = INVOICE_JSON_PROMPT.format(text=text)

    response = client.chat.completions.create(
        model=JSON_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def process_llm_output(llm_output: str, threshold: float = 0.6) -> dict | None:
    """Parse the LLM JSON output and sanitise low-confidence fields."""
    # Strip markdown code fences if present
    cleaned = llm_output.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    def sanitize(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict) and "confidence" in v:
                    if v["confidence"] < threshold:
                        v["value"] = None
                else:
                    sanitize(v)
        elif isinstance(obj, list):
            for item in obj:
                sanitize(item)

    sanitize(data)
    return data


# ── End-to-end pipeline ─────────────────────────────────────────────────────

def process_document(file_bytes: bytes, filename: str) -> tuple[str, dict | None]:
    """
    Full pipeline: file bytes → (raw_text, structured_json).

    Parameters
    ----------
    file_bytes : bytes
        Raw file content (PDF or image).
    filename : str
        Original filename, used to detect file type.

    Returns
    -------
    raw_text : str
        The text extracted by the vision model.
    structured_data : dict | None
        Parsed invoice JSON, or None if parsing failed.
    """
    client = get_groq_client()
    full_text = ""

    if filename.lower().endswith(".pdf"):
        images = pdf_bytes_to_images(file_bytes)
        for img in images:
            full_text += extract_text_with_vision(client, img) + "\n"
    else:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        full_text = extract_text_with_vision(client, image)

    llm_output = extract_invoice_fields(client, full_text)
    structured_data = process_llm_output(llm_output)

    return full_text.strip(), structured_data
