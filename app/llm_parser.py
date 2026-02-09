# LLM Parsing Layer - Local Hugging Face Model
# Extracts structured invoice data from OCR text

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from app.config import MODEL_NAME, DEVICE, MAX_NEW_TOKENS
from app.schemas import InvoiceData

# =============================================================================
# Prompt Templates
# =============================================================================
T5_PROMPT = """Extract invoice fields as JSON: invoice_number, invoice_date, vendor_name, vendor_gst, customer_name, subtotal, tax, total_amount, currency, payment_terms

Text: {text}

JSON:"""

INSTRUCT_PROMPT = """Extract the following invoice fields from the text below. Return ONLY a valid JSON object with no explanation.

Fields: invoice_number, invoice_date, vendor_name, vendor_gst, customer_name, subtotal, tax, total_amount, currency, payment_terms

If a field is missing, use empty string "".

Invoice Text:
{text}

Return ONLY the JSON object:"""

# =============================================================================
# Model Loading (Singleton)
# =============================================================================
_model = None
_tokenizer = None
_is_causal = False


def _load_model():
    """Load model and tokenizer (called once at startup)."""
    global _model, _tokenizer, _is_causal
    
    if _model is None:
        print(f"Loading model: {MODEL_NAME} on {DEVICE}...")
        
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Check if it's a causal LM or seq2seq
        if any(x in MODEL_NAME.lower() for x in ['phi', 'qwen', 'llama', 'mistral', 'gemma']):
            _is_causal = True
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True
            )
        else:
            _is_causal = False
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None
            )
        
        if DEVICE == "cpu":
            _model = _model.to(DEVICE)
        
        # Set pad token if not set
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        
        print(f"Model loaded successfully on {DEVICE} (causal={_is_causal})")
    
    return _model, _tokenizer, _is_causal


def _extract_json_from_response(response: str) -> dict:
    """Extract JSON object from LLM response."""
    print(f"[DEBUG] Raw LLM response: {response[:500]}")
    
    # Try direct JSON parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Find JSON in response
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(response):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(response[start_idx:i+1])
                except json.JSONDecodeError:
                    start_idx = -1
    
    print("[DEBUG] Could not parse JSON from response")
    return {}


def _extract_with_regex(text: str) -> dict:
    """Fallback: Extract invoice fields using regex patterns."""
    result = {}
    
    # Invoice number
    inv_match = re.search(r'\b(INV[O0-9]+)\b', text, re.IGNORECASE)
    if inv_match:
        result['invoice_number'] = inv_match.group(1)
    
    # Date
    date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})', text, re.IGNORECASE)
    if date_match:
        result['invoice_date'] = date_match.group(1)
    
    # Total amount
    total_match = re.search(r'(?:BALANCE DUE|TOTAL)[:\s]*(?:INR|Rs\.?|%|₹)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if total_match:
        result['total_amount'] = total_match.group(1).replace(',', '')
    
    # Subtotal
    sub_match = re.search(r'SUBTOTAL[:\s]*(?:INR|Rs\.?|%|₹)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if sub_match:
        result['subtotal'] = sub_match.group(1).replace(',', '')
    
    # Tax
    tax_match = re.search(r'TAX\s*\([^)]*\)[:\s]*(?:inc|%)?\s*([\d,]+\.?\d*)', text, re.IGNORECASE)
    if tax_match:
        result['tax'] = tax_match.group(1).replace(',', '')
    
    # Customer
    cust_match = re.search(r'BILL TO[:\s]*\n*([A-Za-z]+)', text, re.IGNORECASE)
    if cust_match:
        result['customer_name'] = cust_match.group(1)
    
    # Payment terms
    pay_match = re.search(r'(?:DUE|Due)[:\s]*\n*([A-Za-z\s]+?)(?:\n|$)', text, re.IGNORECASE)
    if pay_match:
        result['payment_terms'] = pay_match.group(1).strip()
    
    # Currency
    if re.search(r'INR|Rs\.?|₹', text):
        result['currency'] = 'INR'
    elif re.search(r'\$|USD', text):
        result['currency'] = 'USD'
    
    print(f"[DEBUG] Regex extracted: {result}")
    return result


def parse_invoice_text(ocr_text: str) -> InvoiceData:
    """Parse OCR text and extract structured invoice data."""
    model, tokenizer, is_causal = _load_model()
    
    # Clean and truncate
    ocr_text = ocr_text.strip()[:800]
    print(f"[DEBUG] OCR text length: {len(ocr_text)}")
    
    # Choose prompt based on model type
    prompt = INSTRUCT_PROMPT.format(text=ocr_text) if is_causal else T5_PROMPT.format(text=ocr_text)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    if DEVICE == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    try:
        with torch.no_grad():
            if is_causal:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                # For causal models, only decode the new tokens
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}")
        response = ""
    
    # Parse response
    extracted = _extract_json_from_response(response)
    
    # Fallback to regex
    if not extracted:
        print("[DEBUG] Using regex fallback")
        extracted = _extract_with_regex(ocr_text)
    
    return InvoiceData(
        invoice_number=str(extracted.get("invoice_number", "")),
        invoice_date=str(extracted.get("invoice_date", "")),
        vendor_name=str(extracted.get("vendor_name", "")),
        vendor_gst=str(extracted.get("vendor_gst", "")),
        customer_name=str(extracted.get("customer_name", "")),
        subtotal=str(extracted.get("subtotal", "")),
        tax=str(extracted.get("tax", "")),
        total_amount=str(extracted.get("total_amount", "")),
        currency=str(extracted.get("currency", "")),
        payment_terms=str(extracted.get("payment_terms", ""))
    )


def init_model():
    """Initialize model at startup."""
    _load_model()
