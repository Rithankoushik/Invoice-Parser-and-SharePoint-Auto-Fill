# OCR Layer - Tesseract + OpenCV
# Handles PDF and image text extraction

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pathlib import Path
from pdf2image import convert_from_path
from typing import List

from app.config import TESSERACT_CMD, POPPLER_PATH

# Set Tesseract path if specified
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def convert_pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of PIL Image objects (one per page)
    """
    kwargs = {}
    if POPPLER_PATH:
        kwargs['poppler_path'] = POPPLER_PATH
    
    images = convert_from_path(str(pdf_path), dpi=300, **kwargs)
    return images


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Steps:
    1. Convert to grayscale
    2. Apply adaptive thresholding
    3. Denoise
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply adaptive thresholding for better text detection
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised


def extract_text_from_image(image: Image.Image) -> str:
    """
    Extract text from a single image using Tesseract.
    
    Args:
        image: PIL Image object
        
    Returns:
        Extracted text string
    """
    # Preprocess the image
    processed = preprocess_image(image)
    
    # Convert back to PIL for pytesseract
    pil_image = Image.fromarray(processed)
    
    # Extract text with optimized config
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    
    return text


def extract_text(file_path: Path) -> str:
    """
    Main OCR function - routes based on file type.
    
    Supports: PDF, JPG, JPEG, PNG
    
    Args:
        file_path: Path to the document
        
    Returns:
        Extracted text from all pages/image
    """
    ext = file_path.suffix.lower()
    all_text = []
    
    if ext == '.pdf':
        # Handle PDF - convert to images first
        images = convert_pdf_to_images(file_path)
        for i, img in enumerate(images):
            page_text = extract_text_from_image(img)
            all_text.append(f"--- Page {i+1} ---\n{page_text}")
    
    elif ext in {'.jpg', '.jpeg', '.png'}:
        # Handle image directly
        image = Image.open(file_path)
        text = extract_text_from_image(image)
        all_text.append(text)
    
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    return '\n\n'.join(all_text)
