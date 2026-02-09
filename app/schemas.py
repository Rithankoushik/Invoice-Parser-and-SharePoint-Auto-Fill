# Pydantic Schemas for Invoice Parser

from pydantic import BaseModel, Field
from typing import Optional


class InvoiceData(BaseModel):
    """Structured invoice data extracted by LLM."""
    invoice_number: str = Field(default="", description="Invoice number/ID")
    invoice_date: str = Field(default="", description="Invoice date")
    vendor_name: str = Field(default="", description="Vendor/Seller name")
    vendor_gst: str = Field(default="", description="Vendor GST/Tax ID")
    customer_name: str = Field(default="", description="Customer/Buyer name")
    subtotal: str = Field(default="", description="Subtotal before tax")
    tax: str = Field(default="", description="Tax amount")
    total_amount: str = Field(default="", description="Total invoice amount")
    currency: str = Field(default="", description="Currency code (INR, USD, etc.)")
    payment_terms: str = Field(default="", description="Payment terms/due date")


class ParseResponse(BaseModel):
    """API response for invoice parsing."""
    parsed_data: InvoiceData
    sharepoint_status: str = Field(
        default="skipped",
        description="Status: success, failed, or skipped"
    )
    error: str = Field(default="", description="Error message if any")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_code: Optional[str] = None
