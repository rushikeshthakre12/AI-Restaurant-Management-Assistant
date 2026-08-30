"""Generate a downloadable PDF bill for a placed order using fpdf2."""
import sys
from pathlib import Path
from fpdf import FPDF

sys.path.append(str(Path(__file__).resolve().parent.parent))
def format_currency_ascii(amount: float) -> str:
    """Bill PDFs use core (non-Unicode) fonts, which can't render ₹ -- use
    'Rs.' there instead of the ₹ symbol used elsewhere in the app."""
    return f"Rs. {amount:,.2f}"

BASE_DIR = Path(__file__).resolve().parent.parent
BILLS_DIR = BASE_DIR / "data" / "bills"
BILLS_DIR.mkdir(parents=True, exist_ok=True)


def generate_bill_pdf(order_id: int, items: list[dict], bill: dict, customer_name: str = "Customer") -> Path:
    """items: [{'name': str, 'quantity': int, 'price': float}], bill: {'subtotal','tax','discount','total'}"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "RESTAURANT BILL", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Order #{order_id}   Customer: {customer_name}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(90, 8, "Item", border="B")
    pdf.cell(30, 8, "Qty", border="B", align="R")
    pdf.cell(35, 8, "Price", border="B", align="R")
    pdf.cell(35, 8, "Amount", border="B", align="R", ln=True)

    pdf.set_font("Helvetica", "", 11)
    for item in items:
        amount = item["quantity"] * item["price"]
        pdf.cell(90, 8, item["name"])
        pdf.cell(30, 8, str(item["quantity"]), align="R")
        pdf.cell(35, 8, format_currency_ascii(item["price"]), align="R")
        pdf.cell(35, 8, format_currency_ascii(amount), align="R", ln=True)

    pdf.ln(4)
    pdf.set_font("Helvetica", "", 11)
    for label, key in [("Subtotal", "subtotal"), ("Tax", "tax"), ("Discount", "discount")]:
        pdf.cell(155, 8, label, align="R")
        pdf.cell(35, 8, format_currency_ascii(bill[key]), align="R", ln=True)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(155, 10, "TOTAL", align="R")
    pdf.cell(35, 10, format_currency_ascii(bill["total"]), align="R", ln=True)

    output_path = BILLS_DIR / f"bill_order_{order_id}.pdf"
    pdf.output(str(output_path))
    return output_path


if __name__ == "__main__":
    demo_items = [
        {"name": "Paneer Pizza", "quantity": 2, "price": 299.0},
        {"name": "Coke", "quantity": 1, "price": 60.0},
    ]
    demo_bill = {"subtotal": 658.0, "tax": 32.9, "discount": 50.0, "total": 640.9}
    path = generate_bill_pdf(999, demo_items, demo_bill, customer_name="Demo Customer")
    print(f"Generated: {path}  (exists: {path.exists()}, size: {path.stat().st_size} bytes)")
