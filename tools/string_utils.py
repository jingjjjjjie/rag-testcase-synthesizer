from pathlib import Path
import PyPDF2 

def extract_text_from_pdf(pdf_path) -> str:
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}") from e
    return text

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()
