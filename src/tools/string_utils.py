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
    
def reformat_objective_facts(data):
    """
    Reformat objective facts from a dictionary into a structured string format.

    Takes a dictionary containing 'objective-facts' (a list of fact strings)
    and formats them into a numbered, XML-tagged string suitable for prompts
    or downstream processing.

    Args:
        data (dict): Dictionary with key 'objective-facts' containing a list of fact strings
                    Example: {"objective-facts": ["fact1", "fact2", "fact3"]}

    Returns:
        str: Formatted string with numbered facts wrapped in XML tags

    Example:
        Input: {"objective-facts": ["The sky is blue", "Water is wet"]}
        Output:
            "Objective Facts:
            1. <detailed-desc>The sky is blue</detailed-desc>
            2. <detailed-desc>Water is wet</detailed-desc>\n"
    """
    # Initialize result dictionary to store formatted facts
    result = {"Objective Facts": []}

    # Reformat each objective fact with numbering and XML tags
    for idx, fact in enumerate(data['objective-facts'], start=1):
        # Format: "1. <detailed-desc>fact text</detailed-desc>"
        result["Objective Facts"].append(
            f"{idx}. <detailed-desc>{fact}</detailed-desc>"
        )

    # Convert the dictionary to a formatted string
    result_str = ""
    for key, values in result.items():
        # Add the header "Objective Facts:"
        result_str += f"{key}:\n"
        # Join all facts with newlines and add a final newline
        result_str += "\n".join(values) + "\n"

    return result_str

def convert_set_to_list(obj):
    """
    Recursively convert sets to lists in a nested structure.

    This is necessary because JSON format doesn't support Python sets,
    so we need to convert them to lists before serialization.

    Args:
        obj: Any Python object (set, dict, list, or primitive type)

    Returns:
        The same object structure with all sets converted to lists

    Example:
        Input: {"items": {1, 2, 3}, "nested": {"vals": {4, 5}}}
        Output: {"items": [1, 2, 3], "nested": {"vals": [4, 5]}}
    """
    # Base case: if object is a set, convert to list
    if isinstance(obj, set):
        return list(obj)

    # Recursive case: if dictionary, process all key-value pairs
    elif isinstance(obj, dict):
        return {key: convert_set_to_list(value) for key, value in obj.items()}

    # Recursive case: if list, process all elements
    elif isinstance(obj, list):
        return [convert_set_to_list(item) for item in obj]

    # Base case: for primitives (int, str, bool, etc.), return as-is
    else:
        return obj