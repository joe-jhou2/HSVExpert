import logging
import os
from datetime import datetime
import zipfile
import json
import re
import json
import spacy
from collections import defaultdict
from pathlib import Path
from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
from adobe.pdfservices.operation.io.stream_asset import StreamAsset
from adobe.pdfservices.operation.pdf_services import PDFServices
from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
import dotenv

# Load environment variables from .env file if it exists
dotenv.load_dotenv()
ADOBE_CLIENT_ID = os.getenv('ADOBE_CLIENT_ID')
ADOBE_CLIENT_SECRET = os.getenv('ADOBE_CLIENT_SECRET')

# Initialize the logger
logging.basicConfig(level=logging.INFO)

class ExtractTextInfoFromPDF:
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path
        self.output_zip_path = None
      
    def extract_text(self):
        """Extract text from PDF and return structured data"""
        try:
            # Validate input file exists
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            # Read input file
            with open(self.pdf_path, 'rb') as file:
                input_stream = file.read()

            # Validate credentials
            client_id = os.getenv('ADOBE_CLIENT_ID')
            client_secret = os.getenv('ADOBE_CLIENT_SECRET')

            if not client_id or not client_secret:
                raise ValueError("Missing Adobe API credentials. Set client_id and client_secret environment variables.")

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=client_id,
                client_secret=client_secret
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset from source file and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT],
            )

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            # Save the result to output file
            self.output_zip_path = self.create_output_file_path()
            with open(self.output_zip_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

            # Process the extracted data
            return self.process_extracted_data()

        except (ServiceApiException, ServiceUsageException, SdkException) as e:
            logging.exception(f'Adobe API exception: {e}')
            raise
        except Exception as e:
            logging.exception(f'Unexpected error during extraction: {e}')
            raise

    def process_extracted_data(self):
        """Process the extracted ZIP file and return structured data"""
        if not self.output_zip_path or not os.path.exists(self.output_zip_path):
            raise FileNotFoundError("Output zip file not found")
        
        try:
            with zipfile.ZipFile(self.output_zip_path, 'r') as archive:
                # Check if structuredData.json exists in the archive
                if 'structuredData.json' not in archive.namelist():
                    raise FileNotFoundError("structuredData.json not found in the extracted ZIP")
                
                with archive.open('structuredData.json') as jsonentry:
                    jsondata = jsonentry.read()
                    data = json.loads(jsondata)
                    
                return data
                
        except zipfile.BadZipFile:
            logging.error(f"Invalid ZIP file: {self.output_zip_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in structuredData.json: {e}")
            raise

    def extract_headings(self, data=None):
        """Extract H1 headings from the structured data"""
        if data is None:
            data = self.process_extracted_data()
        
        headings = []
        
        if "elements" not in data:
            logging.warning("No 'elements' found in structured data")
            return headings
        
        for element in data["elements"]:
            if element.get("Path", "").endswith("/H1"):
                heading_text = element.get("Text", "").strip()
                if heading_text:
                    headings.append(heading_text)
                    print(f"H1: {heading_text}")
        
        return headings

    def extract_all_text_elements(self, data=None):
        """Extract all text elements with their types"""
        if data is None:
            data = self.process_extracted_data()
        
        text_elements = []
        
        if "elements" not in data:
            logging.warning("No 'elements' found in structured data")
            return text_elements
        
        for element in data["elements"]:
            if "Text" in element and element["Text"].strip():
                text_elements.append({
                    "text": element["Text"].strip(),
                    "path": element.get("Path", ""),
                    "font": element.get("Font", {}),
                    "bounds": element.get("Bounds", {})
                })
        
        return text_elements

    def create_output_file_path(self) -> str:
        """Generates a string containing a directory structure and file name for the output file"""
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
        
        # Use pathlib for better path handling
        output_dir = Path("output/ExtractTextInfoFromPDF")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / f"extract_{time_stamp}.zip")

    def cleanup(self):
        """Clean up temporary files"""
        if self.output_zip_path and os.path.exists(self.output_zip_path):
            try:
                os.remove(self.output_zip_path)
                logging.info(f"Cleaned up temporary file: {self.output_zip_path}")
            except OSError as e:
                logging.warning(f"Could not remove temporary file {self.output_zip_path}: {e}")


SECTION_NORMALIZATION_PATTERNS = [
    ("introduction", "Introduction"),
    ("background", "Introduction"),
    ("methods", "Methods"),
    ("materials and methods", "Methods"),
    ("experimental", "Methods"),
    ("study design", "Methods"),
    ("results", "Results"),
    ("findings", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Discussion"),
    ("summary", "Discussion"),
    ("abstract", "Abstract")
]

def clean_section_title(title):
    """Remove leading numbers like '2.1 ' from section titles."""
    return re.sub(r'^\s*\d+(\.\d+)*\.?\s*', '', title).strip()

def normalize_section(title):
    """Fuzzy match section titles to canonical forms."""
    cleaned = clean_section_title(title)
    lower_title = cleaned.lower()
    for pattern, norm in SECTION_NORMALIZATION_PATTERNS:
        if pattern in lower_title:
            return norm
    return cleaned  # fallback: return cleaned title

# Load spaCy model globally
nlp = spacy.load("en_core_sci_sm")  # or en_core_web_sm if fallback needed

def split_sentences(text):
    """Use spaCy to split into sentences."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_sections_from_pdf_json(json_data):
    """Extract metadata and sentence-split hierarchical sections from Adobe PDF Services JSON. Returns a dict with metadata and structured sections."""
    
    # === METADATA EXTRACTION ===
    metadata = {
        "title": None,
        "authors": [],
        "journal": None,
        "year": None,
        "abstract": []
    }
    
    # Extract basic info from elements
    elements = json_data.get("elements", [])
    
    # Try to find title (usually first H1 or large text)
    title_candidates = []
    for element in elements[:10]:  # Check first 10 elements
        if element.get("Path", "").endswith(("/H1", "/Title")) or \
           (element.get("Font", {}).get("size", 0) > 16):
            text = element.get("Text", "").strip()
            if text and len(text) > 10:  # Reasonable title length
                title_candidates.append(text)
    
    if title_candidates:
        metadata["title"] = title_candidates[0]
    
    # Try to extract year from text patterns
    year_pattern = r'\b(19|20)\d{2}\b'
    for element in elements[:20]:  # Check first 20 elements
        text = element.get("Text", "")
        year_match = re.search(year_pattern, text)
        if year_match:
            metadata["year"] = year_match.group()
            break
    
    # === SECTION EXTRACTION ===
    sections = defaultdict(list)
    current_section = "Unknown"
    current_subsection = None
    
    # Track heading hierarchy
    heading_stack = []
    
    for element in elements:
        path = element.get("Path", "")
        text = element.get("Text", "").strip()
        
        if not text:
            continue
        
        # Identify headings based on path
        if "/H1" in path:
            # Main section heading
            cleaned_title = clean_section_title(text)
            normalized_title = normalize_section(cleaned_title)
            current_section = normalized_title
            heading_stack = [normalized_title]
            current_subsection = None
            
        elif "/H2" in path:
            # Subsection heading
            cleaned_title = clean_section_title(text)
            if heading_stack:
                current_subsection = cleaned_title
                full_title = f"{heading_stack[0]} > {cleaned_title}"
                heading_stack = [heading_stack[0], cleaned_title]
            else:
                current_section = normalize_section(cleaned_title)
                heading_stack = [current_section]
                current_subsection = None
                
        elif "/H3" in path or "/H4" in path or "/H5" in path or "/H6" in path:
            # Lower level headings
            cleaned_title = clean_section_title(text)
            if len(heading_stack) >= 2:
                full_title = f"{heading_stack[0]} > {heading_stack[1]} > {cleaned_title}"
            elif len(heading_stack) == 1:
                full_title = f"{heading_stack[0]} > {cleaned_title}"
            else:
                full_title = cleaned_title
            # Don't change current section, just note the subsection
            
        elif "/P" in path:
            # Regular paragraph text
            if text:
                # Determine which section to add to
                if current_subsection and len(heading_stack) >= 2:
                    section_key = f"{heading_stack[0]} > {current_subsection}"
                else:
                    section_key = current_section
                
                # Split into sentences and add
                sentences = split_sentences(text)
                sections[section_key].extend(sentences)
        
        elif path.endswith("/Text") or "/Span" in path:
            # Other text elements - treat as paragraph if substantial
            if len(text) > 50:  # Only substantial text blocks
                section_key = current_section
                if current_subsection and len(heading_stack) >= 2:
                    section_key = f"{heading_stack[0]} > {current_subsection}"
                
                sentences = split_sentences(text)
                sections[section_key].extend(sentences)
    
    # Special handling for Abstract
    abstract_section = None
    for section_name in sections.keys():
        if "abstract" in section_name.lower():
            abstract_section = section_name
            break
    
    if abstract_section:
        metadata["abstract"] = sections[abstract_section]
        # del sections[abstract_section]
    
    # Convert defaultdict to regular dict and filter empty sections
    final_sections = {k: v for k, v in sections.items() if v}
    
    return {
        "metadata": metadata,
        "sections": final_sections
    }

def integrate_with_pdf_extractor(extractor_instance):
    """ Use with your existing ExtractTextInfoFromPDF class. extractor_instance: Instance of ExtractTextInfoFromPDF class"""
    # Get the raw structured data
    raw_data = extractor_instance.process_extracted_data()
    
    # Extract sections using our new function
    return extract_sections_from_pdf_json(raw_data)
