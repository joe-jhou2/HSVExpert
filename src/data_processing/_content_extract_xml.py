
from bs4 import BeautifulSoup
import spacy
import re
import os

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

def extract_sections_from_pmc(xml_path):
    """Extract metadata and sentence-split hierarchical sections from PMC XML. Return a dict with metadata and structured sections."""
    # Check file exists and size
    if not os.path.exists(xml_path):
        print(f"ERROR: File does not exist: {xml_path}")
        return {"metadata": {}, "sections": {}}
    
    file_size = os.path.getsize(xml_path)
    print(f"Processing file {xml_path} ({file_size} bytes)")
    
    try:
        with open(xml_path, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "xml")
    except Exception as e:
        print(f"ERROR: Failed to parse XML: {e}")
        return {"metadata": {}, "sections": {}}

    # Check basic XML structure
    print(f"Root tag: {soup.name if soup.name else 'None'}")
    if soup.find("article"):
        print("Found <article> tag")
    if soup.find("body"):
        print("Found <body> tag")
    sec_tags = soup.find_all("sec")
    print(f"Found {len(sec_tags)} <sec> tags")

    # === METADATA ===
    article_meta = soup.find("article-meta")
    metadata = {
        "title": soup.find("article-title").text.strip() if soup.find("article-title") else None,
        "authors": [], #[name.get_text(" ", strip=True) for name in soup.find_all("name")],
        "journal": soup.find("journal-title").text.strip() if soup.find("journal-title") else None,
        "year": None,
        "abstract": ""
    }

    # Extract authors
    contrib_group = soup.find("contrib-group")
    if contrib_group:
        authors = contrib_group.find_all("contrib", {"contrib-type": "author"})
        for author in authors:
            name_tag = author.find("name")
            if name_tag:
                surname_tag = name_tag.find("surname")
                given_names_tag = name_tag.find("given-names")
                surname = surname_tag.text.strip() if surname_tag else ""
                given_names = given_names_tag.text.strip() if given_names_tag else ""
                full_name = f"{given_names} {surname}".strip()
                if full_name:
                    metadata["authors"].append(full_name)

    # Extract year                
    pub_date = soup.find("pub-date", {"pub-type": "epub"}) or soup.find("pub-date")
    if pub_date and pub_date.find("year"):
        metadata["year"] = pub_date.find("year").text.strip()

    # Extract abstract
    abstract_tag = article_meta.find("abstract") if article_meta else None
    if abstract_tag:
        abstract_text = " ".join(p.get_text(" ", strip=True) for p in abstract_tag.find_all("p"))
        metadata["abstract"] = split_sentences(abstract_text)
    else:
        metadata["abstract"] = []

    # === SECTIONS ===
    body = soup.find("body")
    sections = {}

    def recurse_sections(tag, parent_titles=[]):
        for sec in tag.find_all("sec", recursive=False):
            title_tag = sec.find("title")
            raw_title = title_tag.text.strip() if title_tag else "Untitled"

            cleaned_title = clean_section_title(raw_title)

            # Normalize only top-level titles
            norm_title = normalize_section(cleaned_title) if not parent_titles else cleaned_title
            full_title = " > ".join(parent_titles + [norm_title])

            # Extract paragraph text
            paragraphs = [p.get_text(" ", strip=True) for p in sec.find_all("p", recursive=False)]
            combined_text = " ".join(paragraphs).strip()

            # Sentence split and store
            if combined_text:
                sections[full_title] = sections.get(full_title, []) + split_sentences(combined_text)

            # Recurse into subsections
            recurse_sections(sec, parent_titles + [norm_title])

    if body:
        recurse_sections(body)
        print(f"Total sections extracted: {len(sections)}")
    else:
        print("No <body> tag found")

    return {
        "metadata": metadata,
        "sections": sections
    }

