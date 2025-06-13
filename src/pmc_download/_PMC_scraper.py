
import os
import time
import requests
from pathlib import Path
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from Bio import Entrez
import urllib.request
import ftplib
from urllib.parse import urlparse
import dotenv

# === Load environment ===
dotenv.load_dotenv()
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
Entrez.email = ENTREZ_EMAIL

# === Search PMC ===
def search_pmc_papers(keyword, years, max_results=100):
    """
    Search for PMC papers based on a keyword and date range.
    Args:
        keyword (str): Keyword to search for in the title/abstract.
        years (list): List of two strings representing the start and end year.
        max_results (int): Maximum number of results to return.
    Returns:
        list: List of PMC IDs for the papers found.
    """
    # Search keyword in title/abstract
    # query = f"{keyword} [Title/Abstract]"
    query = f"{keyword}[Title/Abstract] OR {keyword}[Text Word]"

    # Search for papers, years is two-element list
    # e.g. ["2015", "2025"] for papers published between 2015 and 2025
    handle = Entrez.esearch(
        db="pmc", 
        term=query, 
        retmax=max_results,
        mindate=years[0], 
        maxdate=years[1]
        )
    
    # Read the results
    # Entrez.read() returns a dictionary with "IdList" key containing a list of PMC IDs
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]

# === Download XML ===
def download_pmc_xml(pmc_id, output_dir="data/unprocessed"):
    """
    Download the full-text XML of a PMC paper.
    Args:
        pmc_id (str): PMC ID of the paper to download.
        output_dir (str): Directory to save the downloaded XML files.
    Returns:
        str: Path to the downloaded XML file.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"PMC{pmc_id}.xml")
    # Check if the file already exists, avoding re-download
    if os.path.exists(output_path):
        print(f"Skipping PMC{pmc_id} (already downloaded)")
        return output_path
    
    # Also check if the file is already downloaded in the processed folder
    processed_path = os.path.join("data/processed", f"PMC{pmc_id}.xml")
    if os.path.exists(processed_path):
        print(f"Skipping PMC{pmc_id} (already processed)")
        return processed_path

    # Download the XML file
    try:
        print(f"Downloading PMC{pmc_id}...")
        fetch_handle = Entrez.efetch(
            db="pmc", 
            id=pmc_id, 
            rettype="full", 
            retmode="xml"
            )
        with open(output_path, "wb") as f:
            f.write(fetch_handle.read())
        fetch_handle.close()
        time.sleep(0.5)
        return output_path
    except Exception as e:
        print(f"Error downloading PMC{pmc_id}: {e}")
        return None

# === Download PDF from PMC FTP ===
def get_pdf_url_from_ftp(pmc_id):
    """Get PDF URL for a PMC ID"""
    if not pmc_id.startswith('PMC'):
        pmc_id = 'PMC' + pmc_id
    
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        root = ET.fromstring(response.text)
        pdf_link = root.find('.//link[@format="pdf"]')
        
        if pdf_link is not None:
            return pdf_link.get('href')
        else:
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def download_pdf_from_ftp(ftp_url, pmc_id, output_dir="data/unprocessed"):
    """Download PDF from FTP URL"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"PMC{pmc_id}.xml")
    # Check if the file already exists, avoding re-download
    if os.path.exists(output_path):
        print(f"Skipping PMC{pmc_id} (already downloaded)")
        return output_path
    
    try:
        print(f"Downloading from FTP: {ftp_url}")
        
        # Parse FTP URL
        parsed = urlparse(ftp_url)
        server = parsed.netloc
        path = parsed.path
        
        # Connect to FTP server
        ftp = ftplib.FTP(server)
        ftp.login()  # Anonymous login
        
        # Download file
        with open(output_dir, 'wb') as f:
            ftp.retrbinary(f'RETR {path}', f.write)
        
        ftp.quit()
        print(f"✓ Downloaded successfully: {output_dir}")
        return True
        
    except Exception as e:
        print(f"✗ FTP download failed: {e}")
        return False

# === Get PDF URL from article page ===
# Beware, this method maybe blocked by NCBI due to anti-bot measures.
# Switch to EU PMC in the next function download_pdf_from_page, 
def get_pdf_url_from_page(pmc_id):
    """Try to extract PDF URL by scraping the article page."""
    base_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for citation_pdf_url meta tag
        meta_tag = soup.find("meta", {"name": "citation_pdf_url"})
        if meta_tag and meta_tag.get("content"):
            pdf_url = meta_tag["content"]
            print("✓ PDF URL found:", pdf_url)
            return pdf_url
        else:
            print("✗ No PDF meta tag found.")

    except Exception as e:
        print(f"✗ Error fetching page: {e}")

    
def download_pdf_from_page(pmc_id, output_dir="data/unprocessed"):
    """Download PDF from a URL."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"PMC{pmc_id}.pdf")

    if os.path.exists(output_path):
        print(f"✓ Already downloaded: {output_path}")
        return output_path

    print(f"🇪🇺 Trying EU PMC for {pmc_id}...")
    pdf_url = f"https://europepmc.org/articles/PMC{pmc_id}?pdf=render"
    print(f"PDF URL: {pdf_url}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Referer": f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/",
        "Accept": "application/pdf",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })

    print(f"Downloading from Page: {pdf_url}")
    try:
        response = session.get(pdf_url, stream=True, timeout=15, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type:
            print(f"✗ Not a PDF (Content-Type: {content_type}) — saving temp to inspect.")
            with open(output_path + ".html", "wb") as f:
                f.write(response.content)
            return None

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with open(output_path, "rb") as f:
            if f.read(5) != b"%PDF-":
                print("✗ File doesn't start with %PDF-. Likely corrupted. Removing.")
                os.remove(output_path)
                return None

        print(f"✓ PDF downloaded: {output_path}")
        return output_path

    except Exception as e:
        print(f"✗ Exception during download: {e}")
        return None
    
# === Full pipeline ===
def download_papers_pipeline(
        keyword, 
        years, 
        max_results=100,
        output_dir="data/unprocessed",
        download_pdf=True
        ):
    """
    Pipeline to search and download PMC papers.
    Args:
        keyword (str): Keyword to search for in the title/abstract.
        years (list): List of two strings representing the start and end year.
        max_results (int): Maximum number of results to return.
    """
    # Search for papers
    pmc_ids = search_pmc_papers(keyword, years, max_results)
    # Download the XML files
    for pmc_id in pmc_ids:
        # download_pmc_xml(pmc_id, output_dir=output_dir)
        xml_path = download_pmc_xml(pmc_id, output_dir=output_dir)
        if download_pdf:
            pdf_url_ftp = get_pdf_url_from_ftp(pmc_id)
            if pdf_url_ftp:
                download_pdf_from_ftp(pdf_url_ftp, pmc_id, output_dir=output_dir)
            else:
                download_pdf_from_page(pmc_id, output_dir=output_dir)
