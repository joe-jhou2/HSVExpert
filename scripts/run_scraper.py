import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.pmc_download import download_papers_pipeline

def main():
    download_papers_pipeline("HSV", ["2000", "2025"], max_results=50000000, output_dir="data/unprocessed", download_pdf=False)

if __name__ == "__main__":
    main()