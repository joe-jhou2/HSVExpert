import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import glob
import shutil
import dotenv
import traceback
dotenv.load_dotenv()

from src.data_processing._content_extract_xml import extract_sections_from_pmc
from src.data_processing._chunking import chunk_sections
from src.data_processing._Tokenization_Embedding import load_embedding_model, embed_text
from src.data_processing._qdrant import store_chunks_in_qdrant

QDRANT_EC2_IP = os.getenv("QDRANT_EC2_IP")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Notes, the max_tokens and overlap are setup to so after some experimentation, 
# the chunks are around 300 tokens with a 50 token overlap have better performance
# in terms of retrieval and relevance. 
def process_pmc_paper(xml_file_path, max_tokens=300, overlap=50):
    print(f"Processing paper: {xml_file_path}")
    
    # Load embedding model once
    print("Loading PubMedBERT model...")
    pubmedbert_tokenizer, pubmedbert_model = load_embedding_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    
    # Extract sections from downloaded paper
    print("Extracting sections...")
    parsed_paper = extract_sections_from_pmc(xml_file_path)
    
    # Add paper ID from filename
    paper_id = os.path.splitext(os.path.basename(xml_file_path))[0]
    parsed_paper["metadata"]["paper_id"] = paper_id
    print(f"Paper ID: {paper_id}")
    
    # Chunking 
    print(f"Chunking sections (max_tokens={max_tokens}, overlap={overlap})...")
    chunks = chunk_sections(parsed_paper, 
                            max_tokens=max_tokens, 
                            tokenizer=pubmedbert_tokenizer, 
                            overlap=overlap, 
                            model_type="pubmedbert")
    print(f"Created {len(chunks)} chunks")
    
    # Embedding chunks efficiently
    print("Embedding chunks...")
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:  # Progress indicator
            print(f"  Embedding chunk {i+1}/{len(chunks)}")
        try:
            text = chunk["text"]
            embedding = embed_text(text, pubmedbert_tokenizer, pubmedbert_model)
            chunk["embedding"] = embedding
        except Exception as e:
            print(f"  ✗ Error embedding chunk {i+1}: {e}")
            traceback.print_exc()
            raise e
    
    # Extract embeddings for storage
    embeddings = [chunk["embedding"] for chunk in chunks if "embedding" in chunk]
    
    # Store chunks in Qdrant
    print("Storing chunks in Qdrant...")
    try:
        store_chunks_in_qdrant(
            chunks, 
            embeddings, 
            embed_model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            collection_name="hsv_papers",
            source = "pre-processed",
            host=QDRANT_EC2_IP,
            port=QDRANT_PORT
        )
        print(f"  ✓ Chunks successfully stored for paper {paper_id}")
    except Exception as e:
        print(f"  ✗ Error storing chunks for {paper_id}: {e}")
        traceback.print_exc()
        # Raise to skip this paper entirely
        raise e
    
    print(f"Successfully processed {paper_id} with {len(chunks)} chunks")
    
    return {
        "paper_id": paper_id,
        "num_chunks": len(chunks),
        "chunks": chunks,
        "embeddings": embeddings
    }

def main():
    """Main execution function"""
    xml_directory = "data/unprocessed"  # Directory containing XML files
    processed_directory = "data/processed"
    
    if not os.path.exists(xml_directory):
        print(f"Directory not found: {xml_directory}")
        return
    
    # Create processed folder if it doesn't exist
    os.makedirs(processed_directory, exist_ok=True)

    # Find all XML files in the directory
    xml_pattern = os.path.join(xml_directory, "*.xml")
    xml_files = glob.glob(xml_pattern)

    # Find XML files already processed
    xml_pattern_processed = os.path.join(processed_directory, "*.xml")
    xml_files_processed = glob.glob(xml_pattern_processed)
    
    # Get just the filenames (without full path) for comparison
    processed_filenames = set(os.path.basename(f) for f in xml_files_processed)

    # Filter out files that are already processed
    xml_files_to_process = [
        xml_file for xml_file in xml_files 
        if os.path.basename(xml_file) not in processed_filenames
    ]

    if not xml_files:
        print(f"No XML files found in {xml_directory}")
        return
    
    if not xml_files_to_process:
        print(f"All {len(xml_files)} XML files have already been processed")
        return
    
    print(f"Found {len(xml_files)} total XML files")
    print(f"Found {len(xml_files_processed)} already processed files")
    print(f"Found {len(xml_files_to_process)} XML files to process")

    # Process each XML file
    results = []
    for i, xml_file_path in enumerate(xml_files_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {i}/{len(xml_files_to_process)}: {os.path.basename(xml_file_path)}")
        print(f"{'='*60}")
        
        try:
            result = process_pmc_paper(xml_file_path, max_tokens=300, overlap=50)
            results.append(result)
            print(f"✓ Successfully processed {result['paper_id']}")

            # Move file to processed folder
            filename = os.path.basename(xml_file_path)
            destination = os.path.join(processed_directory, filename)
            shutil.move(xml_file_path, destination)
            print(f"→ Moved to {destination}")

        except Exception as e:
            print(f"✗ Error processing {xml_file_path}: {str(e)}")
            traceback.print_exc()
            continue
    
    # Report each processed paper
    if results:
        print("\nProcessed papers:")
        for result in results:
            print(f"  - {result['paper_id']}: {result['num_chunks']} chunks")
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(results)}/{len(xml_files_to_process)}")
    
    total_chunks = sum(result['num_chunks'] for result in results)
    print(f"Total chunks created: {total_chunks}")
    
if __name__ == "__main__":
    main()