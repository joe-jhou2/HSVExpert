import os
from lxml import etree
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from src.data_processing import embed_openai, embed_text

# ==========================================
# XML Validation and Parsing Utilities
# ==========================================
def is_valid_paper(xml_file):
    """Check if the XML file contains a valid paper. If not, delete it."""
    try:
        with open(xml_file, "rb") as f:
            tree = etree.parse(f)
            # Check if there's an <error> tag
            if tree.find(".//error") is not None:
                print(f"Invalid paper found (contains <error>): {xml_file}. Deleting.")
                os.remove(xml_file)
                return False
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}. Deleting.")
        os.remove(xml_file)
        return False
    return True

# ==========================================
# Paper Analysis Utilities
# ==========================================
def analyze_parsed_paper(parsed_paper, tokenizer=None):
    """Analyze the structure and content of a parsed paper"""
    
    print("=" * 60)
    print("PARSED PAPER ANALYSIS")
    print("=" * 60)
    
    # 1. Overall structure
    print(f"Top-level keys: {list(parsed_paper.keys())}")
    print()
    
    # 2. Metadata analysis
    if "metadata" in parsed_paper:
        metadata = parsed_paper["metadata"]
        print("METADATA:")
        for key, value in metadata.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} items")
                if value and len(str(value[0])) < 100:
                    print(f"    First item: {value[0]}")
            else:
                value_str = str(value)
                if len(value_str) > 100:
                    print(f"  {key}: {value_str[:100]}...")
                else:
                    print(f"  {key}: {value}")
        print()
    
    # 3. Sections analysis
    if "sections" in parsed_paper:
        sections = parsed_paper["sections"]
        print(f"SECTIONS: {len(sections)} total sections")
        print()
        
        total_sentences = 0
        problematic_sentences = []
        
        for section_title, sentences in sections.items():
            print(f"Section: '{section_title}'")
            print(f"  Number of sentences: {len(sentences)}")
            
            if sentences:
                # Analyze sentence lengths
                sentence_lengths = []
                for i, sentence in enumerate(sentences):
                    if tokenizer:
                        # Get token count
                        token_count = len(tokenizer.encode(sentence, add_special_tokens=False))
                        sentence_lengths.append(token_count)
                        
                        # Flag very long sentences
                        if token_count > 400:  # Arbitrary threshold for "very long"
                            problematic_sentences.append({
                                'section': section_title,
                                'sentence_idx': i,
                                'token_count': token_count,
                                'text_preview': sentence[:200] + "..." if len(sentence) > 200 else sentence
                            })
                    else:
                        # Fallback to word count
                        word_count = len(sentence.split())
                        sentence_lengths.append(word_count)
                        
                        if word_count > 300:  # Rough equivalent
                            problematic_sentences.append({
                                'section': section_title,
                                'sentence_idx': i,
                                'word_count': word_count,
                                'text_preview': sentence[:200] + "..." if len(sentence) > 200 else sentence
                            })
                
                if sentence_lengths:
                    print(f"  Sentence lengths - Min: {min(sentence_lengths)}, Max: {max(sentence_lengths)}, Avg: {sum(sentence_lengths)/len(sentence_lengths):.1f}")
                    
                    # Show some examples
                    if max(sentence_lengths) > (400 if tokenizer else 300):
                        longest_idx = sentence_lengths.index(max(sentence_lengths))
                        longest_sentence = sentences[longest_idx]
                        print(f"  Longest sentence ({max(sentence_lengths)} {'tokens' if tokenizer else 'words'}):")
                        print(f"    '{longest_sentence[:150]}...'")
                
                total_sentences += len(sentences)
            
            print()
        
        print(f"SUMMARY:")
        print(f"  Total sentences across all sections: {total_sentences}")
        print(f"  Sections with content: {sum(1 for s in sections.values() if s)}")
        print(f"  Empty sections: {sum(1 for s in sections.values() if not s)}")
        
        if problematic_sentences:
            print(f"\n🚨 PROBLEMATIC SENTENCES ({len(problematic_sentences)} found):")
            for i, prob in enumerate(problematic_sentences[:5]):  # Show first 5
                metric = 'tokens' if tokenizer else 'words'
                count_key = 'token_count' if tokenizer else 'word_count'
                print(f"  {i+1}. Section '{prob['section']}', sentence {prob['sentence_idx']}")
                print(f"     Length: {prob[count_key]} {metric}")
                print(f"     Preview: {prob['text_preview']}")
                print()
            
            if len(problematic_sentences) > 5:
                print(f"     ... and {len(problematic_sentences) - 5} more")
        
        print("=" * 60)

# ==========================================
# Tokenization and Embedding Utilities
# ==========================================
def get_embedding_function(embed_type="bert"):
    """Factory function to get the appropriate embedding function"""
    if embed_type.lower() == "openai":
        return embed_openai
    elif embed_type.lower() == "bert":
        return embed_text
    else:
        raise ValueError(f"Unknown embedding type: {embed_type}")

def count_tokens(text, tokenizer=None, model_type="bert"):
    """Count tokens for different model types. Useful for chunking decisions"""
    if tokenizer is None:
        return len(text.split())  # Simple word count fallback

    if model_type.lower() == "openai":
        # tiktoken tokenizer
        return len(tokenizer.encode(text))
    else:
        # BERT/Transformers tokenizer
        return len(tokenizer.encode(text, add_special_tokens=True))


# ==========================================
# PDF Processing Counter Utilities
# ==========================================
def load_pdf_count(counter_file):
    if Path(counter_file).exists():
        with open(counter_file, "r") as f:
            data = json.load(f)
            return data.get("total_pdfs_processed", 0), data.get("last_reset", None)
    return 0, None

def increment_pdf_count(counter_file):
    auto_reset_if_new_month(counter_file)

    count, last_reset = load_pdf_count(counter_file)
    count += 1
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    reset_date = last_reset if last_reset else now

    with open(counter_file, "w") as f:
        json.dump({"total_pdfs_processed": count, "last_reset": reset_date}, f)
    return count

def reset_pdf_count(counter_file):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(counter_file, "w") as f:
        json.dump({"total_pdfs_processed": 0, "last_reset": now}, f)

def auto_reset_if_new_month(counter_file):
    count, last_reset = load_pdf_count(counter_file)
    now = datetime.now(timezone.utc)
    
    if last_reset:
        last_reset_date = datetime.strptime(last_reset, "%Y-%m-%d")
        if now.year != last_reset_date.year or now.month != last_reset_date.month:
            reset_pdf_count(counter_file)
            return True
    return False
