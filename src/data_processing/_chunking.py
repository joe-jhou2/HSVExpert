
# Load Sentence Transformer model
def chunk_sections(parsed_paper, max_tokens=500, tokenizer=None, overlap=50, model_type="bert"):
    """
    Chunk sections/subsections of a parsed paper into chunks of <= max_tokens with overlap.
    If a single sentence exceeds max_tokens, forcibly split it at word-level boundaries.
    Accounts for special tokens that models like BERT add during encoding.
    """
    chunks = []
    chunk_counter = 0  # global counter per paper
    
    # Reserve tokens for special tokens (CLS, SEP, etc.)
    special_token_reserve = 3 if model_type.lower() in ["bert", "roberta", "distilbert", "biobert", "pubmedbert"] else 0
    effective_max_tokens = max_tokens - special_token_reserve
    
    paper_id = parsed_paper.get("metadata", {}).get("paper_id", "unknown_id")
    paper_title = parsed_paper.get("metadata", {}).get("title", "No Title")
    authors = parsed_paper.get("metadata", {}).get("authors", [])
    year = parsed_paper.get("metadata", {}).get("year", None)
    journal = parsed_paper.get("metadata", {}).get("journal", None)
    
    for section_title, sentences in parsed_paper.get("sections", {}).items():
        # Preprocess sentences to ensure they are within effective_max_tokens
        processed_sentences = []
        
        for sentence in sentences:
            
            if tokenizer:
                if hasattr(tokenizer, 'encode') and not hasattr(tokenizer, 'tokenize'):
                    # tiktoken tokenizer
                    tokens = tokenizer.encode(sentence)
                else:
                    # BERT/Transformers tokenizer
                    tokens = tokenizer.encode(sentence, add_special_tokens=False)
            else:
                tokens = sentence.split()

            if len(tokens) <= effective_max_tokens:
                # keep the original sentence
                processed_sentences.append(sentence)
            else:
                # Split the sentence by tokens to fixed-size parts
                start_idx = 0
                while start_idx < len(tokens):
                    end_idx = min(start_idx + effective_max_tokens, len(tokens))
                    if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                        # tiktoken tokenizer
                        chunk_tokens = tokens[start_idx:end_idx]
                        truncated_sentence = tokenizer.decode(chunk_tokens)
                    else:
                        # BERT/Transformers tokenizer
                        chunk_tokens = tokens[start_idx:end_idx]
                        truncated_sentence = tokenizer.decode(chunk_tokens)
                    processed_sentences.append(truncated_sentence)
                    start_idx = end_idx
                            
        # Tokenize sentences and store (sentence, token_len) pairs
        tokenized_sentences = []
        for sentence in processed_sentences:
            if tokenizer:
                if hasattr(tokenizer, 'encode') and hasattr(tokenizer, 'decode'):
                    # tiktoken tokenizer
                    tokens = tokenizer.encode(sentence)
                else:
                    # BERT/Transformers tokenizer  
                    tokens = tokenizer.encode(sentence, add_special_tokens=False)
            else:
                tokens = sentence.split()
            tokenized_sentences.append((sentence, len(tokens)))
        
        # Chunking logic with overlap
        current_chunk = []
        current_token_count = 0
        i = 0
        
        while i < len(tokenized_sentences):
            sentence, token_len = tokenized_sentences[i]
            
            if current_token_count + token_len > effective_max_tokens and current_chunk:
                # Save current chunk
                chunk_text = " ".join([s for s, _ in current_chunk])
                chunks.append({
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "section_title": section_title,
                    "text": chunk_text,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "chunk_id": chunk_counter 
                })
                chunk_counter += 1
                
                # Create overlap - work backwards from current chunk
                overlap_tokens = 0
                overlap_chunk = []
                j = len(current_chunk) - 1
                
                # Build overlap chunk while staying within overlap limit
                while j >= 0 and overlap_tokens < overlap:
                    s, t = current_chunk[j]
                    if overlap_tokens + t <= overlap:  # Only add if it fits in overlap
                        overlap_tokens += t
                        overlap_chunk.insert(0, (s, t))
                    j -= 1
                
                # Start new chunk with overlap
                current_chunk = overlap_chunk
                current_token_count = sum(t for _, t in current_chunk)
            
            # Add current sentence if it fits
            if current_token_count + token_len <= effective_max_tokens:
                current_chunk.append((sentence, token_len))
                current_token_count += token_len
                i += 1
            else:
                print(f"Warning: Sentence still too long after preprocessing: {token_len} tokens")
                i += 1
        
        # Save final chunk
        if current_chunk:
            chunk_text = " ".join([s for s, _ in current_chunk])
            chunks.append({
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section_title": section_title,
                "text": chunk_text,
                "authors": authors,
                "year": year,
                "journal": journal
            })
    
    return chunks