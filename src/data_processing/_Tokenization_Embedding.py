import torch
from transformers import AutoTokenizer, AutoModel
import tiktoken

# ==========================================
# BERT/Transformers Embedding Functions
# ==========================================
def load_embedding_model(model_name: str = "bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

def embed_text(text, tokenizer, model, max_length=512):
    # Double-check token count before processing
    token_count = len(tokenizer.encode(text, add_special_tokens=True))
    if token_count > max_length:
        print(f"Warning: Text has {token_count} tokens, truncating to {max_length}")
    
    # Tokenize and encode
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
    
    # Forward pass to get hidden states
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use CLS token representation
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

# ==========================================
# OpenAI Embedding Functions
# ==========================================

def embed_openai(text, client, model="text-embedding-3-large"):
    """Generate embeddings using OpenAI's embedding API"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    # Extract the embedding
    embedding = response.data[0].embedding
    return embedding

def get_openai_tokenizer(model="text-embedding-3-large"):
    """Get the appropriate tokenizer for OpenAI models. Use this for chunking purposes (token counting)"""
    # text-embedding-3-large uses cl100k_base encoding
    return tiktoken.get_encoding("cl100k_base")


