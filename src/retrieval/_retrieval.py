from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from src.data_processing._Tokenization_Embedding import embed_text

def query_qdrant(query_text, embed_function, tokenizer, model, top_k=5, collection_name="papers", host="localhost", port=6333):
    """
    Query Qdrant for top_k relevant chunks based on the query text.

    Args:
        query_text (str): User question.
        embed_function (callable): Function to embed the query text.
        tokenizer (AutoTokenizer): Tokenizer for the embedding model.
        model (AutoModel): Embedding model.
        top_k (int): Number of results to return.
        collection_name (str): Qdrant collection name.
        host (str): Qdrant host.
        port (int): Qdrant port.

    Returns:
        List of dicts with retrieved chunk payloads.
    """
    # Embed the query text
    query_embedding = embed_function(query_text, tokenizer, model)

    # Convert embedding to list (for Qdrant)
    query_embedding = query_embedding.tolist()

    # Create Qdrant client
    client = QdrantClient(host=host, port=port)

    # Perform similarity search
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )

    # Return the retrieved chunk payloads
    return [hit.payload for hit in search_result]

def build_bm25_index(chunks):
    tokenized_corpus = [chunk["text"].split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def dual_retriever_fusion_query(query, chunks, dense_model, tokenizer, top_k=10):
    # ----- 1. Dense retrieval -----
    query_embedding = embed_text(query, tokenizer, dense_model)
    chunk_embeddings = [chunk["embedding"] for chunk in chunks if "embedding" in chunk]
    
    # Compute cosine similarity manually    
    sims = cosine_similarity([query_embedding], chunk_embeddings)[0]
    dense_scores = {i: score for i, score in enumerate(sims)}
    
    # ----- 2. Sparse (BM25) retrieval -----
    bm25, tokenized = build_bm25_index(chunks)
    bm25_scores = bm25.get_scores(query.split())
    sparse_scores = {i: score for i, score in enumerate(bm25_scores)}
    
    # ----- 3. Reciprocal Rank Fusion (RRF) -----
    def rrf(scores_dict, k=60):
        return {i: 1 / (k + rank) for rank, (i, _) in enumerate(sorted(scores_dict.items(), key=lambda x: -x[1]))}

    dense_rrf = rrf(dense_scores)
    sparse_rrf = rrf(sparse_scores)

    # Combine RRF scores
    final_scores = {}
    for i in set(dense_rrf) | set(sparse_rrf):
        final_scores[i] = dense_rrf.get(i, 0) + sparse_rrf.get(i, 0)

    top_indices = sorted(final_scores.items(), key=lambda x: -x[1])[:top_k]
    return [chunks[i] for i, _ in top_indices]
