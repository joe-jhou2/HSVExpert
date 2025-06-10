from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from typing import List, Dict, Optional
import logging
import numpy as np
from src.data_processing import embed_text
from src.retrieval import query_qdrant

class HybridRetrievalPipeline:
    """
    Multi-stage retrieval pipeline combining:
    1. Dense vector search (Qdrant)
    2. Sparse BM25 retrieval  
    3. Reciprocal Rank Fusion (RRF)
    4. Cross-encoder re-ranking
    """
    
    def __init__(self, 
                 collection_name: str = "hsv_papers",
                 host: str = "localhost", 
                 port: int = 6333,
                 cross_encoder_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 hsv_optimized: bool = True):
        """
        Initialize the hybrid retrieval pipeline.
        
        Args:
            collection_name: Qdrant collection name
            host: Qdrant host
            port: Qdrant port  
            cross_encoder_model: Cross-encoder model for re-ranking
            hsv_optimized: Whether to use HSV-specific optimizations
        """
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.client = QdrantClient(host=host, port=port)
        
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = None
        self.cross_encoder_model = cross_encoder_model
        self._load_cross_encoder()
        
        # HSV-specific keywords for domain boosting
        self.hsv_optimized = hsv_optimized
        if hsv_optimized:
            self.hsv_keywords = [
                'hsv-1', 'hsv-2', 'herpes simplex', 'herpes',
                'acyclovir', 'valacyclovir', 'famciclovir', 'antiviral',
                'suppressive therapy', 'episodic treatment',
                'genital herpes', 'oral herpes', 'cold sores', 'fever blisters',
                'recurrence', 'outbreak', 'lesion', 'vesicle',
                'seroprevalence', 'transmission', 'asymptomatic',
                'immunocompromised', 'neonatal herpes',
                'clinical trial', 'efficacy', 'safety', 'adverse events',
                'immune response', 'antibody', 'seroconversion', 'vaccine',
                'viral shedding', 'latency', 'reactivation'
            ]
        
        # BM25 index (will be built when needed)
        self.bm25_index = None
        self.bm25_chunks = None
        
    def _load_cross_encoder(self):
        """Load cross-encoder model with error handling."""
        try:
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
            logging.info(f"Loaded cross-encoder: {self.cross_encoder_model}")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder: {e}")
            self.cross_encoder = None
    
    def build_bm25_index(self, chunks: List[Dict]):
        """Build BM25 index from chunks."""
        try:
            tokenized_corpus = [chunk["text"].split() for chunk in chunks]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_chunks = chunks
            logging.info(f"Built BM25 index with {len(chunks)} chunks")
        except Exception as e:
            logging.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def dense_retrieval(self, 
                       query_text: str,
                       embed_function,
                       tokenizer,
                       model,
                       top_k: int = 50) -> List[Dict]:
        """Perform dense vector retrieval using Qdrant."""
        try:
            chunks = query_qdrant(
                query_text=query_text,
                embed_function=embed_function,
                tokenizer=tokenizer,
                model=model,
                collection_name=self.collection_name,
                host=self.host,
                port=self.port,
                top_k=top_k
            )
            
            # Add retrieval source and scores
            for i, chunk in enumerate(chunks):
                chunk['dense_rank'] = i + 1
                chunk['retrieval_source'] = 'dense'
                
            return chunks
            
        except Exception as e:
            logging.error(f"Dense retrieval failed: {e}")
            return []
    
    def sparse_retrieval(self, 
                        query_text: str, 
                        chunks: List[Dict],
                        top_k: int = 50) -> List[Dict]:
        """Perform sparse BM25 retrieval."""
        if not self.bm25_index or not chunks:
            # Build index if not available
            if chunks:
                self.build_bm25_index(chunks)
            else:
                return []
        
        try:
            # Get BM25 scores
            query_tokens = query_text.split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Pair chunks with scores and rank
            scored_chunks = []
            for i, (chunk, score) in enumerate(zip(chunks, bm25_scores)):
                chunk_copy = chunk.copy()
                chunk_copy['bm25_score'] = float(score)
                chunk_copy['sparse_rank'] = i + 1
                chunk_copy['retrieval_source'] = 'sparse'
                scored_chunks.append(chunk_copy)
            
            # Sort by BM25 score and return top-k
            ranked_chunks = sorted(scored_chunks, 
                                 key=lambda x: x['bm25_score'], 
                                 reverse=True)
            
            return ranked_chunks[:top_k]
            
        except Exception as e:
            logging.error(f"Sparse retrieval failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, 
                              dense_results: List[Dict],
                              sparse_results: List[Dict],
                              k: int = 60) -> List[Dict]:
        """Apply Reciprocal Rank Fusion to combine dense and sparse results."""
        
        # Create lookup dictionaries for efficient merging
        dense_lookup = {chunk.get('id', f'dense_{rank}'): (chunk, rank + 1) 
                    for rank, chunk in enumerate(dense_results)}
        sparse_lookup = {chunk.get('id', f'sparse_{rank}'): (chunk, rank + 1) 
                        for rank, chunk in enumerate(sparse_results)}
        
        # Calculate RRF scores
        all_chunk_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        rrf_scores = {}
        merged_chunks = {}
        
        for chunk_id in all_chunk_ids:
            rrf_score = 0
            
            # Dense contribution
            if chunk_id in dense_lookup:
                chunk, rank = dense_lookup[chunk_id]
                rrf_score += 1 / (k + rank)
                merged_chunks[chunk_id] = chunk
            
            # Sparse contribution  
            if chunk_id in sparse_lookup:
                chunk, rank = sparse_lookup[chunk_id]
                rrf_score += 1 / (k + rank)
                if chunk_id not in merged_chunks:
                    merged_chunks[chunk_id] = chunk
            
            rrf_scores[chunk_id] = rrf_score
        
        # Add RRF scores to chunks and sort
        for chunk_id, chunk in merged_chunks.items():
            chunk['rrf_score'] = rrf_scores[chunk_id]
            chunk['retrieval_source'] = 'hybrid'
        
        # Sort by RRF score
        ranked_chunks = sorted(merged_chunks.values(),
                             key=lambda x: x['rrf_score'],
                             reverse=True)
        
        return ranked_chunks
    
    def cross_encoder_rerank(self, 
                            query_text: str,
                            chunks: List[Dict],
                            top_n: int = 10) -> List[Dict]:
        """Re-rank chunks using cross-encoder."""
        
        if not chunks or not self.cross_encoder:
            return chunks[:top_n]
        
        try:
            # Prepare query-passage pairs
            pairs = []
            for chunk in chunks:
                text = chunk.get("text", "")
                # Truncate long texts
                if len(text) > 2000:
                    text = text[:2000] + "..."
                pairs.append((query_text, text))
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Add scores to chunks
            for chunk, score in zip(chunks, cross_scores):
                chunk['cross_encoder_score'] = float(score)
                
                # Apply HSV-specific boosting if enabled
                if self.hsv_optimized:
                    boost = self._calculate_hsv_boost(query_text, chunk)
                    chunk['cross_encoder_score'] *= boost
                    chunk['hsv_boost'] = boost
            
            # Sort by cross-encoder score
            reranked_chunks = sorted(chunks,
                                   key=lambda x: x['cross_encoder_score'],
                                   reverse=True)
            
            return reranked_chunks[:top_n]
            
        except Exception as e:
            logging.error(f"Cross-encoder re-ranking failed: {e}")
            return chunks[:top_n]
    
    def _calculate_hsv_boost(self, query_text: str, chunk: Dict) -> float:
        """Calculate HSV-specific boost factor."""
        if not self.hsv_optimized:
            return 1.0
        
        text_lower = chunk.get("text", "").lower()
        query_lower = query_text.lower()
        
        # HSV keyword matching
        hsv_matches = sum(1 for keyword in self.hsv_keywords 
                         if keyword in text_lower)
        
        # Query-text keyword overlap
        query_words = set(query_lower.split())
        text_words = set(text_lower.split())
        overlap = len(query_words.intersection(text_words))
        
        # Section relevance boost
        section_title = chunk.get("section_title", "").lower()
        section_boost = 1.0
        
        if any(term in section_title for term in 
               ['result', 'discussion', 'conclusion', 'treatment', 'method']):
            section_boost = 1.2
        elif any(term in section_title for term in 
                 ['introduction', 'background', 'reference']):
            section_boost = 0.9
        
        # Calculate total boost
        hsv_boost = 1 + (hsv_matches * 0.15)
        overlap_boost = 1 + (overlap * 0.05)
        
        return hsv_boost * overlap_boost * section_boost
    
    def hybrid_search(self,
                     query_text: str,
                     embed_function,
                     tokenizer, 
                     model,
                     initial_retrieval_k: int = 50,
                     rrf_candidates: int = 30,
                     final_top_n: int = 10,
                     use_all_chunks_for_bm25: bool = False,
                     all_chunks: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Complete hybrid search pipeline.
        
        Args:
            query_text: User query
            embed_function: Embedding function
            tokenizer: Tokenizer
            model: Embedding model
            initial_retrieval_k: Initial retrieval count for each method
            rrf_candidates: Number of candidates for RRF
            final_top_n: Final number of results after re-ranking
            use_all_chunks_for_bm25: Whether to use all chunks for BM25 (not just dense results)
            all_chunks: All available chunks (needed if use_all_chunks_for_bm25=True)
        
        Returns:
            List of re-ranked chunks
        """
        
        logging.info(f"Starting hybrid search for query: {query_text[:50]}...")
        
        # Stage 1: Dense retrieval
        dense_results = self.dense_retrieval(
            query_text, embed_function, tokenizer, model, initial_retrieval_k
        )
        
        # Stage 2: Sparse retrieval
        if use_all_chunks_for_bm25 and all_chunks:
            sparse_results = self.sparse_retrieval(query_text, all_chunks, initial_retrieval_k)
        else:
            # Use dense results for BM25 (faster, but less comprehensive)
            sparse_results = self.sparse_retrieval(query_text, dense_results, initial_retrieval_k)
        
        # Stage 3: Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            dense_results[:rrf_candidates], 
            sparse_results[:rrf_candidates]
        )
        
        # Stage 4: Cross-encoder re-ranking
        final_results = self.cross_encoder_rerank(
            query_text, fused_results, final_top_n
        )
        
        # Log results
        logging.info(f"Dense: {len(dense_results)}, Sparse: {len(sparse_results)}, "
                    f"Fused: {len(fused_results)}, Final: {len(final_results)}")
        
        return final_results

# # Convenience function for easy integration
# def enhanced_hybrid_query(query_text: str,
#                          embed_function,
#                          tokenizer,
#                          model,
#                          collection_name: str = "papers",
#                          host: str = "localhost",
#                          port: int = 6333,
#                          top_n: int = 10,
#                          hsv_optimized: bool = True,
#                          all_chunks: Optional[List[Dict]] = None) -> List[Dict]:
#     """
#     Simple interface for hybrid retrieval.
    
#     Args:
#         query_text: User query
#         embed_function: Embedding function
#         tokenizer: Tokenizer  
#         model: Embedding model
#         collection_name: Qdrant collection name
#         host: Qdrant host
#         port: Qdrant port
#         top_n: Number of final results
#         hsv_optimized: Whether to use HSV-specific optimizations
#         all_chunks: All chunks for comprehensive BM25 (optional)
    
#     Returns:
#         List of top-ranked chunks
#     """
    
#     # Initialize pipeline
#     pipeline = HybridRetrievalPipeline(
#         collection_name=collection_name,
#         host=host,
#         port=port,
#         hsv_optimized=hsv_optimized
#     )
    
#     # Perform hybrid search
#     results = pipeline.hybrid_search(
#         query_text=query_text,
#         embed_function=embed_function,
#         tokenizer=tokenizer,
#         model=model,
#         final_top_n=top_n,
#         use_all_chunks_for_bm25=all_chunks is not None,
#         all_chunks=all_chunks
#     )
    
#     return results
