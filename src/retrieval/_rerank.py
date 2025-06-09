from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging
from src.retrieval._retrieval import query_qdrant

class ReRanker:
    """Re-ranking utility class for better chunk relevance scoring."""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """Initialize the re-ranker with a cross-encoder model. model_name: HuggingFace model name for cross-encoder"""
        self.model_name = model_name
        self.cross_encoder = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model with error handling."""
        try:
            self.cross_encoder = CrossEncoder(self.model_name)
            logging.info(f"Loaded cross-encoder model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder model: {e}")
            self.cross_encoder = None
    
    def re_rank_chunks(self, 
                       user_question: str, 
                       retrieved_chunks: List[Dict], 
                       top_n: int = 5,
                       score_threshold: float = None,
                       combine_scores: bool = True,
                       alpha: float = 0.7) -> List[Dict]:
        """
        Re-rank retrieved chunks using cross-encoder for better relevance.
        
        Args:
            user_question: The user's query
            retrieved_chunks: List of chunks from vector search
            top_n: Number of top chunks to return
            score_threshold: Minimum score threshold for inclusion
            combine_scores: Whether to combine vector and cross-encoder scores
            alpha: Weight for combining scores (alpha * cross_score + (1-alpha) * vector_score)
            
        Returns:
            List of re-ranked chunks with scores
        """
        
        if not retrieved_chunks:
            return []
        
        if self.cross_encoder is None:
            logging.warning("Cross-encoder not available, returning original chunks")
            return retrieved_chunks[:top_n]
        
        try:
            # Prepare question-passage pairs
            pairs = []
            for chunk in retrieved_chunks:
                # Clean and truncate text if too long
                text = chunk.get("text", "")
                # Truncate to reasonable length for cross-encoder (512 tokens ~ 2000 chars)
                if len(text) > 2000:
                    text = text[:2000] + "..."
                pairs.append((user_question, text))
            
            # Compute cross-encoder relevance scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Add scores to chunks
            for chunk, cross_score in zip(retrieved_chunks, cross_scores):
                chunk["cross_score"] = float(cross_score)
                
                # Combine with original vector similarity score if available
                if combine_scores and "score" in chunk:
                    vector_score = chunk["score"]
                    # Normalize scores to 0-1 range if needed
                    normalized_cross = max(0, min(1, (cross_score + 1) / 2))
                    normalized_vector = max(0, min(1, vector_score))
                    
                    # Combined score
                    chunk["combined_score"] = alpha * normalized_cross + (1 - alpha) * normalized_vector
                    sort_key = "combined_score"
                else:
                    sort_key = "cross_score"
            
            # Sort by relevance score (descending)
            ranked_chunks = sorted(retrieved_chunks, 
                                 key=lambda x: x.get(sort_key, 0), 
                                 reverse=True)
            
            # Apply score threshold if specified
            if score_threshold is not None:
                ranked_chunks = [chunk for chunk in ranked_chunks 
                               if chunk.get(sort_key, 0) >= score_threshold]
            
            # Return top-N chunks
            final_chunks = ranked_chunks[:top_n]
            
            # Log re-ranking results
            logging.info(f"Re-ranked {len(retrieved_chunks)} chunks, returning top {len(final_chunks)}")
            if final_chunks:
                top_score = final_chunks[0].get(sort_key, 0)
                logging.info(f"Top chunk score: {top_score:.3f}")
            
            return final_chunks
            
        except Exception as e:
            logging.error(f"Error during re-ranking: {e}")
            # Fallback to original ranking
            return retrieved_chunks[:top_n]

# Medical domain-specific re-ranker
class HSVMedicalReRanker(ReRanker):
    """ Optimized re-ranker specifically for HSV research."""
    
    def __init__(self, model_choice: str = "balanced"):
        """ Initialize with recommended model based on performance needs. model_choice: "best", "balanced", "fast", or specific model name """
        
        model_map = {
            "best": "cross-encoder/ms-marco-electra-base",
            "balanced": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "fast": "cross-encoder/ms-marco-MiniLM-L-4-v2",
            "reasoning": "cross-encoder/nli-deberta-v3-base"
        }
        
        model_name = model_map.get(model_choice, model_choice)
        super().__init__(model_name=model_name)
        
        # HSV-specific keywords for boosting
        self.hsv_keywords = [
            # Virus types
            'hsv-1', 'hsv-2', 'herpes simplex', 'herpes',
            # Treatments
            'acyclovir', 'valacyclovir', 'famciclovir', 'antiviral',
            'suppressive therapy', 'episodic treatment',
            # Clinical terms
            'genital herpes', 'oral herpes', 'cold sores', 'fever blisters',
            'recurrence', 'outbreak', 'lesion', 'vesicle',
            # Medical terms
            'seroprevalence', 'transmission', 'asymptomatic',
            'immunocompromised', 'neonatal herpes',
            # Research terms
            'clinical trial', 'efficacy', 'safety', 'adverse events',
            'placebo', 'randomized', 'double-blind'
            # Immunological terms
            'immune response', 'antibody', 'seroconversion', 'vaccine',
            'immunization', 'T-cell', 'B-cell', 'cytokine', 'lesion',
            'pathogenesis', 'viral shedding', 'latency', 'reactivation',
            'tissue residence', 'dendritic cell', 'macrophage',
            'innate immunity', 'adaptive immunity', 'mucosal immunity',
            'antigen presentation', 'immune evasion', 'host-pathogen interaction',
            'immunopathology', 'immune modulation', 'immune memory',
            'immune surveillance', 'immune tolerance', 'immune response kinetics',
            'immune checkpoint', 'immune therapy', 'immune profiling',
            'immune biomarkers', 'immune system', 'immune dysregulation',
            'immune response genes', 'immune response pathways',
            'immune response mechanisms', 'immune response regulation',
            'immune response modulation', 'immune response enhancement',
            'immune response suppression', 'immune response activation',
            'immune response inhibition', 'immune response signaling',
            'immune response activation', 'immune response inhibition',
            'immune response regulation', 'immune response pathways',
            'immune response genes', 'immune response mechanisms',
            'immune response modulation', 'immune response enhancement',
            'immune response suppression', 'immune response activation',
            'immune response inhibition', 'immune response signaling',
            'immune response regulation', 'immune response pathways',
            'immune response genes', 'immune response mechanisms',
            'immune response modulation', 'immune response enhancement',

        ]
    
    def enhanced_medical_rerank(self, 
                               user_question: str, 
                               retrieved_chunks: List[Dict], 
                               top_n: int = 5) -> List[Dict]:
        """ Enhanced re-ranking with HSV-specific optimizations. """
        
        # Standard re-ranking first
        reranked = self.re_rank_chunks(
            user_question, 
            retrieved_chunks, 
            top_n=min(top_n * 3, len(retrieved_chunks)),  # Get more candidates
            combine_scores=True
        )
        
        # Apply HSV-specific boosting
        question_lower = user_question.lower()
        
        for chunk in reranked:
            text_lower = chunk.get("text", "").lower()
            
            # HSV keyword matching
            hsv_matches = sum(1 for keyword in self.hsv_keywords 
                            if keyword in text_lower)
            
            # Question-text keyword overlap
            question_keywords = set(question_lower.split())
            text_keywords = set(text_lower.split())
            keyword_overlap = len(question_keywords.intersection(text_keywords))
            
            # Section relevance (boost certain sections)
            section_title = chunk.get("section_title", "").lower()
            section_boost = 1.0
            
            if any(term in section_title for term in 
                   ['result', 'discussion', 'conclusion', 'treatment', 'method']):
                section_boost = 1.2
            elif any(term in section_title for term in 
                     ['introduction', 'background', 'reference']):
                section_boost = 0.9
            
            # Calculate composite boost
            hsv_boost = 1 + (hsv_matches * 0.15)  # 15% per HSV keyword
            overlap_boost = 1 + (keyword_overlap * 0.05)  # 5% per overlapping word
            
            total_boost = hsv_boost * overlap_boost * section_boost
            
            # Apply boost to cross score
            original_score = chunk.get("cross_score", 0)
            chunk["cross_score"] = original_score * total_boost
            chunk["hsv_boost_applied"] = total_boost
            
        # Final ranking with boosted scores
        final_ranked = sorted(reranked, 
                            key=lambda x: x.get("cross_score", 0), 
                            reverse=True)
        
        return final_ranked[:top_n]

# Usage functions for your main application
def initialize_reranker(use_medical: bool = True):
    """Initialize the appropriate re-ranker."""
    if use_medical:
        return HSVMedicalReRanker()
    else:
        return ReRanker()

def enhanced_query_with_reranking(user_question: str,
                                  embed_function,
                                  tokenizer,
                                  model,
                                  reranker,
                                  collection_name, 
                                  host, 
                                  port,
                                  initial_top_k: int = 40,
                                  final_top_n: int = 10
                                  ):
    """
    Complete query pipeline with re-ranking.
    
    Args:
        user_question: User's query
        embed_function: Embedding function for vector search
        tokenizer: Tokenizer for the embedding model
        model: Model identifier
        reranker: ReRanker instance
        initial_top_k: Initial retrieval count
        final_top_n: Final number after re-ranking
    
    Returns:
        Re-ranked chunks ready for LLM
    """
    
    # Step 1: Vector similarity search (cast a wide net)
    retrieved_chunks = query_qdrant(
        query_text=user_question,
        embed_function=embed_function,
        tokenizer=tokenizer,
        model=model,
        collection_name=collection_name, 
        host=host, 
        port=port,
        top_k=initial_top_k  # Get more candidates for re-ranking
    )
    
    if not retrieved_chunks:
        return []
    
    # Step 2: Re-rank for semantic relevance
    if isinstance(reranker, HSVMedicalReRanker):
        reranked_chunks = reranker.enhanced_medical_rerank(
            user_question, 
            retrieved_chunks, 
            top_n=final_top_n
        )
    else:
        reranked_chunks = reranker.re_rank_chunks(
            user_question,
            retrieved_chunks,
            top_n=final_top_n,
            combine_scores=True  # Combine vector + cross-encoder scores
        )
    
    return reranked_chunks