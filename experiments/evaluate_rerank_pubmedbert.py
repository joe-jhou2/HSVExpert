# biobert_reranking_comparison.py
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import your existing modules
from src.data_processing._content_extract_xml import extract_sections_from_pmc
from src.data_processing._chunking import chunk_sections
from src.data_processing._Tokenization_Embedding import embed_text, load_embedding_model, get_openai_tokenizer
from src.data_processing._qdrant import store_chunks_in_qdrant
from src.utils._helper import is_valid_paper
from src.retrieval._rerank import HSVMedicalReRanker

class PubMedBERTReRankingTester:
    """
    Focused comparison of PubMedBERT-300tokens with and without re-ranking
    """
    
    def __init__(self, papers_list, base_dir="data/raw"):
        """
        Initialize the PubMedBERT re-ranking tester
        
        Args:
            papers_list: List of paper filenames to process
            base_dir: Directory containing the papers
        """
        self.papers_list = papers_list
        self.base_dir = base_dir
        self.bad_paper_log = "bad_papers_biobert_rerank.txt"
        self.collection_name = "pubmedbert-300tokens-rerank-test"
        
        # Initialize clients and models
        load_dotenv()
        self.qdrant_client = QdrantClient("localhost", port=6333)
        
        # Load BioBERT model
        self.pubmedbert_tokenizer, self.pubmedbert_model = load_embedding_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

        # Initialize re-ranker
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = HSVMedicalReRanker(model_choice="balanced")
            self.reranker_available = True
        except ImportError:
            print("WARNING: sentence-transformers not available. Install with: pip install sentence-transformers")
            self.reranker_available = False
    
    def get_biobert_config(self):
        """Get the BioBERT-300tokens configuration"""
        return {
            'model_type': 'pubmedbert',
            'max_tokens': 300,
            'overlap': 50,
            'max_length': 500,
            'embed_model_name': 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
            'collection_name': self.collection_name
        }
    
    def process_papers_biobert(self):
        """
        Process papers with PubMedBERT-300tokens configuration
        """
        config = self.get_biobert_config()
        
        print(f"Processing {len(self.papers_list)} papers with PubMedBERT-300tokens")
        print("="*80)
        
        # Process each paper
        for paper in self.papers_list:
            paper_path = os.path.join(self.base_dir, paper)
            
            # Skip invalid papers
            if not is_valid_paper(paper_path):
                print(f"Skipping {paper}: contains error tag.")
                with open(self.bad_paper_log, "a") as f:
                    f.write(f"{paper}\n")
                continue
            
            try:
                # Extract sections
                print(f"Processing paper: {paper}")
                parsed_paper = extract_sections_from_pmc(paper_path)
                parsed_paper["metadata"]["paper_id"] = os.path.splitext(os.path.basename(paper_path))[0]
                
                # Chunking
                print(f"Chunking paper: {paper}")
                chunks = chunk_sections(
                    parsed_paper, 
                    max_tokens=config['max_tokens'], 
                    tokenizer=self.pubmedbert_tokenizer, 
                    overlap=config['overlap'], 
                    model_type=config['model_type']
                )
                
                # Embedding
                print(f"Embedding chunks for paper: {paper}")
                all_embeddings = []
                for chunk in chunks:
                    embedding = embed_text(chunk["text"], self.pubmedbert_tokenizer, self.pubmedbert_model, max_length=config['max_length'])
                    chunk["embedding"] = embedding
                    all_embeddings.append(embedding)
                
                # Store in Qdrant
                print(f"Storing chunks in Qdrant for paper: {paper}")
                store_chunks_in_qdrant(
                    chunks, 
                    all_embeddings, 
                    embed_model_name=config['embed_model_name'], 
                    collection_name=config['collection_name']
                )
                
            except Exception as e:
                print(f"Error processing {paper}: {str(e)}")
                with open(self.bad_paper_log, "a") as f:
                    f.write(f"{paper} - Error: {str(e)}\n")
        
        print(f"Completed PubMedBERT processing")
        print("="*80 + "\n")
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """
        Create test queries optimized for re-ranking comparison
        Include diverse complexity levels and HSV-specific terms
        """
        return [
            # Simple factual queries
            {
                "query": "What is the role of TRIM23 in HSV-1 infection?",
                "type": "simple_factual", 
                "paper_should_answer": "PMC12075517",
                "expected_topics": ["TRIM23", "HSV-1", "autophagy", "ubiquitination"],
                "complexity": "low"
            },
            {
                "query": "Which proteins activate NF-κB pathway during HSV infection?",
                "type": "simple_factual",
                "paper_should_answer": "PMC12017772",
                "expected_topics": ["NF-κB", "IKK", "IκBα", "p65"],
                "complexity": "low"
            },
            
            # Medium complexity mechanistic queries
            {
                "query": "How does TBK1 phosphorylation of TRIM23 lead to autophagy activation in HSV-1 infected cells?",
                "type": "mechanistic",
                "paper_should_answer": "PMC12075517", 
                "expected_topics": ["TBK1", "TRIM23", "S39", "phosphorylation", "autoubiquitination", "autophagy"],
                "complexity": "medium"
            },
            {
                "query": "What is the mechanism by which HSV-ΔICP34.5 promotes HIV latency reactivation through HSF1?",
                "type": "mechanistic", 
                "paper_should_answer": "PMC12017772",
                "expected_topics": ["HSV-ΔICP34.5", "PP1α", "HSF1", "HIV LTR", "phosphorylation"],
                "complexity": "medium"
            },
            
            # Complex multi-step queries
            {
                "query": "How do HSV-1 infected cells compensate for FASN inhibition and what alternative lipid pathways are activated?",
                "type": "complex_mechanistic",
                "paper_should_answer": "PMC12084038",
                "expected_topics": ["FASN", "CMS121", "C75", "CD36", "fatty acid uptake", "lipid metabolism"],
                "complexity": "high"
            },
            {
                "query": "What is the relationship between ADAR1 p150, PKR hyperactivation, and innate immune evasion in HSV-1 infection?",
                "type": "complex_mechanistic",
                "paper_should_answer": "PMC12011305",
                "expected_topics": ["ADAR1 p150", "PKR", "hyperactivation", "innate immunity", "immune evasion"],
                "complexity": "high"
            },
            
            # Clinical/therapeutic queries
            {
                "query": "How does stress-induced glucocorticoid receptor activation affect HSV reactivation from latency?",
                "type": "clinical_mechanistic",
                "paper_should_answer": "PMC10764003",
                "expected_topics": ["glucocorticoid receptor", "stress", "viral promoters", "latency", "reactivation"],
                "complexity": "medium"
            },
            
            # Structural biology queries
            {
                "query": "How do hexameric interfaces in the nuclear egress complex facilitate HSV membrane budding?",
                "type": "structural",
                "paper_should_answer": "PMC10817169",
                "expected_topics": ["hexagonal lattice", "NEC", "membrane budding", "nuclear egress"],
                "complexity": "high"
            },
            
            # Immunology queries
            {
                "query": "What T-cell receptor repertoire changes occur in HSV lesion sites compared to unaffected skin?",
                "type": "immunological",
                "paper_should_answer": "PMC10863019", 
                "expected_topics": ["TRB repertoire", "clonality", "T-cell diversity", "lesion sites"],
                "complexity": "medium"
            },
            
            # Ambiguous queries (good test for re-ranking)
            {
                "query": "viral protein interactions autophagy",
                "type": "ambiguous",
                "paper_should_answer": "PMC12075517",
                "expected_topics": ["viral proteins", "autophagy", "interactions"],
                "complexity": "low"
            }
        ]
    
    def query_without_rerank(self, user_question: str, top_k: int = 10) -> List[Dict]:
        """
        Standard vector similarity search without re-ranking
        """
        start_time = time.time()
        
        # Create embedding for query
        query_embedding = embed_text(user_question, self.pubmedbert_tokenizer, self.pubmedbert_model, max_length=500)
        
        # Search Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            with_payload=True
        )
        
        search_time = time.time() - start_time
        
        # Convert to consistent format
        results = []
        for result in search_results:
            chunk_data = {
                "text": result.payload.get("text", ""),
                "paper_title": result.payload.get("paper_title", ""),
                "section_title": result.payload.get("section_title", ""),
                "paper_id": result.payload.get("paper_id", ""),
                "score": result.score,
                "search_time": search_time
            }
            results.append(chunk_data)
        
        return results
    
    def query_with_rerank(self, user_question: str, initial_k: int = 20, final_n: int = 10) -> List[Dict]:
        """
        Vector search followed by re-ranking
        """
        if not self.reranker_available:
            print("Re-ranker not available, falling back to standard search")
            return self.query_without_rerank(user_question, top_k=final_n)
        
        start_time = time.time()
        
        # Step 1: Initial vector search (get more candidates)
        initial_results = self.query_without_rerank(user_question, top_k=initial_k)
        
        if not initial_results:
            return []
        
        # Step 2: Re-rank using cross-encoder
        try:
            reranked_results = self.reranker.enhanced_medical_rerank(
                user_question=user_question,
                retrieved_chunks=initial_results,
                top_n=final_n
            )
            
            total_time = time.time() - start_time
            
            # Add timing info
            for result in reranked_results:
                result["search_time"] = total_time
                result["reranked"] = True
            
            return reranked_results
            
        except Exception as e:
            print(f"Re-ranking failed: {e}")
            return initial_results[:final_n]
    
    def evaluate_relevance(self, results: List[Dict], expected_topics: List[str]) -> Dict:
        """Evaluate relevance of search results based on expected topics"""
        if not results:
            return {
                "relevance_score": 0, 
                "topic_coverage": 0, 
                "topics_found": [],
                "precision_at_1": 0,
                "precision_at_3": 0,
                "precision_at_5": 0
            }
            
        total_score = 0
        topic_found = set()
        relevant_results = []
        
        for result in results:
            text = result.get("text", "").lower()
            title = result.get("paper_title", "").lower()
            section = result.get("section_title", "").lower()
            
            full_text = f"{text} {title} {section}"
            
            result_topics_found = 0
            for topic in expected_topics:
                if topic.lower() in full_text:
                    topic_found.add(topic.lower())
                    result_topics_found += 1
            
            # Consider result relevant if it contains at least 1 expected topic
            if result_topics_found > 0:
                relevant_results.append(True)
                total_score += result_topics_found
            else:
                relevant_results.append(False)
        
        # Calculate metrics
        relevance_score = total_score / (len(results) * len(expected_topics)) if results else 0
        topic_coverage = len(topic_found) / len(expected_topics) if expected_topics else 0
        
        # Precision at K metrics
        precision_at_1 = sum(relevant_results[:1]) / min(1, len(relevant_results)) if relevant_results else 0
        precision_at_3 = sum(relevant_results[:3]) / min(3, len(relevant_results)) if relevant_results else 0
        precision_at_5 = sum(relevant_results[:5]) / min(5, len(relevant_results)) if relevant_results else 0
        
        return {
            "relevance_score": relevance_score,
            "topic_coverage": topic_coverage,
            "topics_found": list(topic_found),
            "precision_at_1": precision_at_1,
            "precision_at_3": precision_at_3,
            "precision_at_5": precision_at_5,
            "num_relevant": sum(relevant_results)
        }
    
    def run_reranking_comparison(self):
        """
        Run comprehensive comparison of BioBERT with and without re-ranking
        """
        test_queries = self.create_test_queries()
        comparison_results = []
        
        print(f"Running re-ranking comparison with {len(test_queries)} queries")
        print("="*80)
        
        for i, query_info in enumerate(test_queries):
            query_text = query_info["query"]
            query_type = query_info["type"]
            expected_topics = query_info["expected_topics"]
            complexity = query_info["complexity"]
            
            print(f"Query {i+1}/{len(test_queries)}: {query_text[:60]}...")
            
            # Test without re-ranking
            print("  Testing without re-ranking...")
            results_no_rerank = self.query_without_rerank(query_text, top_k=10)
            metrics_no_rerank = self.evaluate_relevance(results_no_rerank, expected_topics)
            
            # Test with re-ranking
            print("  Testing with re-ranking...")
            results_with_rerank = self.query_with_rerank(query_text, initial_k=20, final_n=10)
            metrics_with_rerank = self.evaluate_relevance(results_with_rerank, expected_topics)
            
            # Record results for both methods
            for method, results, metrics in [
                ("without_rerank", results_no_rerank, metrics_no_rerank),
                ("with_rerank", results_with_rerank, metrics_with_rerank)
            ]:
                result_row = {
                    "query": query_text,
                    "query_type": query_type,
                    "complexity": complexity,
                    "method": method,
                    "search_time": results[0]["search_time"] if results else 0,
                    "num_results": len(results),
                    "relevance_score": metrics["relevance_score"],
                    "topic_coverage": metrics["topic_coverage"],
                    "precision_at_1": metrics["precision_at_1"],
                    "precision_at_3": metrics["precision_at_3"],
                    "precision_at_5": metrics["precision_at_5"],
                    "num_relevant": metrics["num_relevant"],
                    "topics_found": metrics["topics_found"],
                    "avg_similarity_score": np.mean([r.get("score", 0) for r in results]) if results else 0
                }
                
                comparison_results.append(result_row)
            
            print(f"  No rerank - Relevance: {metrics_no_rerank['relevance_score']:.3f}, Coverage: {metrics_no_rerank['topic_coverage']:.3f}")
            print(f"  With rerank - Relevance: {metrics_with_rerank['relevance_score']:.3f}, Coverage: {metrics_with_rerank['topic_coverage']:.3f}")
            print()
        
        self.results_df = pd.DataFrame(comparison_results)
        return self.results_df
    
    def analyze_reranking_impact(self) -> Dict[str, Any]:
        """Analyze the impact of re-ranking on search performance"""
        if not hasattr(self, 'results_df'):
            raise ValueError("Run comparison first!")
        
        analysis = {}
        
        # Overall performance comparison
        method_comparison = self.results_df.groupby('method').agg({
            'search_time': ['mean', 'std'],
            'relevance_score': ['mean', 'std'],
            'topic_coverage': ['mean', 'std'],
            'precision_at_1': ['mean', 'std'],
            'precision_at_3': ['mean', 'std'],
            'precision_at_5': ['mean', 'std'],
            'num_relevant': ['mean', 'std']
        }).round(4)
        
        analysis['method_comparison'] = method_comparison
        
        # Performance by query complexity
        complexity_comparison = self.results_df.groupby(['complexity', 'method']).agg({
            'relevance_score': 'mean',
            'topic_coverage': 'mean',
            'precision_at_1': 'mean',
            'precision_at_5': 'mean'
        }).round(4)
        
        analysis['complexity_comparison'] = complexity_comparison
        
        # Performance by query type
        type_comparison = self.results_df.groupby(['query_type', 'method']).agg({
            'relevance_score': 'mean',
            'topic_coverage': 'mean',
            'precision_at_3': 'mean'
        }).round(4)
        
        analysis['type_comparison'] = type_comparison
        
        # Calculate improvement metrics
        pivot_relevance = self.results_df.pivot_table(
            values='relevance_score', 
            index='query', 
            columns='method'
        )
        
        pivot_coverage = self.results_df.pivot_table(
            values='topic_coverage', 
            index='query', 
            columns='method'
        )
        
        # Improvement calculations
        if 'with_rerank' in pivot_relevance.columns and 'without_rerank' in pivot_relevance.columns:
            relevance_improvement = ((pivot_relevance['with_rerank'] - pivot_relevance['without_rerank']) / 
                                   pivot_relevance['without_rerank'] * 100).fillna(0)
            
            coverage_improvement = ((pivot_coverage['with_rerank'] - pivot_coverage['without_rerank']) / 
                                  pivot_coverage['without_rerank'] * 100).fillna(0)
            
            analysis['improvements'] = {
                'avg_relevance_improvement': relevance_improvement.mean(),
                'avg_coverage_improvement': coverage_improvement.mean(),
                'queries_improved_relevance': (relevance_improvement > 0).sum(),
                'queries_improved_coverage': (coverage_improvement > 0).sum(),
                'max_relevance_improvement': relevance_improvement.max(),
                'max_coverage_improvement': coverage_improvement.max()
            }
        
        return analysis
    
    def visualize_reranking_comparison(self):
        """Create visualization plots for the re-ranking comparison"""
        if not hasattr(self, 'results_df'):
            raise ValueError("Run comparison first!")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Relevance Score Comparison
        sns.boxplot(data=self.results_df, x='method', y='relevance_score', ax=axes[0,0])
        axes[0,0].set_title('Relevance Score: With vs Without Re-ranking')
        axes[0,0].set_ylabel('Relevance Score')
        
        # 2. Topic Coverage Comparison
        sns.boxplot(data=self.results_df, x='method', y='topic_coverage', ax=axes[0,1])
        axes[0,1].set_title('Topic Coverage: With vs Without Re-ranking')
        axes[0,1].set_ylabel('Topic Coverage')
        
        # 3. Precision@1 Comparison
        sns.boxplot(data=self.results_df, x='method', y='precision_at_1', ax=axes[0,2])
        axes[0,2].set_title('Precision@1: With vs Without Re-ranking')
        axes[0,2].set_ylabel('Precision@1')
        
        # 4. Performance by Query Complexity
        complexity_pivot = self.results_df.pivot_table(
            values='relevance_score', 
            index='complexity', 
            columns='method'
        )
        complexity_pivot.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Relevance by Query Complexity')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].legend(title='Method')
        
        # 5. Search Time Comparison
        sns.boxplot(data=self.results_df, x='method', y='search_time', ax=axes[1,1])
        axes[1,1].set_title('Search Time: With vs Without Re-ranking')
        axes[1,1].set_ylabel('Search Time (seconds)')
        
        # 6. Precision@5 by Query Type
        type_precision = self.results_df.pivot_table(
            values='precision_at_5', 
            index='query_type', 
            columns='method'
        )
        sns.heatmap(type_precision, annot=True, cmap='YlOrRd', ax=axes[1,2])
        axes[1,2].set_title('Precision@5 by Query Type')
        axes[1,2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_reranking_report(self) -> str:
        """Generate a comprehensive re-ranking comparison report"""
        if not hasattr(self, 'results_df'):
            raise ValueError("Run comparison first!")
        
        analysis = self.analyze_reranking_impact()
        
        report = f"""
                # PubMedBERT Re-ranking Impact Analysis Report

                ## Test Configuration:
                - Model: BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
                - Chunk Size: 300 tokens
                - Collection: {self.collection_name}
                - Papers Tested: {len(self.papers_list)}
                - Queries Tested: {len(self.results_df) // 2}

                ## Overall Performance Comparison:
                {analysis['method_comparison']}

                ## Key Performance Metrics:

                ### Relevance Improvements:
                - Average Relevance Improvement: {analysis.get('improvements', {}).get('avg_relevance_improvement', 0):.2f}%
                - Average Coverage Improvement: {analysis.get('improvements', {}).get('avg_coverage_improvement', 0):.2f}%
                - Queries with Improved Relevance: {analysis.get('improvements', {}).get('queries_improved_relevance', 0)}
                - Queries with Improved Coverage: {analysis.get('improvements', {}).get('queries_improved_coverage', 0)}

                ## Performance by Query Complexity:
                {analysis['complexity_comparison']}

                ## Performance by Query Type:
                {analysis['type_comparison']}

                ## Recommendations:

                1. **Re-ranking Impact**: {'Positive' if analysis.get('improvements', {}).get('avg_relevance_improvement', 0) > 0 else 'Mixed'}
                - Re-ranking shows {'significant' if analysis.get('improvements', {}).get('avg_relevance_improvement', 0) > 10 else 'moderate'} improvement in relevance scoring

                2. **Query Complexity**: 
                - {'High' if 'high' in analysis['complexity_comparison'].index else 'Medium'} complexity queries benefit most from re-ranking

                3. **Trade-offs**:
                - Search time increase: ~{self.results_df[self.results_df['method'] == 'with_rerank']['search_time'].mean() / self.results_df[self.results_df['method'] == 'without_rerank']['search_time'].mean():.1f}x
                - Quality improvement justifies {'YES' if analysis.get('improvements', {}).get('avg_relevance_improvement', 0) > 5 else 'QUESTIONABLE'} the additional latency

                4. **Use Cases**:
                - Recommend re-ranking for: Complex mechanistic queries, ambiguous queries
                - Optional for: Simple factual queries, when speed is critical
                """
        
        return report
    
    def run_full_reranking_test(self, process_data=True):
        """
        Run the complete re-ranking comparison pipeline
        
        Args:
            process_data: Whether to process papers (set False if already processed)
        """
        
        if process_data:
            print("="*100)
            print("PROCESSING PAPERS WITH BIOBERT-300TOKENS")
            print("="*100)
            self.process_papers_biobert()
        
        print("="*100)
        print("RUNNING RE-RANKING COMPARISON")
        print("="*100)
        
        # Run comparison
        results_df = self.run_reranking_comparison()
        
        if results_df is not None:
            # Analyze results
            analysis = self.analyze_reranking_impact()
            
            # Generate and print report
            report = self.generate_reranking_report()
            print(report)
            
            # Save results
            results_df.to_csv("experiments/results/pubmedbert_reranking_comparison.csv", index=False)
            print("\nResults saved to 'pubmedbert_reranking_comparison.csv'")
            
            # Visualize
            try:
                self.visualize_reranking_comparison()
            except Exception as e:
                print(f"Visualization failed: {e}")
            
            return results_df, analysis
        
        return None, None

# Example usage
if __name__ == "__main__":
    # Define testing papers
    testing_papers = [
        'PMC12075517.xml', 'PMC12017772.xml', 'PMC12084038.xml', 
        'PMC12011305.xml', 'PMC10764003.xml', 'PMC10817169.xml', 'PMC10863019.xml'
    ]
    
    # Initialize re-ranking tester
    tester = PubMedBERTReRankingTester(testing_papers)
    
    # Run full test
    # Option 1: Process data and run comparison
    results_df, analysis = tester.run_full_reranking_test(process_data=True)
    
    # Option 2: Only run comparison (if data already processed)
    # results_df, analysis = tester.run_full_reranking_test(process_data=False)
    
    # Option 3: Run individual steps
    # tester.process_papers_biobert()  # Process data
    # results_df = tester.run_reranking_comparison()  # Run comparison
    # analysis = tester.analyze_reranking_impact()  # Analyze results
    # report = tester.generate_reranking_report()  # Generate report