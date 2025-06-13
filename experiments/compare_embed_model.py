
import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import your existing modules
from src.data_processing._content_extract_xml import extract_sections_from_pmc
from src.data_processing._chunking import chunk_sections
from src.data_processing._Tokenization_Embedding import embed_text, load_embedding_model
from src.data_processing._qdrant import store_chunks_in_qdrant
from src.utils._helper import is_valid_paper
from src.data_processing._Tokenization_Embedding import get_openai_tokenizer, embed_openai

class ComprehensivePipelineTester:
    """
    Complete pipeline for processing papers with different embedding methods
    and comparing their performance
    """
    
    def __init__(self, papers_list: List[str], base_dir: str = "data/raw"):
        """
        Initialize the pipeline tester
        
        Args:
            papers_list: List of paper filenames to process
            base_dir: Directory containing the papers
        """
        self.papers_list = papers_list
        self.base_dir = base_dir
        self.bad_paper_log = "bad_papers.txt"
        self.processed_collections = []
        
        # Initialize clients and models
        self._initialize_clients_and_models()
    
    def _initialize_clients_and_models(self):
        """Initialize all clients and models with error handling"""
        try:
            load_dotenv()
            
            # Initialize OpenAI client
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("WARNING: OPENAI_API_KEY not found in environment variables")
                self.openai_client = None
            else:
                self.openai_client = OpenAI(api_key=api_key)
            
            # Initialize Qdrant client
            try:
                self.qdrant_client = QdrantClient("localhost", port=6333)
                # Test connection
                self.qdrant_client.get_collections()
                print("✓ Connected to Qdrant successfully")
            except Exception as e:
                print(f"✗ Failed to connect to Qdrant: {e}")
                raise
            
            # Load embedding models
            try:
                self.bert_tokenizer, self.bert_model = load_embedding_model("bert-base-uncased")
                print("✓ BERT model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load BERT model: {e}")
                self.bert_tokenizer, self.bert_model = None, None
            
            try:
                self.pubmedbert_tokenizer, self.pubmedbert_model = load_embedding_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
                print("✓ PubMedBERT model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load BERT model: {e}")
                self.pubmedbert_tokenizer, self.pubmedbert_model = None, None

            try:
                self.biobert_tokenizer, self.biobert_model = load_embedding_model("dmis-lab/biobert-base-cased-v1.2")
                print("✓ BioBERT model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load BioBERT model: {e}")
                self.biobert_tokenizer, self.biobert_model = None, None
            
            try:
                self.openai_tokenizer = get_openai_tokenizer()
                print("✓ OpenAI tokenizer loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load OpenAI tokenizer: {e}")
                self.openai_tokenizer = None
                
        except Exception as e:
            print(f"Error initializing clients and models: {e}")
            raise
    
    def get_embedding_configs(self) -> List[Dict[str, Any]]:
        """Define all embedding configurations to test"""
        configs = []
        
        # BERT configurations
        if self.bert_tokenizer and self.bert_model:
            configs.extend([
                {
                    'model_type': 'bert',
                    'max_tokens': 500,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'bert-base-uncased',
                    'collection_name': 'bert-500tokens'
                },
                {
                    'model_type': 'bert',
                    'max_tokens': 300,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'bert-base-uncased',
                    'collection_name': 'bert-300tokens'
                }
            ])
        
        # PubMedBERT configurations
        if self.pubmedbert_tokenizer and self.pubmedbert_model:
            configs.extend([
                {
                    'model_type': 'pubmedbert',
                    'max_tokens': 500,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'collection_name': 'pubmedbert-500tokens'
                },
                {
                    'model_type': 'pubmedbert',
                    'max_tokens': 300,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                    'collection_name': 'pubmedbert-300tokens'
                }
            ])

        # BioBERT configurations
        if self.biobert_tokenizer and self.biobert_model:
            configs.extend([
                {
                    'model_type': 'biobert',
                    'max_tokens': 500,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'biobert-base-cased-v1.2',
                    'collection_name': 'biobert-500tokens'
                },
                {
                    'model_type': 'biobert',
                    'max_tokens': 300,
                    'overlap': 50,
                    'max_length': 500,
                    'embed_model_name': 'biobert-base-cased-v1.2',
                    'collection_name': 'biobert-300tokens'
                }
            ])
        
        # OpenAI configurations
        if self.openai_client and self.openai_tokenizer:
            configs.extend([
                {
                    'model_type': 'openai',
                    'max_tokens': 1000,
                    'overlap': 100,
                    'embed_model_name': 'text-embedding-3-large',
                    'collection_name': 'openai-large-1000tokens',
                    'openai_model': 'text-embedding-3-large'
                },
                {
                    'model_type': 'openai',
                    'max_tokens': 500,
                    'overlap': 50,
                    'embed_model_name': 'text-embedding-3-large',
                    'collection_name': 'openai-large-500tokens',
                    'openai_model': 'text-embedding-3-large'
                },
                {
                    'model_type': 'openai',
                    'max_tokens': 300,
                    'overlap': 50,
                    'embed_model_name': 'text-embedding-3-large',
                    'collection_name': 'openai-large-300tokens',
                    'openai_model': 'text-embedding-3-large'
                }
            ])
        
        if not configs:
            raise RuntimeError("No valid embedding configurations available. Check model initialization.")
        
        return configs
    
    def process_papers_with_config(self, config: Dict[str, Any]) -> bool:
        """Process papers with a specific embedding configuration"""
        
        print(f"\nProcessing {len(self.papers_list)} papers with config: {config['model_type']}-{config['max_tokens']}tokens")
        print("="*80)
        
        try:
            # Initialize models and tokenizers based on config
            if config['model_type'] == 'bert':
                if not (self.bert_tokenizer and self.bert_model):
                    print(f"SKIPPING: BERT models not available")
                    return False
                tokenizer, model = self.bert_tokenizer, self.bert_model
                embed_func = lambda text: embed_text(text, tokenizer, model, max_length=config['max_length'])
            
            elif config['model_type'] == 'pubmedbert':
                if not (self.pubmedbert_tokenizer and self.pubmedbert_model):
                    print(f"SKIPPING: PubMedBERT models not available")
                    return False
                tokenizer, model = self.pubmedbert_tokenizer, self.pubmedbert_model
                embed_func = lambda text: embed_text(text, tokenizer, model, max_length=config['max_length'])

            elif config['model_type'] == 'biobert':
                if not (self.biobert_tokenizer and self.biobert_model):
                    print(f"SKIPPING: BioBERT models not available")
                    return False
                tokenizer, model = self.biobert_tokenizer, self.biobert_model
                embed_func = lambda text: embed_text(text, tokenizer, model, max_length=config['max_length'])
                
            elif config['model_type'] == 'openai':
                if not (self.openai_client and self.openai_tokenizer):
                    print(f"SKIPPING: OpenAI client/tokenizer not available")
                    return False
                tokenizer = self.openai_tokenizer
                embed_func = lambda text: embed_openai(text, self.openai_client, model=config.get('openai_model', 'text-embedding-3-large'))
            else:
                raise ValueError(f"Unsupported model_type: {config['model_type']}")
            
            # Process each paper
            successful_papers = 0
            failed_papers = 0
            
            for paper in self.papers_list:
                paper_path = os.path.join(self.base_dir, paper)
                
                # Check if paper file exists
                if not os.path.exists(paper_path):
                    print(f"ERROR: Paper file not found: {paper_path}")
                    failed_papers += 1
                    continue
                
                # Skip invalid papers (only for BERT/BioBERT)
                if config['model_type'] in ['bert', 'biobert'] and not is_valid_paper(paper_path):
                    print(f"Skipping {paper}: contains error tag.")
                    with open(self.bad_paper_log, "a") as f:
                        f.write(f"{paper} - Invalid paper (error tag)\n")
                    failed_papers += 1
                    continue
                
                try:
                    # Extract sections
                    print(f"  Processing paper: {paper}")
                    parsed_paper = extract_sections_from_pmc(paper_path)
                    
                    if not parsed_paper or not parsed_paper.get("sections"):
                        print(f"  WARNING: No sections extracted from {paper}")
                        failed_papers += 1
                        continue
                    
                    parsed_paper["metadata"]["paper_id"] = os.path.splitext(os.path.basename(paper_path))[0]
                    
                    # Chunking
                    print(f"  Chunking paper: {paper}")
                    chunks = chunk_sections(
                        parsed_paper, 
                        max_tokens=config['max_tokens'], 
                        tokenizer=tokenizer, 
                        overlap=config['overlap'], 
                        model_type=config['model_type']
                    )
                    
                    if not chunks:
                        print(f"  WARNING: No chunks created for {paper}")
                        failed_papers += 1
                        continue
                    
                    print(f"  Created {len(chunks)} chunks")
                    
                    # Embedding
                    print(f"  Embedding chunks for paper: {paper}")
                    all_embeddings = []
                    chunk_count = 0
                    
                    for i, chunk in enumerate(chunks):
                        try:
                            embedding = embed_func(chunk["text"])
                            if embedding is not None:
                                chunk["embedding"] = embedding
                                all_embeddings.append(embedding)
                                chunk_count += 1
                            else:
                                print(f"    WARNING: Failed to embed chunk {i}")
                        except Exception as e:
                            print(f"    ERROR embedding chunk {i}: {e}")
                            continue
                    
                    if not all_embeddings:
                        print(f"  ERROR: No embeddings created for {paper}")
                        failed_papers += 1
                        continue
                    
                    print(f"  Successfully embedded {chunk_count}/{len(chunks)} chunks")
                    
                    # Store in Qdrant
                    print(f"  Storing chunks in Qdrant for paper: {paper}")
                    store_chunks_in_qdrant(
                        [chunk for chunk in chunks if "embedding" in chunk], 
                        all_embeddings, 
                        embed_model_name=config['embed_model_name'], 
                        collection_name=config['collection_name']
                    )
                    
                    successful_papers += 1
                    print(f"  ✓ Successfully processed {paper}")
                    
                except Exception as e:
                    print(f"  ✗ Error processing {paper}: {str(e)}")
                    with open(self.bad_paper_log, "a") as f:
                        f.write(f"{paper} - Error: {str(e)}\n")
                    failed_papers += 1
            
            # Add to processed collections for comparison
            if successful_papers > 0:
                self.processed_collections.append(config['collection_name'])
                print(f"\n✓ Completed processing with config: {config['model_type']}-{config['max_tokens']}tokens")
                print(f"  Successfully processed: {successful_papers}/{len(self.papers_list)} papers")
                if failed_papers > 0:
                    print(f"  Failed papers: {failed_papers}")
                return True
            else:
                print(f"\n✗ Failed to process any papers with config: {config['model_type']}-{config['max_tokens']}tokens")
                return False
                
        except Exception as e:
            print(f"✗ Fatal error in config {config['model_type']}-{config['max_tokens']}tokens: {e}")
            return False
        finally:
            print("="*80 + "\n")
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create test queries for HSV research evaluation"""
        return [
            {
                "query": "What phosphorylation site on TRIM23 is targeted by TBK1 during HSV-1 infection?",
                "type": "factual_comprehension", 
                "paper_should_answer": "PMC12075517",
                "expected_topics": ["TRIM23", "TBK1", "S39", "phosphorylation"]
            },
            {
                "query": "How does TBK1-mediated TRIM23 phosphorylation lead to autophagy activation?",
                "type": "mechanistic",
                "paper_should_answer": "PMC12075517", 
                "expected_topics": ["TBK1", "TRIM23", "autoubiquitination", "GTPase activity", "autophagy"]
            },
            {
                "query": "Which proteins are involved in the NF-κB pathway activation by HSV-ΔICP34.5?",
                "type": "factual_comprehension",
                "paper_should_answer": "PMC12017772",
                "expected_topics": ["IKKα/β", "IκBα", "p65", "NF-κB"]
            },
            {
                "query": "How does HSV-ΔICP34.5 promote HSF1 phosphorylation to reactivate HIV latency?",
                "type": "mechanistic", 
                "paper_should_answer": "PMC12017772",
                "expected_topics": ["HSV-ΔICP34.5", "PP1α", "HSF1", "HIV LTR", "phosphorylation"]
            },
            {
                "query": "Which inhibitors were used to disrupt FASN in the HSV-1 lipid metabolism study?",
                "type": "factual_comprehension",
                "paper_should_answer": "PMC12084038", 
                "expected_topics": ["FASN", "CMS121", "C75", "inhibitors"]
            },
            {
                "query": "How do external lipid sources compensate for FASN depletion in HSV-1 infected cells?",
                "type": "mechanistic",
                "paper_should_answer": "PMC12084038",
                "expected_topics": ["CD36", "fatty acid uptake", "FASN depletion", "compensation"]
            },
            {
                "query": "What happens to PKR in ADAR1-deficient cells infected with HSV-1?",
                "type": "factual_comprehension",
                "paper_should_answer": "PMC12011305",
                "expected_topics": ["PKR", "hyperactivation", "ADAR1-deficient", "HSV-1"]
            },
            {
                "query": "How does ADAR1 p150 prevent PKR hyperactivation during HSV-1 infection?",
                "type": "mechanistic",
                "paper_should_answer": "PMC12011305", 
                "expected_topics": ["ADAR1 p150", "PKR interaction", "activation threshold", "innate immunity"]
            },
            {
                "query": "Which viral proteins are stimulated by GR and stress-induced transcription factors?",
                "type": "factual_comprehension",
                "paper_should_answer": "PMC10764003",
                "expected_topics": ["ICP0", "ICP4", "ICP27", "VP16", "GR"]
            },
            {
                "query": "How does stress-mediated GR activation accelerate HSV reactivation from latency?",
                "type": "mechanistic", 
                "paper_should_answer": "PMC10764003",
                "expected_topics": ["glucocorticoid receptor", "viral promoters", "immune suppression", "latency"]
            }
        ]
    
    def safe_embedding_wrapper(self, original_embed_func, model_name: str):
        """Wrapper to handle embedding function errors"""
        def wrapped_embed_func(text):
            try:
                if not text or not text.strip():
                    print(f"WARNING: Empty text provided to {model_name}")
                    return None
                    
                embedding = original_embed_func(text)
                
                if embedding is None:
                    print(f"WARNING: {model_name} returned None embedding")
                    return None
                    
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                    
                # Validate embedding shape
                if embedding.size == 0 or len(embedding.shape) == 0:
                    print(f"WARNING: {model_name} returned empty or invalid embedding")
                    return None
                    
                return embedding
                
            except Exception as e:
                print(f"ERROR in {model_name}: {e}")
                return None
        
        return wrapped_embed_func
    
    def search_collection(self, collection: str, query_text: str, query_embedding: np.ndarray, top_k: int = 5) -> Dict:
        """Search a specific collection and return results with metadata"""
        
        if query_embedding is None:
            print(f"ERROR: query_embedding is None for query: '{query_text[:50]}...' in collection: '{collection}'")
            return {
                "collection": collection,
                "query": query_text,
                "search_time": 0,
                "results": [],
                "scores": [],
                "avg_score": 0
            }
        
        try:
            # Convert to list if it's a numpy array or tensor
            if hasattr(query_embedding, 'tolist'):
                query_vector = query_embedding.tolist()
            elif hasattr(query_embedding, 'numpy'):
                query_vector = query_embedding.numpy().tolist()
            else:
                query_vector = list(query_embedding)
            
            start_time = time.time()
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection not in collection_names:
                print(f"ERROR: Collection '{collection}' does not exist")
                return {
                    "collection": collection,
                    "query": query_text,
                    "search_time": 0,
                    "results": [],
                    "scores": [],
                    "avg_score": 0
                }
            
            search_results = self.qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            
            search_time = time.time() - start_time
            
            return {
                "collection": collection,
                "query": query_text,
                "search_time": search_time,
                "results": search_results,
                "scores": [result.score for result in search_results],
                "avg_score": np.mean([result.score for result in search_results]) if search_results else 0
            }
            
        except Exception as e:
            print(f"Error in search_collection for {collection}: {e}")
            return {
                "collection": collection,
                "query": query_text,
                "search_time": 0,
                "results": [],
                "scores": [],
                "avg_score": 0
            }
    
    def evaluate_relevance(self, results: List, expected_topics: List[str]) -> Dict:
        """Evaluate relevance of search results based on expected topics"""
        if not results:
            return {"relevance_score": 0, "topic_coverage": 0, "topics_found": []}
            
        total_score = 0
        topic_found = set()
        
        for result in results:
            text = result.payload.get("text", "").lower()
            title = result.payload.get("paper_title", "").lower()
            section = result.payload.get("section_title", "").lower()
            
            full_text = f"{text} {title} {section}"
            
            for topic in expected_topics:
                if topic.lower() in full_text:
                    topic_found.add(topic.lower())
                    total_score += 1
        
        relevance_score = total_score / (len(results) * len(expected_topics)) if results and expected_topics else 0
        topic_coverage = len(topic_found) / len(expected_topics) if expected_topics else 0
        
        return {
            "relevance_score": relevance_score,
            "topic_coverage": topic_coverage,
            "topics_found": list(topic_found)
        }
    
    def create_bert_embedding_func(self, tokenizer, model):
        """Create a proper embedding function with correct closure"""
        def embed_func(text):
            return embed_text(text, tokenizer, model, max_length=500)
        return embed_func

    def create_openai_embedding_func(self, client, model_name):
        """Create a proper OpenAI embedding function"""
        def embed_func(text):
            return embed_openai(text, client, model=model_name)
        return embed_func

    def run_comparison(self, collections_to_compare: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Run comparison analysis across collections"""
        if collections_to_compare is None:
            collections_to_compare = self.processed_collections
        
        if not collections_to_compare:
            print("ERROR: No collections to compare. Process papers first.")
            return None
        
        print(f"Available collections: {collections_to_compare}")
        
        # Define embedding functions for each collection type
        embedding_functions = {}
        for collection in collections_to_compare:
            if 'bert-' in collection and 'biobert' not in collection and 'pubmedbert' not in collection:
                if self.bert_tokenizer and self.bert_model:
                    base_func = self.create_bert_embedding_func(self.bert_tokenizer, self.bert_model)
                    embedding_functions[collection] = self.safe_embedding_wrapper(base_func, f"bert-{collection}")
                else:
                    print(f"WARNING: Skipping {collection} - BERT models not available")
                    continue
            
            elif 'pubmedbert' in collection:
                if self.pubmedbert_tokenizer and self.pubmedbert_model:
                    base_func = self.create_bert_embedding_func(self.pubmedbert_tokenizer, self.pubmedbert_model)
                    embedding_functions[collection] = self.safe_embedding_wrapper(base_func, f"pubmedbert-{collection}")
                else:
                    print(f"WARNING: Skipping {collection} - PubMedBERT models not available")
                    continue
                    
            elif 'biobert' in collection:
                if self.biobert_tokenizer and self.biobert_model:
                    base_func = self.create_bert_embedding_func(self.biobert_tokenizer, self.biobert_model)
                    embedding_functions[collection] = self.safe_embedding_wrapper(base_func, f"biobert-{collection}")
                else:
                    print(f"WARNING: Skipping {collection} - BioBERT models not available")
                    continue
                    
            elif 'openai' in collection:
                if self.openai_client:
                    base_func = self.create_openai_embedding_func(self.openai_client, "text-embedding-3-large")
                    embedding_functions[collection] = self.safe_embedding_wrapper(base_func, f"openai-{collection}")
                else:
                    print(f"WARNING: Skipping {collection} - OpenAI client not available")
                    continue
        
        if not embedding_functions:
            print("ERROR: No valid embedding functions available")
            return None
        
        test_queries = self.create_test_queries()
        comparison_results = []
        
        print(f"Running comparison on {len(embedding_functions)} collections with {len(test_queries)} queries")
        print("="*80)
        
        for i, query_info in enumerate(test_queries):
            query_text = query_info["query"]
            query_type = query_info["type"]
            expected_topics = query_info["expected_topics"]
            
            print(f"Testing query {i+1}/{len(test_queries)}: {query_text[:50]}...")
            
            for collection in embedding_functions.keys():
                try:
                    embedding_func = embedding_functions[collection]
                    query_embedding = embedding_func(query_text)

                    if query_embedding is None:
                        print(f"    WARNING: {collection} returned None for embedding")
                        continue
                    
                    search_result = self.search_collection(collection, query_text, query_embedding)

                    if not search_result["results"]:
                        print(f"    WARNING: No search results for {collection}")
                        # Still record the result with zero scores
                        result_row = {
                            "query": query_text,
                            "query_type": query_type,
                            "collection": collection,
                            "search_time": search_result["search_time"],
                            "avg_similarity_score": 0,
                            "relevance_score": 0,
                            "topic_coverage": 0,
                            "topics_found": [],
                            "num_results": 0
                        }
                        comparison_results.append(result_row)
                        continue
                    
                    relevance_metrics = self.evaluate_relevance(search_result["results"], expected_topics)
                    
                    result_row = {
                        "query": query_text,
                        "query_type": query_type,
                        "collection": collection,
                        "search_time": search_result["search_time"],
                        "avg_similarity_score": search_result["avg_score"],
                        "relevance_score": relevance_metrics["relevance_score"],
                        "topic_coverage": relevance_metrics["topic_coverage"],
                        "topics_found": relevance_metrics["topics_found"],
                        "num_results": len(search_result["results"])
                    }
                    
                    comparison_results.append(result_row)
                    print(f"    ✓ {collection}: {len(search_result['results'])} results, avg_score: {search_result['avg_score']:.4f}")
                    
                except Exception as e:
                    print(f"    ✗ Error with {collection}: {e}")
                    continue
        
        if not comparison_results:
            print("ERROR: No comparison results generated")
            return None
        
        self.results_df = pd.DataFrame(comparison_results)
        print(f"\nComparison completed. Generated {len(comparison_results)} result rows.")
        return self.results_df

    def generate_simple_report(self) -> str:
        """Generate a simple text report"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise ValueError("No results to analyze. Run comparison first!")
        
        analysis = self.analyze_results()
        
        report = f"""
                  # Embedding Method Comparison Report
                  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                  ## Collections Compared:
                  {chr(10).join([f"- {col}" for col in self.processed_collections])}

                  ## Key Findings:
                  - **Fastest Search**: {analysis['best_performers']['fastest_search']}
                  - **Highest Relevance**: {analysis['best_performers']['highest_relevance']}  
                  - **Best Topic Coverage**: {analysis['best_performers']['best_topic_coverage']}
                  - **Highest Similarity Scores**: {analysis['best_performers']['highest_similarity']}

                  ## Performance Summary:
                  {analysis['collection_means'][['search_time', 'relevance_score', 'topic_coverage', 'avg_similarity_score']].round(4).to_string()}

                  ## Total Results: {len(self.results_df)} comparisons across {len(self.results_df['query'].unique())} queries
                """
        
        return report

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze the comparison results"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise ValueError("No results to analyze. Run comparison first!")
        
        # Select only numeric columns for aggregation
        numeric_columns = ['search_time', 'avg_similarity_score', 'relevance_score', 'topic_coverage', 'num_results']
        
        # Filter to only include columns that exist and are numeric
        available_numeric_cols = []
        for col in numeric_columns:
            if col in self.results_df.columns:
                # Convert to numeric if not already, handling any errors
                try:
                    self.results_df[col] = pd.to_numeric(self.results_df[col], errors='coerce')
                    available_numeric_cols.append(col)
                except Exception as e:
                    print(f"Warning: Could not convert {col} to numeric: {e}")
        
        if not available_numeric_cols:
            raise ValueError("No numeric columns available for analysis")
        
        # Overall performance statistics - keep the original detailed format
        overall_performance = self.results_df.groupby('collection')[available_numeric_cols].agg({
            'search_time': ['mean', 'std'],
            'avg_similarity_score': ['mean', 'std'],
            'relevance_score': ['mean', 'std'],
            'topic_coverage': ['mean', 'std']
        }).round(4)
        
        # Best performers
        collection_means = self.results_df.groupby('collection')[available_numeric_cols].mean()
        
        best_performers = {
            'fastest_search': collection_means['search_time'].idxmin() if 'search_time' in collection_means.columns else 'N/A',
            'highest_similarity': collection_means['avg_similarity_score'].idxmax() if 'avg_similarity_score' in collection_means.columns else 'N/A',
            'highest_relevance': collection_means['relevance_score'].idxmax() if 'relevance_score' in collection_means.columns else 'N/A',
            'best_topic_coverage': collection_means['topic_coverage'].idxmax() if 'topic_coverage' in collection_means.columns else 'N/A'
        }
        
        # Performance by query type - keep original detailed format
        type_performance = None
        if 'query_type' in self.results_df.columns:
            try:
                type_performance = self.results_df.groupby(['query_type', 'collection'])[['search_time','avg_similarity_score', 'relevance_score', 'topic_coverage']].agg('mean').round(4)
            except Exception as e:
                print(f"Warning: Could not analyze performance by query type: {e}")
        
        return {
            'overall_performance': overall_performance,
            'best_performers': best_performers,
            'type_performance': type_performance,
            'collection_means': collection_means
        }

    def generate_comprehensive_report(self, pdf_filename: str = None, csv_filename: str = None) -> str:
        """Generate a comprehensive text report with all details"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise ValueError("No results to analyze. Run comparison first!")
        
        analysis = self.analyze_results()
        
        # Format the comprehensive report
        text_report = f"""
                        # Embedding Method Comparison Report
                        Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                        ## Collections Compared:
                        {chr(10).join([f"- {col}" for col in self.processed_collections])}

                        ## Overall Performance Summary:
                        {analysis['overall_performance']}

                        ## Key Findings:
                        - **Fastest Search**: {analysis['best_performers']['fastest_search']}
                        - **Highest Relevance**: {analysis['best_performers']['highest_relevance']}  
                        - **Best Topic Coverage**: {analysis['best_performers']['best_topic_coverage']}
                        - **Highest Similarity Scores**: {analysis['best_performers']['highest_similarity']}

                        ## Performance by Query Type:
                        {analysis['type_performance'] if analysis['type_performance'] is not None else 'No query type analysis available'}

                        ## Recommendations:
                        Based on the analysis, consider the following:
                        1. For speed-critical applications: Use the fastest performing model
                        2. For accuracy-critical applications: Use the highest relevance scoring model
                        3. For domain-specific queries: Consider how well each model handles medical terminology

                        ## Files Generated:"""
        
        if not pdf_filename and not csv_filename:
            text_report += "\n- No files generated"        
        if pdf_filename:
            text_report += f"\n- PDF Report: {pdf_filename}"
        if csv_filename:
            text_report += f"\n- Detailed CSV: {csv_filename}"
        
        text_report += f"\n\n## Total Results: {len(self.results_df)} comparisons across {len(self.results_df['query'].unique())} queries"
        
        return text_report

    def generate_pdf_report(self, filename: Optional[str] = None) -> str:
        """Generate a comprehensive PDF report with visualizations and save to experiments/results"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise ValueError("No results to analyze. Run comparison first!")
        
        # Create experiments/results directory if it doesn't exist
        results_dir = os.path.join("experiments", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_comparison_report_{timestamp}.pdf"
        
        # Full path to save in experiments/results
        full_path = os.path.join(results_dir, filename)
        
        analysis = self.analyze_results()
        
        # Create PDF with multiple pages
        with PdfPages(full_path) as pdf:
            # Page 1: Summary Statistics
            fig = plt.figure(figsize=(11, 8.5))
            gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
            
            # Title
            fig.suptitle('Embedding Method Comparison Report', fontsize=16, fontweight='bold')
            
            # Summary table
            ax1 = fig.add_subplot(gs[0, :])
            ax1.axis('tight')
            ax1.axis('off')
            
            summary_data = analysis['collection_means'][['search_time', 'avg_similarity_score', 'relevance_score', 'topic_coverage']].round(4)
            table = ax1.table(cellText=summary_data.values,
                            rowLabels=summary_data.index,
                            colLabels=['Search Time (s)', 'Similarity Score', 'Relevance Score', 'Topic Coverage'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            ax1.set_title('Performance Summary by Collection', fontweight='bold', pad=20)
            
            # Performance comparison charts
            ax2 = fig.add_subplot(gs[1, 0])
            collections = analysis['collection_means'].index
            relevance_scores = analysis['collection_means']['relevance_score']

            # Split collection names for better readability
            split_labels = []
            for col in collections:
                parts = col.split('-')
                if len(parts) == 1:
                    # No hyphen found, keep as is or split by length
                    split_labels.append(col)
                elif len(parts) == 2:
                    # Single hyphen, split into two lines
                    split_labels.append(f"{parts[0]}\n{parts[1]}")
                else:
                    # Multiple hyphens, split roughly in half
                    mid = len(parts) // 2
                    split_label = '-'.join(parts[:mid]) + '\n' + '-'.join(parts[mid:])
                    split_labels.append(split_label)

            bars = ax2.bar(range(len(collections)), relevance_scores, color='skyblue', alpha=0.7)
            ax2.set_xlabel('Collections')
            ax2.set_ylabel('Relevance Score')
            ax2.set_title('Relevance Score Comparison')
            ax2.set_xticks(range(len(collections)))
            ax2.set_xticklabels(split_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, relevance_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax3 = fig.add_subplot(gs[1, 1])
            topic_coverage = analysis['collection_means']['topic_coverage']
            
            bars = ax3.bar(range(len(collections)), topic_coverage, color='lightcoral', alpha=0.7)
            ax3.set_xlabel('Collections')
            ax3.set_ylabel('Topic Coverage')
            ax3.set_title('Topic Coverage Comparison')
            ax3.set_xticks(range(len(collections)))
            ax3.set_xticklabels(split_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, score in zip(bars, topic_coverage):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Search time comparison
            ax4 = fig.add_subplot(gs[2, :])
            search_times = analysis['collection_means']['search_time']
            
            bars = ax4.bar(range(len(collections)), search_times, color='lightgreen', alpha=0.7)
            ax4.set_xlabel('Collections')
            ax4.set_ylabel('Average Search Time (seconds)')
            ax4.set_title('Search Performance Comparison')
            ax4.set_xticks(range(len(collections)))
            ax4.set_xticklabels(split_labels, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, time in zip(bars, search_times):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                        f'{time:.4f}s', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Query Performance Details
            if 'query_type' in self.results_df.columns:
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle('Performance by Query Type', fontsize=16, fontweight='bold')
                
                query_types = self.results_df['query_type'].unique()
                
                for i, qtype in enumerate(query_types):
                    if i >= 4:  # Only show first 4 query types
                        break
                        
                    ax = axes[i//2, i%2]
                    type_data = self.results_df[self.results_df['query_type'] == qtype]
                    type_summary = type_data.groupby('collection')['relevance_score'].mean()
                    
                    bars = ax.bar(range(len(type_summary)), type_summary.values, alpha=0.7)
                    ax.set_title(f'{qtype.title()} Queries')
                    ax.set_xlabel('Collections')
                    ax.set_ylabel('Avg Relevance Score')
                    ax.set_xticks(range(len(type_summary)))
                    ax.set_xticklabels(split_labels, rotation=45, ha='right')
                    
                    # Add value labels
                    for bar, score in zip(bars, type_summary.values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontsize=8)
                
                # Hide empty subplots
                for i in range(len(query_types), 4):
                    axes[i//2, i%2].set_visible(False)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
            # Page 3: Detailed Results Text
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Generate comprehensive text report
            report_text = self.generate_comprehensive_report(pdf_filename=full_path)
            
            # Add text to the plot
            ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"PDF report saved to: {full_path}")
        return full_path

    def save_results(self, prefix: str = "embedding_comparison", save_pdf: bool = True) -> List[str]:
        """Save results to CSV files and PDF report in experiments/results folder"""
        if not hasattr(self, 'results_df') or self.results_df.empty:
            raise ValueError("No results to save. Run comparison first!")
        
        # Create experiments/results directory if it doesn't exist
        results_dir = os.path.join("experiments", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files_saved = []
        
        # Save detailed results CSV
        detailed_filename = f"{prefix}_detailed_{timestamp}.csv"
        detailed_path = os.path.join(results_dir, detailed_filename)
        self.results_df.to_csv(detailed_path, index=False)
        files_saved.append(detailed_path)
        
        # Save summary results CSV
        analysis = self.analyze_results()
        summary_filename = f"{prefix}_summary_{timestamp}.csv"
        summary_path = os.path.join(results_dir, summary_filename)
        analysis['collection_means'].to_csv(summary_path)
        files_saved.append(summary_path)
        
        # Save PDF report
        if save_pdf:
            try:
                pdf_filename = f"{prefix}_report_{timestamp}.pdf"
                pdf_path = self.generate_pdf_report(pdf_filename)
                files_saved.append(pdf_path)
                
                # Generate and print the comprehensive text report
                text_report = self.generate_comprehensive_report(
                    pdf_filename=pdf_path, 
                    csv_filename=detailed_path
                )
                print("\n" + "="*80)
                print("COMPREHENSIVE ANALYSIS REPORT")
                print("="*80)
                print(text_report)
                
            except Exception as e:
                print(f"WARNING: Could not generate PDF report: {e}")
        
        print(f"\nAll files saved to: experiments/results/")
        print(f"Files generated: {[os.path.basename(f) for f in files_saved]}")
        return files_saved

    def run_full_pipeline(self, run_processing: bool = True, run_comparison: bool = True, 
                            specific_configs: Optional[List[int]] = None) -> tuple:
            """
            Run the complete pipeline: processing + comparison
            
            Args:
                run_processing: Whether to run the processing step
                run_comparison: Whether to run the comparison step
                specific_configs: List of config indices to run (if None, runs all)
                
            Returns:
                tuple: (results_df, analysis_dict, saved_files)
            """
            results_df = None
            analysis_dict = None
            saved_files = []
            
            print("Starting Full Pipeline Execution")
            print("="*50)
            
            if run_processing:
                print("PHASE 1: Processing Papers with Different Embedding Methods")
                print("-"*50)
                
                # Get all embedding configurations
                all_configs = self.get_embedding_configs()
                
                if not all_configs:
                    print("ERROR: No valid embedding configurations found!")
                    return None, None, []
                
                # Filter configs if specific ones are requested
                if specific_configs is not None:
                    if not all(isinstance(i, int) and 0 <= i < len(all_configs) for i in specific_configs):
                        print(f"ERROR: Invalid config indices. Valid range: 0-{len(all_configs)-1}")
                        return None, None, []
                    configs_to_run = [all_configs[i] for i in specific_configs]
                    print(f"Running specific configs: {specific_configs}")
                else:
                    configs_to_run = all_configs
                    print(f"Running all {len(configs_to_run)} configurations")
                
                # Process papers with each configuration
                successful_configs = 0
                for i, config in enumerate(configs_to_run):
                    print(f"\nConfig {i+1}/{len(configs_to_run)}: {config['collection_name']}")
                    success = self.process_papers_with_config(config)
                    if success:
                        successful_configs += 1
                        print(f"✓ Successfully processed config: {config['collection_name']}")
                    else:
                        print(f"✗ Failed to process config: {config['collection_name']}")
                
                print(f"\nProcessing Phase Complete: {successful_configs}/{len(configs_to_run)} configs successful")
                
                if successful_configs == 0:
                    print("ERROR: No configurations processed successfully!")
                    return None, None, []
            
            if run_comparison:
                print("\nPHASE 2: Running Comparison Analysis")
                print("-"*50)
                
                if not self.processed_collections:
                    print("ERROR: No processed collections found for comparison!")
                    print("Available collections in Qdrant:")
                    try:
                        collections = self.qdrant_client.get_collections().collections
                        for col in collections:
                            print(f"  - {col.name}")
                        return None, None, []
                    except Exception as e:
                        print(f"ERROR: Could not retrieve collections: {e}")
                        return None, None, []
                
                print(f"Running comparison on {len(self.processed_collections)} collections:")
                for col in self.processed_collections:
                    print(f"  - {col}")
                
                # Run the comparison
                results_df = self.run_comparison()
                
                if results_df is not None and not results_df.empty:
                    print(f"✓ Comparison completed: {len(results_df)} result rows generated")
                    
                    # Analyze results
                    try:
                        analysis_dict = self.analyze_results()
                        print("✓ Analysis completed")
                        
                        # Generate and print simple report
                        report = self.generate_simple_report()
                        print("\n" + "="*50)
                        print("RESULTS SUMMARY")
                        print("="*50)
                        print(report)
                        
                        # Save results
                        try:
                            saved_files = self.save_results()
                            print(f"✓ Results saved to {len(saved_files)} files")
                        except Exception as e:
                            print(f"WARNING: Could not save results: {e}")
                            
                    except Exception as e:
                        print(f"ERROR: Analysis failed: {e}")
                        analysis_dict = None
                else:
                    print("✗ Comparison failed - no results generated")
            
            print("\n" + "="*50)
            print("PIPELINE EXECUTION COMPLETE")
            print("="*50)
            
            return results_df, analysis_dict, saved_files

    def cleanup_collections(self, collections_to_delete: Optional[List[str]] = None):
            """
            Clean up Qdrant collections (useful for testing)
            
            Args:
                collections_to_delete: List of collection names to delete. If None, deletes all processed collections.
            """
            if collections_to_delete is None:
                collections_to_delete = self.processed_collections
            
            if not collections_to_delete:
                print("No collections to clean up")
                return
            
            print(f"Cleaning up {len(collections_to_delete)} collections...")
            
            for collection_name in collections_to_delete:
                try:
                    self.qdrant_client.delete_collection(collection_name=collection_name)
                    print(f"✓ Deleted collection: {collection_name}")
                except Exception as e:
                    print(f"✗ Failed to delete collection {collection_name}: {e}")
            
            # Clear the processed collections list
            self.processed_collections = [col for col in self.processed_collections if col not in collections_to_delete]
            print("Cleanup complete")


# Example usage and testing functions
def main():
    """Example of how to use the ComprehensivePipelineTester"""
    
    # Example paper list - replace with your actual papers
    example_papers = [
        "PMC12075517.xml",
        "PMC12017772.xml", 
        "PMC12084038.xml",
        "PMC12011305.xml",
        "PMC10764003.xml"
    ]
    
    try:
        # Initialize the tester
        print("Initializing Pipeline Tester...")
        tester = ComprehensivePipelineTester(
            papers_list=example_papers,
            base_dir="data/raw"
        )
        
        # Run the full pipeline
        results_df, analysis, saved_files = tester.run_full_pipeline(
            run_processing=True,
            run_comparison=True,
            specific_configs=None  # Run all configs, or specify [0, 1, 2] for specific ones
        )
        
        if results_df is not None:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Generated {len(results_df)} comparison results")
            print(f"💾 Saved {len(saved_files)} result files")
            
            # Optional: Clean up collections after testing
            # tester.cleanup_collections()
            
        else:
            print("❌ Pipeline failed to generate results")
            
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()