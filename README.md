# 🧠 HSV Expert

> A sophisticated biomedical research assistant powered by AI that intelligently searches, indexes, and summarizes scientific literature from PubMed Central (PMC), specializing in HSV research.

This project enhances the efficiency of literature reviews, systematic reviews, and research tasks by automating the retrieval and summarization of HSV-related articles. HSV Expert leverages cutting-edge NLP technologies including transformer-based embeddings and large language models to provide researchers with accurate, contextual answers from biomedical literature.

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Processing      │───▶│   Storage       │
│                 │    │                  │    │                 │
│ • PubMed/PMC    │    │ • Text Chunking  │    │ • Qdrant Vector │
│ • User Uploads  │    │ • Embedding      │    │   Database      │
│ • Scheduled     │    │ • PDF Parsing    │    │ • Metadata      │
│   Auto-search   │    │ • Preprocessing  │    │ • Indexing      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                  │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ User Interface  │◀───│    Retrieval     │◀───│   Query Engine  │
│                 │    │                  │    │                 │
│ • Streamlit App │    │ • Hybrid Search  │    │ • Dense Embed.  │
│ • Search UI     │    │ • Cross-Encoder  │    │ • BM25 Sparse   │
│ • Results View  │    │ • RRF Fusion     │    │ • Reranking     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 📦 Core Components

### 🔄 Data Ingestion
- **Auto HSV Literature Harvester**: Automatically retrieves HSV-related articles from PubMed Central using Bio.Entrez
- **Multi-format Document Processing**: Handles PDF extraction via NCBI OA API and Adobe PDF Services
- **Resilient Download Manager**: Implements multiple XML and PDF download methods with fallback sources when primary sources fail

### ⚙️ Text Processing
- **Intelligent Document Segmenter**: Implements document segmentation with overlap based on article structure to maintain coherence
- **Comprehensive PDF Parser**: Advanced PDF parsing and text extraction to generate structured data from complex documents
- **Structure-aware Chunking**: Splits documents into manageable chunks for embedding while preserving logical document flow
- **Biomedical Text Embedder**: Generates embeddings using PubMedBERT, specialized for biomedical text understanding

### 💾 Vector Storage
- **Vector Database Manager**: Manages Qdrant vector database operations with optimized indexing
- **Metadata Management**: Tracks source, publication date, and document hierarchy for comprehensive filtering
- **Schema Design**: Optimized for fast retrieval and filtering with HSV-specific metadata fields

### 🔍 Advanced Retrieval System

#### **Hybrid Search Pipeline**
The system implements a sophisticated 4-stage hybrid search approach that combines multiple retrieval methods for optimal accuracy:

**Stage 1: Dense Retrieval**
- Semantic vector search using PubMedBERT embeddings
- Captures semantic similarity and biomedical context
- Configurable initial retrieval count (default: 50 candidates)

**Stage 2: Sparse Retrieval (BM25)**
- Keyword-based retrieval using BM25 algorithm
- Two operational modes:
  - **Fast Mode**: BM25 on dense retrieval results (default)
  - **Comprehensive Mode**: BM25 across entire document corpus
- Balances speed vs. comprehensiveness based on research needs

**Stage 3: Reciprocal Rank Fusion (RRF)**
- Intelligently combines dense and sparse retrieval results
- Mitigates individual method weaknesses through rank-based fusion
- Configurable candidate pool (default: 30 documents)

**Stage 4: Cross-Encoder Reranking**
- Final precision step using transformer-based reranking
- Optimizes result relevance through query-document interaction modeling
- Returns top-N most relevant results (default: 10)

#### **Retrieval Features**
- **Adaptive Search Strategy**: Automatically selects optimal retrieval approach based on query characteristics
- **Configurable Parameters**: Fine-tune retrieval behavior for different research scenarios
- **Performance Optimization**: Balanced approach between retrieval quality and computational efficiency
- **Comprehensive Coverage**: Option to search entire corpus or focus on semantically similar documents

### 🤖 AI-Powered Answering
- **Contextual Response Generator**: Context-aware response generation using GPT-3.5 with HSV domain knowledge
- **Prompt Engineering**: Specialized prompts optimized for biomedical accuracy and HSV research context
- **Citation Integration**: Automatic source attribution and reference linking with PubMed IDs
- **Multi-stage Quality Control**: Retrieval quality assessment and answer validation

---

## 🚀 Key Advantages

### **Research Efficiency**
- **Intelligent Literature Discovery**: Automatically finds relevant HSV research across multiple retrieval paradigms
- **Quality-Focused Results**: Multi-stage refinement ensures high-relevance results for biomedical research
- **Flexible Search Modes**: Adapt search strategy based on research phase (exploratory vs. targeted)

### **Domain Specialization**
- **Biomedical Optimization**: All components tuned specifically for biomedical and HSV research contexts
- **Expert-Level Accuracy**: Specialized embeddings and prompts ensure research-grade output quality
- **Comprehensive Coverage**: Hybrid approach captures both semantic relationships and exact keyword matches

---
