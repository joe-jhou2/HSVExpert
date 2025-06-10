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
│ • Streamlit App │    │ • Vector Search  │    │ • Embedding     │
│ • Search UI     │    │ • Reranking      │    │ • Similarity    │
│ • Results View  │    │ • Answer Gen.    │    │ • Filtering     │
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

### 🔍 Query & Retrieval
- **Semantic Search Engine**: Performs semantic search with configurable similarity thresholds for HSV research
- **Hybrid Search**: Combines vector similarity with metadata filtering for precise results
- **Reranking**: Optional CrossEncoder for improved result quality and relevance scoring

### 🤖 AI-Powered Answering
- **Contextual Response Generator**: Context-aware response generation using GPT-3.5 with HSV domain knowledge
- **Prompt Engineering**: Specialized prompts optimized for biomedical accuracy and HSV research context
- **Citation Integration**: Automatic source attribution and reference linking with PubMed IDs

### 🖥️ User Interface
- **Research Dashboard**: Streamlit-based web application tailored for biomedical researchers
- **Interactive Features**: Expandable results, source highlighting, error handling, and HSV-specific search filters

---