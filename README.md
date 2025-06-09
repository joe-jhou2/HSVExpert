# 🧠 HSV Expert

HSV Expert is a research assistant for biomedical literature that intelligently searches, indexes, and summarizes articles from PubMed Central (PMC). The system is designed around modern NLP and vector search infrastructure, leveraging transformer-based embeddings and large language models.

---

## 🧱 Project Structure & Implementation


## ⚙️ Core Modules

### 1. **Ingestion Layer**
- **`fetch_pmc.py`**: Uses `Bio.Entrez` to search and retrieve PMC article metadata and XML links.
- **PDF fallback**: If XML is unavailable, retrieves PDF via NCBI OA API or Adobe PDF Services.
- Handles `ftp`, `http`, and redirect logic intelligently.

### 2. **Processing Layer**
- **`chunking.py`**: Splits long documents into overlapping chunks for embedding.
- **`embed_text.py`**: Encodes text chunks using `PubMedBERT` or similar transformers to produce dense vectors.

### 3. **Storage Layer**
- **`store.py`**: Stores embedded vectors and metadata (like `source=user-uploaded`) into a `Qdrant` collection.
- Metadata schema includes title, source, date, and chunk-level text.

### 4. **Retrieval Layer**
- **`query.py`**: Takes user question, embeds it, and retrieves top-K similar chunks from Qdrant based on cosine distance.
- Vector search is efficient and filtered (e.g., by `source=user-uploaded`).

### 5. **Summarization & Answering**
- **`generate_answer.py`**: Sends the user question + top relevant contexts to GPT-3.5 for summarization or Q&A.
- Uses system prompts for clarity and biomedical tone control.

### 6. **User Interface**
- **`app.py`**: Built with Streamlit. Simple search box with expandable summaries and reference highlights.
- Handles errors gracefully and displays provenance of answers.

---

## 📡 External Integrations

- **Entrez / PubMed**: Article search & metadata
- **NCBI OA Web Service**: Official access to open-access PMC documents
- **Adobe PDF Services**: PDF content extraction fallback
- **Qdrant Vector DB**: High-performance embedding storage & search
- **OpenAI GPT-3.5**: Semantic summarization and QA generation

---

## 🧪 Design Highlights

- 🧠 **PubMedBERT for embeddings**: domain-specific transformer for biomedical texts
- ⚡ **Chunk-level retrieval**: Enables accurate, context-rich answers
- 🗂️ **Metadata tagging**: Source tracking (`user-uploaded`, `NCBI`) to support fine-grained filtering
- 🔍 **CrossEncoder ready**: Supports reranking if needed
- 💬 **LLM-ready prompts**: Crafted system prompts to guide GPT-3.5 behavior

---


## 🔗 Quick Access

Run the app here: [HSV Expert Streamlit App](https://<your-streamlit-cloud-link>)

---


