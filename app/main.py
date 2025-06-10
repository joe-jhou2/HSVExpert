import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dotenv
dotenv.load_dotenv()

import streamlit as st
import tempfile
import uuid
from openai import OpenAI
from src.data_processing import ExtractTextInfoFromPDF, integrate_with_pdf_extractor, \
    chunk_sections, \
    load_embedding_model, embed_text, get_openai_tokenizer, \
    store_chunks_in_qdrant
from src.generation._rag_llm import generate_answer_with_gpt35
from src.retrieval import initialize_reranker, enhanced_query_with_reranking, query_qdrant, dual_retriever_fusion_query, \
    HybridRetrievalPipeline
from src.utils import load_pdf_count, increment_pdf_count, reset_pdf_count, auto_reset_if_new_month

# PDF Counter JSON file path
COUNTER_FILE = "data/pdf_counter.json"
PDF_LIMIT = 500

# Qdrant configuration
QDRANT_EC2_IP = os.getenv("QDRANT_EC2_IP")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "hsv_papers"

# Load embedding model
# biobert_tokenizer,biobert_model = load_embedding_model("dmis-lab/biobert-base-cased-v1.1")
pubmedbert_tokenizer, pubmedbert_model = load_embedding_model("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    
# Initialize OpenAI openai_client 
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_tokenizer = get_openai_tokenizer()

# Load Reranker model if needed (For BioBERT)
reranker = initialize_reranker(use_medical=True)

# Set up Streamlit page configuration
st.set_page_config(page_title="HSV Expert", page_icon="💬", layout="wide")

# Initialize session state at module level
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

def generate_paper_summary(text_content, openai_client):
    """Generate a comprehensive summary of the research paper"""
    try:
        prompt = f"""
        Please provide a comprehensive summary of this research paper. The summary should include:
        
        1. **Main Research Question/Objective**: What is the primary research question or goal?
        2. **Methodology**: What methods were used in the study?
        3. **Key Findings**: What are the most important results and discoveries?
        4. **Clinical Implications**: What are the practical applications or implications for treatment/practice?
        5. **Limitations**: What are the main limitations of the study?
        6. **Future Directions**: What future research directions are suggested?
        
        Keep the summary concise but comprehensive, around 300-400 words.
        
        Research Paper Content:
        {text_content[:8000]}
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert research analyst specializing in medical and scientific literature. Provide clear, accurate, and well-structured summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and add to vector database"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.session_state.processing_status = "Extracting text from PDF..."
        
        extractor = ExtractTextInfoFromPDF(tmp_file_path)
        extractor.extract_text()
        
        st.session_state.processing_status = "Integrating and chunking text..."
        
        result = integrate_with_pdf_extractor(extractor)
        
        st.session_state.processing_status = "Generating paper summary..."
        full_text = " ".join([section.get('text', '') for section in result if isinstance(section, dict) and 'text' in section])
        if not full_text:
            full_text = str(result)[:8000]
        
        paper_summary = generate_paper_summary(full_text, openai_client)
        
        st.session_state.processing_status = "Adding to vector database..."
        
        chunks = chunk_sections(result, max_tokens=300, tokenizer=pubmedbert_tokenizer, overlap=50, model_type="PubMedBERT")  

        user_paper_uuid = str(uuid.uuid4())

        embeddings = []
        for chunk in chunks:
            embedding = embed_text(chunk["text"], pubmedbert_tokenizer, pubmedbert_model)
            chunk["embedding"] = embedding
            chunk["paper_id"] = user_paper_uuid
            embeddings.append(embedding)
        
        store_chunks_in_qdrant(chunks,
                               embeddings, 
                               embed_model_name="BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                               collection_name=COLLECTION_NAME,
                               source="user-uploaded",
                               host=QDRANT_EC2_IP,
                               port=QDRANT_PORT)

        os.unlink(tmp_file_path)
        
        file_info = {
            "name": uploaded_file.name,
            "size": len(uploaded_file.getvalue()),
            "status": "Successfully processed",
            "summary": paper_summary,
            "chunks_added": len(chunks)
        }
        st.session_state.uploaded_files.append(file_info)
        st.session_state.processing_status = f"✅ Successfully processed {uploaded_file.name} - Added {len(chunks)} chunks to database"
        
        return True, f"Successfully processed and added {uploaded_file.name} to the knowledge base!"
        
    except Exception as e:
        st.session_state.processing_status = f"❌ Error processing {uploaded_file.name}: {str(e)}"
        return False, f"Error processing {uploaded_file.name}: {str(e)}"

def user_input():
    # Initialize session state for input clearing
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0
    
    # Use dynamic key that changes to clear the input
    question = st.text_input(
        "Ask me anything about HSV research:",
        "",
        key=f"input_{st.session_state.input_key}",
        placeholder="e.g., What are the latest treatments for HSV-2?"
    )
    
    return question

def clear_input():
    """Call this function to clear the input box"""
    st.session_state.input_key += 1

def display_chat():
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"""
                <div style="background-color:#DCF8C6; padding:15px; border-radius:15px; width:fit-content; max-width:80%; margin-left:auto; margin-bottom:10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <b>You:</b><br>{chat['content']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color:#F8F9FA; padding:15px; border-radius:15px; width:fit-content; max-width:80%; margin-right:auto; margin-bottom:10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 4px solid #007bff;">
                <b>HSV Expert:</b><br>{chat['content']}
                </div>
                """, unsafe_allow_html=True)

def sidebar_controls():
    """Sidebar for database management and PDF uploads"""
    st.sidebar.header("🗄️ Knowledge Base Manager")
    
    # Database status
    st.sidebar.subheader("📊 Database Status")
    st.sidebar.info("✅ Connected to HSV Research Database\n\n🔍 Ready to answer questions from existing knowledge base")
    
    # Check and auto-reset if new month
    auto_reset_if_new_month(COUNTER_FILE)
    
    # Get current PDF count
    current_count, _ = load_pdf_count(COUNTER_FILE)

    # Optional PDF upload section
    st.sidebar.subheader("➕ Expand Knowledge Base")
    st.sidebar.markdown("*Optional: Add new research papers*")
    
    # Check if service should halt (500 PDF limit)
    if current_count >= PDF_LIMIT:
        # Service halted
        st.sidebar.error("❌ PDF processing service unavailable - monthly limit of 500 PDFs reached")
        st.sidebar.markdown("*Service will resume next month*")
    else:
        # Service available - show normal upload interface
        uploaded_files = st.sidebar.file_uploader(
            "Upload new PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload additional PDF documents to expand the knowledge base"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if not any(f["name"] == uploaded_file.name for f in st.session_state.uploaded_files):
                    if st.sidebar.button(f"📄 Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            success, message = process_uploaded_pdf(uploaded_file)
                            if success:
                                st.sidebar.success(message)
                                st.rerun()
                            else:
                                st.sidebar.error(message)
    
    # Show simple count info
    st.sidebar.caption(f"PDFs processed this month: {current_count}/500")

    # Processing status
    if st.session_state.processing_status:
        st.sidebar.info(st.session_state.processing_status)
    
    # Show uploaded files this session
    if st.session_state.uploaded_files:
        st.sidebar.subheader("📋 Files Added This Session")
        for file_info in st.session_state.uploaded_files:
            with st.sidebar.expander(f"📄 {file_info['name']}"):
                st.write(f"**Status:** {file_info['status']}")
                st.write(f"**Size:** {file_info['size']/1024:.1f} KB")
                st.write(f"**Chunks Added:** {file_info.get('chunks_added', 'N/A')}")
                if 'summary' in file_info:
                    st.write("**Summary:**")
                    st.write(file_info['summary'])
    
    # Clear session data
    if st.sidebar.button("🗑️ Clear Session Data"):
        st.session_state.uploaded_files = []
        st.session_state.processing_status = ""
        st.rerun()

def handle_query(query):
    """Handle user query and return response"""
    try:
        pipeline = HybridRetrievalPipeline(hsv_optimized=True)
        pipeline.__init__(collection_name=COLLECTION_NAME, host=QDRANT_EC2_IP, port=QDRANT_PORT)
        results = pipeline.hybrid_search(
            query_text=query,
            embed_function=embed_text,
            tokenizer=pubmedbert_tokenizer,
            model=pubmedbert_model,
            initial_retrieval_k=100,
            rrf_candidates=25,
            final_top_n=10
            )

        if not results:
            return "I couldn't find specific information about that topic in the current knowledge base. Could you try rephrasing your question or asking about a different aspect of HSV research?"
        
        # Generate answer using retrieved chunks
        answer = generate_answer_with_gpt35(query, results)
        return answer
        
    except Exception as e:
        return f"I encountered an error while searching the database: {str(e)}. Please try again or rephrase your question."

def handle_query_and_update(query):
    """Handle query processing and chat history update"""
    if query and query.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Generate response
        with st.spinner("🔍 Searching research database and generating response..."):
            answer = handle_query(query)

        # Add assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

def main():
    sidebar_controls()
    
    # Main content area
    st.title("🧬 HSV Expert - Research Assistant")
    st.markdown("""
    **Welcome to HSV Expert!** I'm a specialized AI assistant with access to a comprehensive database of HSV research papers. 
    
    🔍 **Ask me anything about:**
    - HSV Pathology and Immunology
    - Clinical Research Findings
    - HSV Treatments and Therapies
    - Epidemiological data
    - And much more!
    """)
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Vaccine Research"):
            handle_query_and_update("What is the HSV vaccine progress?")
    
    with col2:
        if st.button("Cell Receptors"):
            handle_query_and_update("What cellular receptors are HSV used to entry host cell?")
    
    with col3:
        if st.button("Skin Immunology"):
            handle_query_and_update("What is skin immunology for HSV?")
    
    st.markdown("---")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")
        display_chat()
    else:
        st.info("👋 Start by asking a question about HSV research!")
    
    # User input with Enter key support
    with st.form(key='chat_form', clear_on_submit=True):
        query = st.text_input(
            "Ask a question about HSV research:",
            placeholder="e.g., What are the latest treatments for HSV-1?",
            help="Type your question and press Enter or click Send"
        )
        
        submitted = st.form_submit_button("Send", type="primary")
        
        if submitted and query.strip():
            handle_query_and_update(query)

    # Show database info and tips
    with st.expander("ℹ️ About HSV Expert"):
        st.markdown("""
        ### 🎯 How HSV Expert Works
        
        **Existing Knowledge Base:**
        - I have preloaded with HSV research papers from PubMed Central (PMC), from 2000 to 2025
        - You can ask questions immediately without uploading anything
        - I will search through the existing knowledge base to provide evidence-based answers
        
        **Optional Expansion:**
        - Use the sidebar to upload additional PDF research papers
        - New papers are automatically processed and added to the knowledge base
        - Auto-generated summaries help you understand each paper's key contributions
        
        ### 💡 Example Questions:
        - "What are the most effective antiviral treatments for HSV-2?"
        - "How do HSV infections affect pregnancy outcomes?"
        - "What are the latest developments in HSV vaccine research?"
        - "Compare the efficacy of acyclovir vs valacyclovir"
        - "What are the diagnostic challenges for asymptomatic HSV?"
        
        ### 🔬 Features:
        - **Real-time search** through research database
        - **Evidence-based answers** with source citations
        - **Automatic paper summarization** for uploaded PDFs
        - **Expandable knowledge base** via PDF uploads

        """)

if __name__ == "__main__":
    main()