# _rag_llm.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def generate_answer_with_gpt35(question, retrieved_chunks, max_context_tokens=3000):
    """
    Generate an answer using GPT-3.5 Turbo with retrieved context.

    Args:
        question (str): The user's question.
        retrieved_chunks (list): List of dicts with 'text' field and optional metadata.
        max_context_tokens (int): Maximum tokens to use for context.

    Returns:
        str: The generated answer.
    """
    
    # Input validation
    if not question or not question.strip():
        return "Please provide a valid question."
    
    if not retrieved_chunks:
        return "I couldn't find relevant information to answer your question. Please try rephrasing or asking about a different topic."
    
    # Enhanced system prompt for medical/research context
    system_prompt = """You are an expert medical research assistant specializing in HSV (Herpes Simplex Virus) research. 
    
    CRITICAL: Only use information from the provided research context. Do NOT supplement 
    with general medical knowledge from your training data.

    Guidelines:
    1. Base answers STRICTLY on the provided research context
    2. If the context lacks specific details, state: "The available research papers 
    don't provide information about [specific detail]"
    3. Never fill in gaps with general medical knowledge
    4. Include relevant citations when possible (mention paper/study details)
    5. Distinguish between established facts and preliminary findings
    6. Use clear, professional medical language
    7. If asked about treatments, always recommend consulting healthcare providers
    8. Acknowledge uncertainty when research is conflicting or limited
    
    Format your response with clear structure when appropriate."""
    
    # Prepare context with metadata and source information
    context_parts = []
    total_tokens = 0
    
    for i, chunk in enumerate(retrieved_chunks):
        # Extract text and metadata
        chunk_text = chunk.get("text", "")
        paper_id = chunk.get("paper_id", f"Source_{i+1}")
        section_title = chunk.get("section_title", "")
        
        # Create formatted context entry
        source_info = f"[Source {i+1}"
        if paper_id != f"Source_{i+1}":
            source_info += f" - {paper_id}"
        if section_title:
            source_info += f" - {section_title}"
        source_info += "]"
        
        formatted_chunk = f"{source_info}\n{chunk_text}"
        
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(formatted_chunk) // 4
        
        if total_tokens + estimated_tokens > max_context_tokens:
            break
            
        context_parts.append(formatted_chunk)
        total_tokens += estimated_tokens
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Enhanced user message with clear instructions
    user_message = f"""Based on the following research context, please answer the question. 
    If the context doesn't contain sufficient information, please state this clearly.

    RESEARCH CONTEXT:
    {context_text}

    QUESTION: {question}

    Please provide a comprehensive answer based on the research context above."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,  # Very low for factual consistency
            max_tokens=800,   # Reasonable limit for responses
            top_p=0.9,       # Slight creativity control
            frequency_penalty=0.1,  # Reduce repetition
            presence_penalty=0.1    # Encourage diverse vocabulary
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Add source count information
        source_count = len(context_parts)
        if source_count > 0:
            answer += f"\n\n*Response based on {source_count} research source(s).*"
        
        return answer
        
    except Exception as e:
        return f"I encountered an error while generating the response: {str(e)}. Please try again."

def generate_answer_with_conversation_context(question, retrieved_chunks, conversation_history=None, max_context_tokens=3000):
    """
    Enhanced version that considers conversation history for better context.
    
    Args:
        question (str): The user's current question.
        retrieved_chunks (list): Retrieved context chunks.
        conversation_history (list): Previous conversation messages.
        max_context_tokens (int): Maximum tokens for context.
    
    Returns:
        str: Generated answer with conversation awareness.
    """
    
    # Build conversation context
    conversation_context = ""
    if conversation_history:
        recent_history = conversation_history[-4:]  # Last 2 exchanges
        for msg in recent_history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            conversation_context += f"{role}: {msg['content']}\n"
    
    # Enhanced system prompt with conversation awareness
    system_prompt = f"""You are an expert medical research assistant specializing in HSV research.

    {"Previous conversation context:\n" + conversation_context + "\n" if conversation_context else ""}
    
    Guidelines:
    1. Consider the conversation context when relevant
    2. Base answers on provided research context
    3. Acknowledge limitations in available research
    4. Use professional medical language
    5. Always recommend consulting healthcare providers for medical decisions
    6. Distinguish between established facts and preliminary research
    """
    
    # Use the same context preparation as the main function
    context_parts = []
    total_tokens = 0
    
    for i, chunk in enumerate(retrieved_chunks):
        chunk_text = chunk.get("text", "")
        paper_id = chunk.get("paper_id", f"Source_{i+1}")
        section_title = chunk.get("section_title", "")
        
        source_info = f"[Source {i+1}"
        if paper_id != f"Source_{i+1}":
            source_info += f" - {paper_id}"
        if section_title:
            source_info += f" - {section_title}"
        source_info += "]"
        
        formatted_chunk = f"{source_info}\n{chunk_text}"
        estimated_tokens = len(formatted_chunk) // 4
        
        if total_tokens + estimated_tokens > max_context_tokens:
            break
            
        context_parts.append(formatted_chunk)
        total_tokens += estimated_tokens
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    user_message = f"""RESEARCH CONTEXT:
                    {context_text}

                    CURRENT QUESTION: {question}

                    Please provide a comprehensive answer based on the research context."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        source_count = len(context_parts)
        if source_count > 0:
            answer += f"\n\n*Response based on {source_count} research source(s).*"
        
        return answer
        
    except Exception as e:
        return f"I encountered an error while generating the response: {str(e)}. Please try again."