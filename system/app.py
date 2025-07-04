import os
import streamlit as st
import pinecone
import requests
import json
import uuid
from datetime import datetime
import pandas as pd
import re
from ollama_config import config
from ollama_utils import OllamaClient, OllamaEmbeddings, format_prompt, extract_verdict, SimpleTextSplitter
from mmr_utils import mmr_similarity_search
from dotenv import load_dotenv
from pdf_loader import PDFDocumentLoader

# Streamlit page configuration - MUST be first!
st.set_page_config(
    page_title="RAG-based Explainable Fact-Checker",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables from .env file
load_dotenv()

# Initialize environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fact-checker-index")

# Check if Pinecone credentials are available
if not PINECONE_API_KEY:
    st.warning("""
    ⚠️ Configuration Pinecone manquante !
    
    Veuillez créer un fichier `.env` avec les informations suivantes :
    
    ```
    PINECONE_API_KEY=votre_cle_api_pinecone
    PINECONE_INDEX_NAME=fact-checker-index
    ```
    
    L'application fonctionnera avec des documents simulés.
    """)
    # Initialize Pinecone with None values (will use fallback)
    PINECONE_API_KEY = None

# Initialize Pinecone only if credentials are available
if PINECONE_API_KEY:
    try:
        # New Pinecone API - no environment needed
        pinecone.init(api_key=PINECONE_API_KEY)
        st.success("✅ Connexion Pinecone établie")
    except Exception as e:
        st.error(f"❌ Erreur de connexion Pinecone: {e}")
        st.info("L'application fonctionnera avec des documents simulés")
else:
    st.info("🔧 Mode sans Pinecone - utilisation de documents simulés")

# Initialize Ollama client
ollama_client = OllamaClient()

# Create embeddings and vector store (using Ollama if available, otherwise fallback)
try:
    embeddings = OllamaEmbeddings()
    # Note: For now, we'll use a simple approach without Pinecone vector store
    # since Ollama embeddings might not be compatible with Pinecone
    vectorstore = None
except Exception as e:
    st.warning(f"Ollama embeddings not available: {e}")
    vectorstore = None

# Define prompts as templates
retrieval_prompt_template = """I need to fact check the following claim:

Claim: {claim}

What specific keywords or search queries should I use to find reliable information related to this claim?
Please provide 3-5 different search queries that would help gather relevant evidence.
"""

analysis_prompt_template = """You are an objective fact-checker. Your job is to verify the following claim using ONLY the provided evidence.

Claim: {claim}

Evidence:
{retrieved_docs}

Please analyze the claim step by step:
1. Break down the claim into its key components
2. For each component, identify supporting or contradicting evidence from the provided sources
3. Note any missing information that would be needed for a complete verification
4. Assign a verdict to the claim from the following options:
   - TRUE: The claim is completely supported by the evidence
   - MOSTLY TRUE: The claim is mostly accurate but contains minor inaccuracies
   - MIXED: The claim contains both accurate and inaccurate elements
   - MOSTLY FALSE: The claim contains some truth but is misleading overall
   - FALSE: The claim is contradicted by the evidence
   - UNVERIFIABLE: Cannot be determined from the provided evidence

Format your response as follows:
CLAIM COMPONENTS:
- [List key components]

EVIDENCE ANALYSIS:
- Component 1: [Analysis with direct quotes from sources]
- Component 2: [Analysis with direct quotes from sources]
...

MISSING INFORMATION:
- [List any info gaps]

VERDICT: [Your verdict]

EXPLANATION:
[Brief explanation of verdict]
"""

summary_prompt_template = """Based on the following fact-check analysis:

{analysis}

Generate a concise summary of the fact-check that explains:
1. What was claimed
2. What the evidence shows
3. The final verdict and why

Keep your summary under 200 words and make it accessible to general audiences.
"""

def generate_search_queries(claim):
    """Generate search queries for a given claim."""
    prompt = format_prompt(retrieval_prompt_template, claim=claim)
    result = ollama_client.generate(prompt, temperature=0.3)
    st.session_state.tokens_used += ollama_client.tokens_used
    return result

def retrieve_documents(claim, data_dir="/home/moi/Documents/internship/climat-misinformation-detection/rapport", k=3, lambda_param=0.5):
    """Retrieve relevant documents from local PDF files using MMR selection."""
    # Générer les requêtes de recherche
    queries_text = generate_search_queries(claim)
    queries = re.findall(r'(?:^|\n)(?:\d+\.|\*|\-)\s*(.+?)(?=\n|$)', queries_text)
    if not queries:
        queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
    if not queries:
        queries = [claim]
    # On prend la première requête générée pour l'embedding
    query_for_embedding = queries[0]

    # Charger et splitter les documents PDF
    documents = PDFDocumentLoader.load_directory(data_dir)
    splitter = SimpleTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    if not chunks:
        st.warning("Aucun document PDF trouvé dans le dossier spécifié.")
        return []

    # Générer les embeddings des chunks et de la requête
    embeddings_model = OllamaEmbeddings()
    chunk_texts = [chunk['page_content'] for chunk in chunks]
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    query_embedding = embeddings_model.embed_query(query_for_embedding)

    # Sélectionner les meilleurs chunks avec MMR
    selected_indices = mmr_similarity_search(chunk_embeddings, query_embedding, k=k, lambda_param=lambda_param)
    selected_chunks = [chunks[i] for i in selected_indices]

    # Formater les documents pour l'affichage/traitement
    all_docs = []
    doc_sources = {}
    for i, chunk in enumerate(selected_chunks):
        doc_id = str(uuid.uuid4())[:8]
        doc_content = chunk['page_content']
        doc_source = chunk['metadata'].get('source', 'Unknown')
        all_docs.append(f"[Document {doc_id}]\n{doc_content}\n")
        doc_sources[doc_id] = {
            'source': doc_source,
            'content': doc_content,
            'query': query_for_embedding
        }
    st.session_state.current_sources = doc_sources
    return all_docs

def analyze_claim(claim, retrieved_docs):
    """Analyze the claim using retrieved documents."""
    prompt = format_prompt(analysis_prompt_template, claim=claim, retrieved_docs='\n\n'.join(retrieved_docs))
    result = ollama_client.generate(prompt, temperature=0.2)
    st.session_state.tokens_used += ollama_client.tokens_used
    return result

def generate_summary(analysis):
    """Generate a concise summary of the analysis."""
    prompt = format_prompt(summary_prompt_template, analysis=analysis)
    result = ollama_client.generate(prompt, temperature=0.3)
    st.session_state.tokens_used += ollama_client.tokens_used
    return result

def save_fact_check(claim, analysis, summary, sources):
    """Save the fact check to the history."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Extract verdict from analysis
    verdict = extract_verdict(analysis)
    
    fact_check = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'claim': claim,
        'analysis': analysis,
        'summary': summary,
        'verdict': verdict,
        'sources': sources
    }
    
    st.session_state.history.append(fact_check)
    return fact_check

# Initialize session state
if 'tokens_used' not in st.session_state:
    st.session_state.tokens_used = 0
if 'current_sources' not in st.session_state:
    st.session_state.current_sources = {}
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("📊 RAG-based Explainable Fact-Checker")
st.markdown("""
This tool uses RAG (Retrieval-Augmented Generation) to fact-check claims against a database of reliable sources.
It provides transparency by showing which sources were used and how the verdict was determined.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This fact-checking system:
    - Breaks claims into verifiable components
    - Retrieves relevant information from trusted sources
    - Provides transparent reasoning and source attribution
    - Generates human-readable explanations
    """)
    
    st.header("Statistics")
    st.metric("Tokens Used", f"{st.session_state.tokens_used:,}")
    
    st.header("History")
    if st.session_state.history:
        for i, check in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"#{len(st.session_state.history)-i}: {check['claim'][:50]}..."):
                st.write(f"**Verdict:** {check['verdict']}")
                st.write(f"**Date:** {check['timestamp']}")
                if st.button("Load", key=f"load_{check['id']}"):
                    st.session_state.current_claim = check['claim']
                    st.session_state.current_analysis = check['analysis']
                    st.session_state.current_summary = check['summary']
                    st.session_state.current_sources = check['sources']
                    st.session_state.showing_results = True
                    st.experimental_rerun()

# Main interface
claim = st.text_area("Enter the claim to fact-check:", height=100)

col1, col2 = st.columns([1, 3])
with col1:
    check_button = st.button("🔍 Fact Check", type="primary")
with col2:
    st.markdown("*This will retrieve relevant documents, analyze the claim, and provide an explanation.*")

# Process the claim when button is clicked
if check_button and claim:
    with st.spinner("Fact checking in progress..."):
        # Store the claim
        st.session_state.current_claim = claim
        
        # Step 1: Retrieve relevant documents
        st.markdown("### 🔎 Retrieving relevant information...")
        retrieved_docs = retrieve_documents(claim)
        
        # Step 2: Analyze the claim
        st.markdown("### 🧠 Analyzing claim against evidence...")
        analysis = analyze_claim(claim, retrieved_docs)
        st.session_state.current_analysis = analysis
        
        # Step 3: Generate summary
        st.markdown("### 📝 Generating summary...")
        summary = generate_summary(analysis)
        st.session_state.current_summary = summary
        
        # Save the fact check
        save_fact_check(claim, analysis, summary, st.session_state.current_sources)
        
        # Set flag to show results
        st.session_state.showing_results = True
        
        st.experimental_rerun()

# Display results if available
if 'showing_results' in st.session_state and st.session_state.showing_results:
    st.markdown("## Results")
    
    # Extract verdict for styling
    verdict = extract_verdict(st.session_state.current_analysis)
    
    # Style based on verdict
    verdict_colors = {
        "TRUE": "green",
        "MOSTLY TRUE": "lightgreen",
        "MIXED": "yellow",
        "MOSTLY FALSE": "orange",
        "FALSE": "red",
        "UNVERIFIABLE": "gray"
    }
    verdict_color = verdict_colors.get(verdict, "gray")
    
    # Display summary with verdict highlight
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
        <h3 style="margin-top: 0;">Claim:</h3>
        <p>{st.session_state.current_claim}</p>
        <h3>Verdict: <span style="color:{verdict_color}; font-weight:bold;">{verdict}</span></h3>
        <p>{st.session_state.current_summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for detailed analysis and source tracking
    tab1, tab2 = st.tabs(["📊 Detailed Analysis", "📚 Source Tracking"])
    
    with tab1:
        st.markdown("### Detailed Analysis")
        st.text_area("Full Analysis", st.session_state.current_analysis, height=400)
    
    with tab2:
        st.markdown("### Sources Used")
        
        for doc_id, doc_info in st.session_state.current_sources.items():
            with st.expander(f"Document {doc_id} - {doc_info['source']}"):
                st.markdown(f"**Query used**: {doc_info['query']}")
                st.markdown(f"**Source**: {doc_info['source']}")
                st.text_area(f"Content", doc_info['content'], height=200)
        
        # Highlight where sources are referenced in analysis
        st.markdown("### Source References in Analysis")
        analysis_with_highlights = st.session_state.current_analysis
        for doc_id in st.session_state.current_sources:
            analysis_with_highlights = analysis_with_highlights.replace(
                f"Document {doc_id}",
                f"<span style='background-color: #fff2cc; padding: 2px 4px; border-radius: 3px;'>Document {doc_id}</span>"
            )
        st.markdown(analysis_with_highlights, unsafe_allow_html=True)
    
    # Reset button
    if st.button("New Fact Check"):
        st.session_state.showing_results = False
        st.experimental_rerun()
