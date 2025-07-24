import os
import uuid
import shutil
import time
from typing import List, Generator
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import chromadb

# Directory setup
UPLOAD_DIR = "uploaded_pdfs"
CHROMA_DIR = "chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "clear_flag" not in st.session_state:
    st.session_state.clear_flag = False
if "client" not in st.session_state:
    st.session_state.client = None
if "uploader_counter" not in st.session_state:
    st.session_state.uploader_counter = 0

# Initialize components
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=1)
llm = OllamaLLM(model="qwen2.5:latest", temperature=0.3, num_gpu=1)

class FiDRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.technical_terms = None
        self.seen_documents = set()
        
    def retrieve(self, query: str, k: int = 8) -> List[Document]:
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        docs = [doc for doc, _ in docs_with_scores]
        
        doc_ids = [doc.metadata.get('source', '') + str(doc.metadata.get('page', '')) for doc in docs]
        self.seen_documents.update(doc_ids)
        
        if self.technical_terms is None:
            self._calculate_tfidf_terms(docs)
        
        for doc in docs:
            doc.metadata['combined_score'] = self._calculate_combined_score(query, doc)
        
        docs.sort(key=lambda x: x.metadata['combined_score'], reverse=True)
        # print(docs[:2], " scores: " , [doc.metadata['combined_score'] for doc in docs])
        return docs[:2]  # Return only top 2 documents
    
    def _calculate_tfidf_terms(self, docs: List[Document]):
        texts = [doc.page_content for doc in docs]
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf_scores = defaultdict(float)
        
        for i, doc in enumerate(texts):
            feature_index = tfidf_matrix[i,:].nonzero()[1]
            for idx in feature_index:
                avg_tfidf_scores[feature_names[idx]] += tfidf_matrix[i, idx]
        
        for term in avg_tfidf_scores:
            avg_tfidf_scores[term] /= len(texts)
        
        sorted_terms = sorted(avg_tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        self.technical_terms = [term for term, score in sorted_terms[:10]]
    
    def _calculate_combined_score(self, query: str, doc: Document) -> float:
        semantic_score = self._normalize_score(
            self.vectorstore.similarity_search_with_score(query, k=1)[0][1]
        )
        
        if self.technical_terms:
            technical_score = sum(
                1 for term in self.technical_terms 
                if term.lower() in doc.page_content.lower()
            ) / len(self.technical_terms)
            # print("technical_score: ", technical_score)
        else:
            technical_score = 0.5
        
        doc_id = doc.metadata.get('source', '') + str(doc.metadata.get('page', ''))
        recency_score = 0.2 if doc_id in self.seen_documents else 0.8
        
        # print(0.6*semantic_score + 0.2*technical_score + 0.2*recency_score)
        return 0.6*semantic_score + 0.2*technical_score + 0.2*recency_score
    
    def _normalize_score(self, score: float) -> float:
        return 1 / (1 + np.exp(-score))

class FiDChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def run(self, question: str) -> Generator[str, None, None]:
        docs = self.retriever.retrieve(question)
        fid_prompt = self._build_fid_prompt(question, docs)
        
        # Create a generator that yields the response in chunks
        response = self.llm.stream(fid_prompt)
        for chunk in response:
            yield chunk
    
    def _build_fid_prompt(self, question: str, docs: List[Document]) -> str:
        prompt = """You are an expert analyst capable of multi-document synthesis. Using the following documents:\n\n"""
        
        for i, doc in enumerate(docs):
            prompt += f"""--- Document {i+1} (Score: {doc.metadata['combined_score']:.2f}) ---\n{doc.page_content}\n\n"""
        
        prompt += f"""\nPlease answer this question:\n{question}\n\nYour response should:\n"""
        prompt += """1. Synthesize information from these documents\n2. Resolve any contradictions\n3. Cite relevant standards\n4. Follow this structure:\n"""
        prompt += """- Analysis Summary\n- Recommended Approach\n- Reference Standards\n"""
        
        return prompt

def process_uploaded_pdfs(uploaded_files):
    documents = []
    
    for uploaded_file in uploaded_files:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            for chunk in chunks:
                chunk.metadata['source'] = uploaded_file.name
            documents.extend(chunks)
        except Exception as e:
            os.remove(file_path)
            raise e
    
    if not documents:
        raise ValueError("No valid documents were processed")
    
    # Initialize Chroma client with explicit settings
    st.session_state.client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Delete any existing collection to ensure clean start
    try:
        st.session_state.client.delete_collection("pdf_collection")
    except:
        pass
    
    # Create vector store with explicit client
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        client=st.session_state.client,
        collection_name="pdf_collection"
    )
    
    return vectorstore

def clear_documents():
    try:
        # Reset session state first to release resources
        st.session_state.vectorstore = None
        st.session_state.documents_processed = False
        st.session_state.messages = []
        st.session_state.clear_flag = True
        
        # Explicitly delete the Chroma collection if it exists
        if st.session_state.client:
            try:
                st.session_state.client.delete_collection("pdf_collection")
            except Exception as e:
                st.warning(f"Warning: Could not delete collection - {str(e)}")
        
        # Close the Chroma client
        if st.session_state.client:
            try:
                st.session_state.client = None
            except:
                pass
        
        # Give time for resources to be released
        time.sleep(2)
        
        # Remove all uploaded PDFs
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Remove Chroma database with retries and force deletion
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if os.path.exists(CHROMA_DIR):
                    # On Windows, we need to handle file locking explicitly
                    if os.name == 'nt':
                        os.system(f'rmdir /s /q "{CHROMA_DIR}"')
                    else:
                        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to clear ChromaDB after {max_retries} attempts: {str(e)}")
                    break
                time.sleep(2)
        
        # Increment the uploader key to reset the file uploader
        st.session_state.uploader_counter += 1
        st.session_state.clear_flag = False
        
        st.success("All documents have been completely cleared. You can upload new documents.")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing documents: {str(e)}")

def main():
    st.set_page_config(page_title="PDF Q&A with FiD RAG", page_icon="📚")
    st.title("📚 PDF Q&A with Fusion-in-Decoder RAG")
    
    # Sidebar for document management
    with st.sidebar:
        st.header("Document Management")
        
        # File uploader with unique key that changes when cleared
        uploader_key = f"file_uploader_{st.session_state.uploader_counter}"
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key=uploader_key
        )
        
        # Clear documents button
        if st.button("Clear All Documents"):
            clear_documents()
    
    # Only process documents if we have files and haven't just cleared
    if uploaded_files and not st.session_state.documents_processed and not st.session_state.clear_flag:
        with st.spinner("Processing documents..."):
            try:
                vectorstore = process_uploaded_pdfs(uploaded_files)
                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = FiDRetriever(vectorstore)
                st.session_state.chain = FiDChain(st.session_state.retriever, llm)
                
                st.session_state.documents_processed = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I'm ready to answer questions about your documents!"
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        if not st.session_state.chain:
            with st.chat_message("assistant"):
                st.warning("Please upload PDF documents first")
        else:
            with st.chat_message("assistant"):
                response = st.write_stream(st.session_state.chain.run(prompt))
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()