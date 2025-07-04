"""
Utility functions for Ollama integration
"""

import requests
import json
import numpy as np
from typing import List, Dict, Any
import re
from ollama_config import config

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url="http://localhost:11434", model="llama2:7b"):
        self.base_url = base_url
        self.model = model
        self.tokens_used = 0
    
    def generate(self, prompt, temperature=0.7, max_tokens=2000):
        """Generate text using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            # Estimate tokens (rough approximation)
            self.tokens_used += len(prompt.split()) + len(result.get('response', '').split())
            return result.get('response', '')
        except Exception as e:
            print(f"Error calling Ollama: {str(e)}")
            return ""

class OllamaEmbeddings:
    """Simple embeddings class to replace OpenAI embeddings"""
    
    def __init__(self, model="llama2:7b"):
        self.model = model
        self.base_url = config.base_url
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get('embedding', [])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            # Return a dummy embedding if Ollama doesn't support embeddings
            return [0.0] * 384  # Default dimension

class SimpleTextSplitter:
    """Simple text splitter to replace LangChain's RecursiveCharacterTextSplitter"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        splits = []
        for doc in documents:
            text_chunks = self.split_text(doc.get('page_content', doc.get('content', '')))
            for chunk in text_chunks:
                split_doc = doc.copy()
                split_doc['page_content'] = chunk
                splits.append(split_doc)
        return splits

class SimpleDocumentLoader:
    """Simple document loader to replace LangChain loaders"""
    
    @staticmethod
    def load_text_file(file_path: str) -> Dict[str, Any]:
        """Load a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                'page_content': content,
                'metadata': {'source': file_path}
            }
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
    
    @staticmethod
    def load_directory(directory_path: str, glob_pattern="**/*.txt") -> List[Dict[str, Any]]:
        """Load all text files from a directory"""
        import os
        import glob
        
        documents = []
        pattern = os.path.join(directory_path, glob_pattern)
        
        for file_path in glob.glob(pattern, recursive=True):
            doc = SimpleDocumentLoader.load_text_file(file_path)
            if doc:
                documents.append(doc)
        
        return documents

def format_prompt(template: str, **kwargs) -> str:
    """Simple prompt formatting to replace PromptTemplate"""
    return template.format(**kwargs)

def extract_verdict(text: str) -> str:
    """Extract verdict from analysis text"""
    verdict_patterns = [
        r'VERDICT:\s*(\w+)',
        r'Verdict:\s*(\w+)',
        r'Result:\s*(\w+)',
        r'Conclusion:\s*(\w+)'
    ]
    
    for pattern in verdict_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return "UNKNOWN" 