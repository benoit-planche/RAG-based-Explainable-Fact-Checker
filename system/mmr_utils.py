"""
Utility functions for MMR (Maximum Marginal Relevance) implementation
"""

import numpy as np
from typing import List, Dict

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def calculate_mmr_score(query_embedding: List[float], doc_embedding: List[float], 
                       selected_embeddings: List[List[float]], lambda_param: float = 0.5) -> float:
    """
    Calculate MMR score for a document
    
    Args:
        query_embedding: Query embedding
        doc_embedding: Document embedding
        selected_embeddings: List of already selected document embeddings
        lambda_param: Balance between relevance (λ) and diversity (1-λ)
    
    Returns:
        MMR score
    """
    # Relevance score (similarity to query)
    relevance = cosine_similarity(query_embedding, doc_embedding)
    
    # Diversity score (minimum similarity to already selected documents)
    if not selected_embeddings:
        diversity = 1.0  # First document has maximum diversity
    else:
        similarities = [cosine_similarity(doc_embedding, selected_emb) for selected_emb in selected_embeddings]
        diversity = 1.0 - max(similarities)  # 1 - max similarity = diversity
    
    # MMR score = λ * relevance + (1-λ) * diversity
    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
    
    return mmr_score

def mmr_similarity_search(embeddings: List[List[float]], query_embedding: List[float], 
                         k: int = 5, lambda_param: float = 0.5) -> List[int]:
    """
    Perform MMR-based similarity search
    
    Args:
        embeddings: List of document embeddings
        query_embedding: Query embedding
        k: Number of documents to retrieve
        lambda_param: MMR parameter (0.0 = max diversity, 1.0 = max relevance)
    
    Returns:
        List of document indices selected by MMR
    """
    selected_indices = []
    selected_embeddings = []
    
    # First document: highest relevance
    similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
    first_idx = max(range(len(similarities)), key=lambda i: similarities[i])
    selected_indices.append(first_idx)
    selected_embeddings.append(embeddings[first_idx])
    
    # Remaining documents: MMR selection
    remaining_indices = [i for i in range(len(embeddings)) if i != first_idx]
    
    for _ in range(min(k - 1, len(remaining_indices))):
        best_mmr_score = -1
        best_idx = -1
        
        for idx in remaining_indices:
            mmr_score = calculate_mmr_score(
                query_embedding, 
                embeddings[idx], 
                selected_embeddings, 
                lambda_param
            )
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            selected_embeddings.append(embeddings[best_idx])
            remaining_indices.remove(best_idx)
    
    return selected_indices

def calculate_diversity_metrics(embeddings: List[List[float]]) -> Dict[str, float]:
    """
    Calculate diversity metrics for a set of embeddings
    
    Args:
        embeddings: List of document embeddings
    
    Returns:
        Dictionary with diversity metrics
    """
    if len(embeddings) < 2:
        return {
            'average_similarity': 1.0,
            'min_similarity': 1.0,
            'max_similarity': 1.0,
            'diversity_score': 0.0
        }
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    max_sim = np.max(similarities)
    diversity_score = 1.0 - avg_sim  # Higher diversity = lower average similarity
    
    return {
        'average_similarity': avg_sim,
        'min_similarity': min_sim,
        'max_similarity': max_sim,
        'diversity_score': diversity_score
    }

def simple_similarity_search(embeddings: List[List[float]], query_embedding: List[float], k: int = 5) -> List[int]:
    """Simple similarity search for comparison with MMR"""
    similarities = []
    for i, embedding in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, embedding)
        similarities.append((i, sim))
    
    # Sort by similarity and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in similarities[:k]] 