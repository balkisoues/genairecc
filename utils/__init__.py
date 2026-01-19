"""
utils/__init__.py

Convenience imports for utility modules.

Usage:
    # Instead of:
    from utils.embeddings import encode_text
    from utils.vector_db import get_vector_db
    
    # You can do:
    from utils import encode_text, get_vector_db
"""

# Embedding utilities
from .embeddings import (
    EmbeddingManager,
    get_embedder,
    encode_text,
    encode_texts,
    compute_similarity,
    profile_to_embedding,
    concept_to_embedding
)

# Vector database utilities
from .vector_db import (
    VectorDBManager,
    get_vector_db,
    init_learner_collection,
    init_content_collection,
    add_learner_profile,
    find_similar_learners,
    add_educational_content,
    retrieve_relevant_content
)

__all__ = [
    # Embedding utilities
    'EmbeddingManager',
    'get_embedder',
    'encode_text',
    'encode_texts',
    'compute_similarity',
    'profile_to_embedding',
    'concept_to_embedding',
    
    # Vector DB utilities
    'VectorDBManager',
    'get_vector_db',
    'init_learner_collection',
    'init_content_collection',
    'add_learner_profile',
    'find_similar_learners',
    'add_educational_content',
    'retrieve_relevant_content',
]
