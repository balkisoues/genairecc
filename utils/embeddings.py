"""
utils/embeddings.py"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union, List, Dict, Optional
import hashlib
import pickle
import os

from config.settings import Config


class EmbeddingManager:
    """
    embedding generation 
    """
    
    def __init__(self, model_name: Optional[str] = None, use_cache: bool = True):
        """
        embedding manager.
        """
        self.model_name = model_name or Config.model.embedding_model  #incase we dont have a model name defined
        self.model = SentenceTransformer(self.model_name)
        self.use_cache = use_cache  #check caching to speed up process 
        
        # cache directory
        self.cache_dir = ".embedding_cache"
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)	#create cache folder if its missing 
        
        # in mem cache, no sidk 
        self._memory_cache: Dict[str, np.ndarray] = {}
        
        print(f"✓ EmbeddingManager initialized (model: {self.model_name})")
    
    
    def encode(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        generate embedding for a single text, input text to embedand returns embedding vector as numpy array
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # check cache 
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(text)
            
            # ceck memory cache 1st
            if cache_key in self._memory_cache:
                return self._memory_cache[cache_key]
            
            # check disk cache
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                self._memory_cache[cache_key] = cached_embedding
                return cached_embedding
        
        # generate new embedding
        embedding = self.model.encode(text)
        
        # save to cache
        if use_cache and self.use_cache:
            cache_key = self._get_cache_key(text)
            self._memory_cache[cache_key] = embedding
            self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        gen embeddings for multiple texts efficiently.
        """
        if not texts:
            return np.array([])
        
        #filter out empty texts
        valid_texts = [t for t in texts if t and isinstance(t, str)]
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        # batch encode
        embeddings = self.model.encode(
            valid_texts, 
            batch_size=batch_size,
            show_progress_bar=len(valid_texts) > 100
        )
        
        return embeddings
    
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        compute cosine similarity between two embeddings.
          embedding1: First embedding vector and ebedding 2 and returns sim score between -1 and 1
        """
        # normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
       find most similar embeddings to a query.
        """
        # Compute similarities
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.compute_similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        #sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    
    def get_embedding_dimension(self) -> int:
        """get the dimension of embeddings from this model"""
        return self.model.get_sentence_embedding_dimension()
    
    
    def clear_cache(self):
        """Clear both memory and disk cache"""
        self._memory_cache.clear()
        
        if self.use_cache and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                os.remove(os.path.join(self.cache_dir, file))
            print("✓ Cache cleared")
    
    
  
    def _get_cache_key(self, text: str) -> str:
        """generate cache key from text"""
        # Use hash of text + model name for cache key
        content = f"{self.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """save embedding to disk cache"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"⚠ Cache save failed: {e}")
    
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embedding from disk cache"""
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"⚠ Cache load failed: {e}")
        
        return None


_global_embedder: Optional[EmbeddingManager] = None


def get_embedder() -> EmbeddingManager:
    """get or create global embedding manager instance"""
    global _global_embedder
    if _global_embedder is None:
        _global_embedder = EmbeddingManager()
    return _global_embedder


def encode_text(text: str) -> np.ndarray:
    """
    Convenience function to encode a single text.
    Uses global embedder instance.
    """
    return get_embedder().encode(text)


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Convenience function to encode multiple texts.
    Uses global embedder instance.
    """
    return get_embedder().encode_batch(texts)


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.
    """
    embedder = get_embedder()
    emb1 = embedder.encode(text1)
    emb2 = embedder.encode(text2)
    return embedder.compute_similarity(emb1, emb2)


def profile_to_embedding(profile: Dict) -> np.ndarray:
    """
    convert learner profile dictionary to embedding.
    standardized across all agents.
    """
    # create natural language description
    text = (
        f"Student {profile.get('id_student', 'unknown')} "
        f"in module {profile.get('code_module', 'unknown')}. "
        f"Average score {profile.get('avg_score', 0):.1f}, "
        f"engagement level {profile.get('engagement_level', 'medium')}, "
        f"{profile.get('total_clicks', 0)} total clicks, "
        f"{profile.get('num_prev_attempts', 0)} previous attempts. "
        f"Learning style: {profile.get('learning_style', 'balanced')}. "
        f"Final result: {profile.get('final_result', 'unknown')}."
    )
    
    return encode_text(text)


def concept_to_embedding(concept: str, difficulty: str = "intermediate") -> np.ndarray:
    """
    convert learning concept to embedding.
    """
    text = f"{difficulty} level content about {concept}"
    return encode_text(text)


########################################### TESTING


if __name__ == "__main__":
    """test the embedding utilities"""
    
    print("="*70)
    print("EMBEDDING UTILITIES TEST")
    
    # TEST 1: Single embedding
    print("\n[Test 1] Single text embedding:")
    embedder = EmbeddingManager()
    text = "Machine learning is a subset of artificial intelligence"
    embedding = embedder.encode(text)
    print(f"  Text: {text[:50]}...")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding dim: {embedder.get_embedding_dimension()}")
    
    # TEST 2: Batch embedding
    print("\n[Test 2] Batch embedding:")
    texts = [
        "Python programming fundamentals",
        "Advanced data structures",
        "Machine learning algorithms"
    ]
    embeddings = embedder.encode_batch(texts)
    print(f"  Number of texts: {len(texts)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # TEST 3: Similarity
    print("\n[Test 3] Semantic similarity:")
    text1 = "Python programming"
    text2 = "Coding in Python"
    text3 = "Quantum physics"
    
    sim_high = compute_similarity(text1, text2)
    sim_low = compute_similarity(text1, text3)
    
    print(f"  '{text1}' vs '{text2}': {sim_high:.3f}")
    print(f"  '{text1}' vs '{text3}': {sim_low:.3f}")
    
    # TEST 4: Profile embedding
    print("\n[Test 4] Profile to embedding:")
    profile = {
        'id_student': '28400',
        'code_module': 'AAA',
        'avg_score': 75.5,
        'engagement_level': 'high',
        'total_clicks': 4500,
        'num_prev_attempts': 0,
        'learning_style': 'visual',
        'final_result': 'Pass'
    }
    
    profile_emb = profile_to_embedding(profile)
    print(f"  Profile embedding shape: {profile_emb.shape}")
    
    # TEST 5: Cache
    print("\n[Test 5] Cache test:")
    text = "Test caching functionality"
    
    import time
    start = time.time()
    emb1 = embedder.encode(text, use_cache=True)
    time1 = time.time() - start
    
    start = time.time()
    emb2 = embedder.encode(text, use_cache=True)  # SHOUDL BE CACHEDDDD
    time2 = time.time() - start
    
    print(f"  First encoding: {time1*1000:.2f}ms")
    print(f"  Cached encoding: {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x")
    print(f"  Embeddings match: {np.allclose(emb1, emb2)}")
    

    print("✓ All tests passed!")
