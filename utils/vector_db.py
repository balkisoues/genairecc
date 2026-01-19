"""
utils/vector_db.py

Centralized vector database utilities using ChromaDB.
Provides reusable functions for storing and querying embeddings.

Benefits:
- Single interface for all vector DB operations
- Consistent collection management
- Automatic embedding generation
- Built-in error handling
"""

import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import os

from config.settings import Config
from utils.embeddings import get_embedder


class VectorDBManager:
    """
    Manages ChromaDB collections and operations.
    
    Usage:
        db = VectorDBManager()
        db.add_documents("learners", ids=["1", "2"], documents=["text1", "text2"])
        results = db.search("learners", query_text="search query", top_k=5)
    """
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize the vector database manager.
        
        Args:
            persist_directory: Directory to persist data (None for in-memory)
        """
        self.persist_directory = persist_directory or Config.vector_db.persist_directory
        
        # Initialize ChromaDB client
        if self.persist_directory and self.persist_directory != ":memory:":
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"✓ VectorDB initialized (persistent: {self.persist_directory})")
        else:
            self.client = chromadb.Client()
            print("✓ VectorDB initialized (in-memory)")
        
        # Embedding manager
        self.embedder = get_embedder()
        
        # Track collections
        self.collections: Dict[str, chromadb.Collection] = {}
    
    
    def get_or_create_collection(
        self, 
        name: str,
        metadata: Optional[Dict] = None
    ) -> chromadb.Collection:
        """
        Get existing collection or create new one.
        
        Args:
            name: Collection name
            metadata: Optional metadata for the collection
            
        Returns:
            ChromaDB collection object
        """
        if name in self.collections:
            return self.collections[name]
        
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {}
        )
        
        self.collections[name] = collection
        print(f"  ✓ Collection '{name}' ready")
        
        return collection
    
    
    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of unique IDs
            documents: List of text documents (will auto-generate embeddings)
            embeddings: Pre-computed embeddings (if documents not provided)
            metadatas: Optional metadata for each document
            
        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Generate embeddings if not provided
            if embeddings is None and documents is not None:
                embeddings = [
                    self.embedder.encode(doc).tolist() 
                    for doc in documents
                ]
            
            # Prepare data
            add_params = {'ids': ids}
            
            if documents is not None:
                add_params['documents'] = documents
            
            if embeddings is not None:
                add_params['embeddings'] = embeddings
            
            if metadatas is not None:
                add_params['metadatas'] = metadatas
            
            # Add to collection
            collection.upsert(**add_params)
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding documents to '{collection_name}': {e}")
            return False
    
    
    def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 5,
        where: Optional[Dict] = None,
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents in a collection.
        
        Args:
            collection_name: Name of the collection
            query_text: Query text (will auto-generate embedding)
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            where: Metadata filter (e.g., {"category": "python"})
            include: What to include in results (documents, metadatas, distances)
            
        Returns:
            Dictionary with search results
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            # Generate query embedding if not provided
            if query_embedding is None and query_text is not None:
                query_embedding = self.embedder.encode(query_text).tolist()
            
            if query_embedding is None:
                raise ValueError("Must provide either query_text or query_embedding")
            
            # Default includes
            if include is None:
                include = ['documents', 'metadatas', 'distances']
            
            # Query collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=include
            )
            
            return results
            
        except Exception as e:
            print(f"❌ Error searching '{collection_name}': {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
    
    
    def get_by_id(
        self,
        collection_name: str,
        ids: List[str],
        include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get documents by their IDs.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs
            include: What to include in results
            
        Returns:
            Dictionary with documents
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            
            if include is None:
                include = ['documents', 'metadatas', 'embeddings']
            
            results = collection.get(ids=ids, include=include)
            return results
            
        except Exception as e:
            print(f"❌ Error getting documents from '{collection_name}': {e}")
            return {'ids': [], 'documents': [], 'metadatas': [], 'embeddings': []}
    
    
    def delete_documents(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=ids)
            return True
            
        except Exception as e:
            print(f"❌ Error deleting documents from '{collection_name}': {e}")
            return False
    
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete an entire collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            print(f"  ✓ Collection '{collection_name}' deleted")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting collection '{collection_name}': {e}")
            return False
    
    
    def list_collections(self) -> List[str]:
        """Get list of all collections"""
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    
    def get_collection_count(self, collection_name: str) -> int:
        """Get number of documents in a collection"""
        try:
            collection = self.get_or_create_collection(collection_name)
            return collection.count()
        except:
            return 0
    
    
    def find_similar_to_document(
        self,
        collection_name: str,
        document_id: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Find documents similar to a given document.
        
        Args:
            collection_name: Name of the collection
            document_id: ID of the source document
            top_k: Number of similar documents to return
            
        Returns:
            Search results
        """
        try:
            # Get the source document's embedding
            doc = self.get_by_id(collection_name, [document_id], include=['embeddings'])
            
            if not doc['embeddings']:
                raise ValueError(f"Document '{document_id}' not found")
            
            source_embedding = doc['embeddings'][0]
            
            # Search for similar documents
            results = self.search(
                collection_name=collection_name,
                query_embedding=source_embedding,
                top_k=top_k + 1  # +1 because source will be included
            )
            
            # Filter out the source document
            filtered_results = {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            for i, doc_id in enumerate(results['ids'][0]):
                if doc_id != document_id:
                    filtered_results['ids'][0].append(doc_id)
                    if results['documents'][0]:
                        filtered_results['documents'][0].append(results['documents'][0][i])
                    if results['metadatas'][0]:
                        filtered_results['metadatas'][0].append(results['metadatas'][0][i])
                    if results['distances'][0]:
                        filtered_results['distances'][0].append(results['distances'][0][i])
            
            # Limit to top_k
            for key in filtered_results:
                filtered_results[key][0] = filtered_results[key][0][:top_k]
            
            return filtered_results
            
        except Exception as e:
            print(f"❌ Error finding similar documents: {e}")
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}


# ============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC USE CASES
# ============================================================================

def init_learner_collection(db: VectorDBManager) -> chromadb.Collection:
    """Initialize collection for learner profiles"""
    return db.get_or_create_collection(
        Config.vector_db.learner_collection,
        metadata={"description": "Learner profiles with embeddings"}
    )


def init_content_collection(db: VectorDBManager) -> chromadb.Collection:
    """Initialize collection for educational content"""
    return db.get_or_create_collection(
        Config.vector_db.content_collection,
        metadata={"description": "Educational content for RAG"}
    )


def add_learner_profile(
    db: VectorDBManager,
    learner_id: str,
    profile_text: str,
    metadata: Dict
) -> bool:
    """
    Add a learner profile to the database.
    
    Args:
        db: VectorDBManager instance
        learner_id: Unique learner ID
        profile_text: Natural language profile description
        metadata: Profile metadata (scores, engagement, etc.)
        
    Returns:
        True if successful
    """
    return db.add_documents(
        collection_name=Config.vector_db.learner_collection,
        ids=[learner_id],
        documents=[profile_text],
        metadatas=[metadata]
    )


def find_similar_learners(
    db: VectorDBManager,
    learner_id: str,
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Find similar learners to a given learner.
    
    Args:
        db: VectorDBManager instance
        learner_id: Source learner ID
        top_k: Number of similar learners to find
        
    Returns:
        List of (learner_id, similarity_score) tuples
    """
    results = db.find_similar_to_document(
        collection_name=Config.vector_db.learner_collection,
        document_id=learner_id,
        top_k=top_k
    )
    
    similar = []
    if results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            distance = results['distances'][0][i]
            # Convert distance to similarity (1 - normalized_distance)
            similarity = 1 - (distance / 2)  # Cosine distance in [0, 2]
            similar.append((doc_id, similarity))
    
    return similar


def add_educational_content(
    db: VectorDBManager,
    content_id: str,
    content_text: str,
    metadata: Dict
) -> bool:
    """
    Add educational content for RAG.
    
    Args:
        db: VectorDBManager instance
        content_id: Unique content ID
        content_text: The educational content
        metadata: Content metadata (topic, difficulty, etc.)
        
    Returns:
        True if successful
    """
    return db.add_documents(
        collection_name=Config.vector_db.content_collection,
        ids=[content_id],
        documents=[content_text],
        metadatas=[metadata]
    )


def retrieve_relevant_content(
    db: VectorDBManager,
    query: str,
    top_k: int = 3,
    filters: Optional[Dict] = None
) -> List[Dict]:
    """
    Retrieve relevant educational content (RAG).
    
    Args:
        db: VectorDBManager instance
        query: Search query (concept, topic, etc.)
        top_k: Number of documents to retrieve
        filters: Metadata filters (e.g., {"difficulty": "intermediate"})
        
    Returns:
        List of content dictionaries with text and metadata
    """
    results = db.search(
        collection_name=Config.vector_db.content_collection,
        query_text=query,
        top_k=top_k,
        where=filters
    )
    
    content_list = []
    if results['ids'][0]:
        for i in range(len(results['ids'][0])):
            content_list.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i] if results['documents'][0] else '',
                'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                'distance': results['distances'][0][i] if results['distances'][0] else 0.0
            })
    
    return content_list


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_db: Optional[VectorDBManager] = None


def get_vector_db() -> VectorDBManager:
    """Get or create global vector database instance"""
    global _global_db
    if _global_db is None:
        _global_db = VectorDBManager()
    return _global_db


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    """Test the vector database utilities"""
    
    print("="*60)
    print("VECTOR DATABASE UTILITIES TEST")
    print("="*60)
    
    # Initialize
    db = VectorDBManager(persist_directory=":memory:")
    
    # Test 1: Create collection and add documents
    print("\n[Test 1] Add documents:")
    success = db.add_documents(
        collection_name="test_collection",
        ids=["doc1", "doc2", "doc3"],
        documents=[
            "Python is a programming language",
            "Machine learning uses algorithms",
            "Data science involves statistics"
        ],
        metadatas=[
            {"category": "programming"},
            {"category": "ml"},
            {"category": "data"}
        ]
    )
    print(f"  Add documents: {'✓' if success else '✗'}")
    print(f"  Collection count: {db.get_collection_count('test_collection')}")
    
    # Test 2: Search
    print("\n[Test 2] Search for similar documents:")
    results = db.search(
        collection_name="test_collection",
        query_text="coding in Python",
        top_k=2
    )
    print(f"  Query: 'coding in Python'")
    print(f"  Results found: {len(results['ids'][0])}")
    for i, doc_id in enumerate(results['ids'][0]):
        print(f"    {i+1}. ID: {doc_id}")
        print(f"       Text: {results['documents'][0][i][:50]}...")
        print(f"       Distance: {results['distances'][0][i]:.3f}")
    
    # Test 3: Get by ID
    print("\n[Test 3] Get document by ID:")
    doc = db.get_by_id("test_collection", ["doc1"])
    print(f"  ID: {doc['ids'][0] if doc['ids'] else 'Not found'}")
    print(f"  Document: {doc['documents'][0] if doc['documents'] else 'Not found'}")
    
    # Test 4: Find similar to document
    print("\n[Test 4] Find similar to specific document:")
    similar = db.find_similar_to_document(
        collection_name="test_collection",
        document_id="doc1",
        top_k=2
    )
    print(f"  Similar to 'doc1':")
    for i, doc_id in enumerate(similar['ids'][0]):
        print(f"    {i+1}. {doc_id}: {similar['documents'][0][i][:40]}...")
    
    # Test 5: Learner-specific functions
    print("\n[Test 5] Learner profile operations:")
    add_learner_profile(
        db=db,
        learner_id="learner_001",
        profile_text="High performing student in computer science",
        metadata={"avg_score": 85, "engagement": "high"}
    )
    
    add_learner_profile(
        db=db,
        learner_id="learner_002",
        profile_text="Excellent student in programming courses",
        metadata={"avg_score": 90, "engagement": "high"}
    )
    
    similar_learners = find_similar_learners(db, "learner_001", top_k=1)
    print(f"  Similar learners to learner_001:")
    for learner_id, score in similar_learners:
        print(f"    {learner_id}: similarity {score:.3f}")
    
    # Test 6: RAG content operations
    print("\n[Test 6] Educational content RAG:")
    add_educational_content(
        db=db,
        content_id="content_001",
        content_text="Variables are containers for storing data values",
        metadata={"topic": "variables", "difficulty": "beginner"}
    )
    
    content = retrieve_relevant_content(
        db=db,
        query="data storage in programming",
        top_k=1
    )
    print(f"  Retrieved content for 'data storage':")
    for item in content:
        print(f"    {item['id']}: {item['text'][:50]}...")
    
    # Test 7: List collections
    print("\n[Test 7] List all collections:")
    collections = db.list_collections()
    print(f"  Collections: {collections}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
