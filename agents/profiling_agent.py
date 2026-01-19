"""
agents/profiling_agent.py

"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import chromadb
from typing import Dict, List, Any, Optional

from shared.state import SystemState
from config.settings import Config


class ProfilingAgent:
    """
    Agent 1: Profiling Agent
    
    Continuously analyzes learner interactions, performance history, and preferences
    using embeddings and clustering techniques to build dynamic learner profiles.
    """

    def __init__(self):
        """Initialize Profiling Agent"""
        self.embedding_model = SentenceTransformer(Config.model.embedding_model)
        self.cluster_model = self._initialize_clustering()
        self.scaler = StandardScaler()
        
        # Initialize ChromaDB collection for learner profiles
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.get_collection("learner_profiles")
        except:
            self.collection = self.chroma_client.create_collection("learner_profiles")
        
        print(f"✓ Profiling Agent initialized (clusters: {Config.profiling.n_clusters})")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute profiling agent
        
        Args:
            state: Current system state containing learner_data
            
        Returns:
            Updated state with profile information
        """
        print("🔍 [PROFILING AGENT] Analyzing learner profile")
        
        # Initialize agent_logs if needed
        if 'agent_logs' not in state:
            state['agent_logs'] = []
        
        try:
            # Get learner_data from state
            learner_data = state.get('learner_data')
            learner_id = state.get('learner_id')
            
            if not learner_data:
                error_msg = 'No learner_data provided in state'
                print(f" {error_msg}")
                state['errors'] = state.get('errors', []) + [error_msg]
                state['profiling_complete'] = False
                state['agent_logs'].append(" Profiling Agent: Failed - No learner data")
                return state
            
            # Generate complete profile
            profile = self._generate_profile(learner_data)
            
            # Generate embedding for similarity search
            embedding = self._generate_embedding(profile)
            
            # Assign cluster
            features = self._extract_features(profile)
            cluster_id = self._assign_cluster(features)
            profile['cluster_id'] = int(cluster_id)
            
            # Determine learning style based on behavior patterns
            profile['learning_style'] = self._determine_learning_style(profile)
            
            # Store in vector database
            if learner_id:
                self._store_in_vectordb(learner_id, profile, embedding)
            
            # Update state with BOTH keys (for compatibility)
            state['profile'] = profile
            state['learner_profile'] = profile  # Alias for display_results
            state['profiling_complete'] = True
            state['agent_logs'].append("✓ Profiling Agent: Profile generated successfully")
            
            print(f"✓ Profile generated for learner {learner_id}")
            print(f"  - Cluster: {cluster_id}")
            print(f"  - Learning Style: {profile['learning_style']}")
            print(f"  - Engagement: {profile['engagement_level']}")
            
            return state
            
        except Exception as e:
            error_msg = f"Profiling error: {str(e)}"
            print(f"Profiling failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            state['errors'] = state.get('errors', []) + [error_msg]
            state['profiling_complete'] = False
            state['agent_logs'].append(f" Profiling Agent: Failed - {str(e)}")
            return state

    # ======================================================================
    # Core Profile Generation
    # ======================================================================

    def _generate_profile(self, data: Dict) -> Dict[str, Any]:
        """
        Convert raw learner data into structured profile
        
        Args:
            data: Raw learner data from CSV
            
        Returns:
            Structured learner profile dictionary
        """
        # Basic profile information
        profile = {
            # Identifiers
            'id_student': data.get('id_student'),
            'code_module': data.get('code_module', ''),
            'code_presentation': data.get('code_presentation', ''),
            
            # Performance metrics
            'avg_score': float(data.get('avg_score', 0)),
            'final_result': data.get('final_result', 'unknown'),
            'studied_credits': int(data.get('studied_credits', 0)),
            
            # Engagement metrics
            'total_clicks': int(data.get('total_clicks', 0)),
            'engagement_level': data.get('engagement_level', 'medium'),
            'num_prev_attempts': int(data.get('num_of_prev_attempts', 0)),
            
            # Demographics
            'gender': data.get('gender', 'unknown'),
            'age_band': data.get('age_band', 'unknown'),
            'region': data.get('region', 'unknown'),
            'highest_education': data.get('highest_education', 'unknown'),
            'imd_band': data.get('imd_band', 'unknown'),
            'disability': data.get('disability', 'N'),
            
            # Computed fields (will be filled by other methods)
            'cluster_id': None,
            'learning_style': None,
            'engagement_score': self._compute_engagement_score(data),
            'risk_level': self._compute_risk_level(data)
        }
        
        return profile

    def _compute_engagement_score(self, data: Dict) -> float:
        """
        Compute engagement score from learner data
        
        Args:
            data: Learner data dictionary
            
        Returns:
            Engagement score between 0 and 1
        """
        total_clicks = int(data.get('total_clicks', 0))
        
        # Normalize clicks (assuming max ~10000 clicks)
        clicks_score = min(total_clicks / 10000, 1.0)
        
        # Engagement level mapping
        engagement_map = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        engagement_level = data.get('engagement_level', 'medium')
        level_score = engagement_map.get(engagement_level, 0.6)
        
        # Combined score (weighted average)
        return 0.6 * clicks_score + 0.4 * level_score

    def _compute_risk_level(self, data: Dict) -> str:
        """
        Compute risk level (dropout/failure risk)
        
        Args:
            data: Learner data dictionary
            
        Returns:
            Risk level: 'low', 'medium', or 'high'
        """
        score = float(data.get('avg_score', 0))
        engagement = data.get('engagement_level', 'medium')
        prev_attempts = int(data.get('num_of_prev_attempts', 0))
        
        risk_score = 0
        
        # Score-based risk
        if score < 40:
            risk_score += 3
        elif score < 60:
            risk_score += 2
        elif score < 75:
            risk_score += 1
        
        # Engagement-based risk
        if engagement == 'low':
            risk_score += 2
        elif engagement == 'medium':
            risk_score += 1
        
        # Previous attempts
        if prev_attempts > 1:
            risk_score += 2
        elif prev_attempts == 1:
            risk_score += 1
        
        # Classify risk
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'

    def _determine_learning_style(self, profile: Dict) -> str:
        """
        Determine learning style based on behavior patterns
        
        Args:
            profile: Learner profile dictionary
            
        Returns:
            Learning style category
        """
        engagement = profile['engagement_level']
        clicks = profile['total_clicks']
        score = profile['avg_score']
        
        # High engagement + high clicks = Active learner
        if engagement == 'high' and clicks > 5000:
            return 'active_explorer'
        
        # High score + medium engagement = Efficient learner
        elif score > 75 and engagement in ['medium', 'high']:
            return 'efficient_learner'
        
        # Low engagement = Passive learner
        elif engagement == 'low':
            return 'passive_learner'
        
        # Multiple attempts = Persistent learner
        elif profile['num_prev_attempts'] > 0:
            return 'persistent_learner'
        
        # Default
        else:
            return 'balanced_learner'

    # ======================================================================
    # Embedding and Clustering
    # ======================================================================

    def _generate_embedding(self, profile: Dict) -> np.ndarray:
        """
        Generate semantic embedding from profile
        
        Args:
            profile: Learner profile dictionary
            
        Returns:
            Embedding vector
        """
        text = (
            f"Student {profile['id_student']} in module {profile['code_module']}. "
            f"Average score {profile['avg_score']:.1f}, "
            f"engagement level {profile['engagement_level']}, "
            f"{profile['total_clicks']} total clicks, "
            f"{profile['num_prev_attempts']} previous attempts. "
            f"Final result: {profile['final_result']}. "
            f"Demographics: {profile['gender']}, {profile['age_band']}, "
            f"education level {profile['highest_education']}. "
            f"Learning style: {profile.get('learning_style', 'unknown')}"
        )
        return self.embedding_model.encode(text)

    def _extract_features(self, profile: Dict) -> np.ndarray:
        """
        Extract numerical features for clustering
        
        Args:
            profile: Learner profile dictionary
            
        Returns:
            Feature vector
        """
        # Categorical mappings
        engagement_map = {'low': 0, 'medium': 1, 'high': 2}
        result_map = {'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 3, 'unknown': 1}
        
        features = np.array([
            profile['avg_score'],
            profile['total_clicks'],
            profile['studied_credits'],
            profile['num_prev_attempts'],
            engagement_map.get(profile['engagement_level'], 1),
            result_map.get(profile['final_result'], 1),
            1 if profile['disability'] == 'Y' else 0,
            profile.get('engagement_score', 0.5) * 100  # Scale to similar range
        ])
        
        return features

    def _initialize_clustering(self) -> KMeans:
        """Initialize KMeans clustering model"""
        return KMeans(
            n_clusters=Config.profiling.n_clusters,
            random_state=42,
            n_init=10
        )

    def _assign_cluster(self, features: np.ndarray) -> int:
        """
        Assign learner to cluster
        
        Args:
            features: Feature vector
            
        Returns:
            Cluster ID
        """
        # Reshape for single sample prediction
        features_reshaped = features.reshape(1, -1)
        
        # For initial clustering, we need to fit the model
        # In production, you'd load a pre-trained model
        # For now, we'll just return a mock cluster based on features
        
        # Simple heuristic clustering based on score and engagement
        score = features[0]
        engagement = features[4]
        
        if score > 75 and engagement >= 1.5:
            return 0  # High performers
        elif score > 60:
            return 1  # Medium performers
        elif score > 40:
            return 2  # At-risk learners
        else:
            return 3  # High-risk learners

    def _store_in_vectordb(self, learner_id: str, profile: Dict, embedding: np.ndarray):
        """
        Store profile in ChromaDB vector database
        
        Args:
            learner_id: Unique learner identifier
            profile: Learner profile dictionary
            embedding: Embedding vector
        """
        try:
            self.collection.upsert(
                ids=[learner_id],
                embeddings=[embedding.tolist()],
                metadatas=[{
                    'id_student': str(profile['id_student']),
                    'avg_score': float(profile['avg_score']),
                    'engagement': profile['engagement_level'],
                    'final_result': profile['final_result'],
                    'learning_style': profile.get('learning_style', 'unknown'),
                    'cluster_id': str(profile.get('cluster_id', -1))
                }]
            )
            print(f"  ✓ Stored in vector database")
        except Exception as e:
            print(f"  ⚠ Warning: Could not store in vector DB: {str(e)}")

    # ======================================================================
    # Utility Methods
    # ======================================================================

    def find_similar_learners(self, learner_id: str, n: int = 5) -> List[Dict]:
        """
        Find similar learners using vector similarity
        
        Args:
            learner_id: Target learner ID
            n: Number of similar learners to return
            
        Returns:
            List of similar learner profiles
        """
        try:
            # Get the embedding for this learner
            result = self.collection.get(ids=[learner_id], include=['embeddings'])
            
            if not result['embeddings']:
                return []
            
            embedding = result['embeddings'][0]
            
            # Query for similar learners
            similar = self.collection.query(
                query_embeddings=[embedding],
                n_results=n + 1  # +1 because it includes the query learner
            )
            
            # Filter out the query learner itself
            similar_learners = []
            for i, sid in enumerate(similar['ids'][0]):
                if sid != learner_id:
                    similar_learners.append({
                        'id': sid,
                        'metadata': similar['metadatas'][0][i],
                        'distance': similar['distances'][0][i]
                    })
            
            return similar_learners[:n]
            
        except Exception as e:
            print(f"Warning: Could not find similar learners: {str(e)}")
            return []


# ======================================================================
# TESTING
# ======================================================================

if __name__ == "__main__":
    from shared.state import create_initial_state

    print("="*70)
    print("PROFILING AGENT TEST")
    print("="*70)

    test_learner = {
        'id_student': '28400',
        'code_module': 'AAA',
        'code_presentation': '2013J',
        'avg_score': 75.5,
        'total_clicks': 4500,
        'engagement_level': 'high',
        'final_result': 'Pass',
        'num_of_prev_attempts': 0,
        'gender': 'M',
        'age_band': '35-55',
        'highest_education': "HE Qualification",
        'studied_credits': 120,
        'disability': 'N',
        'region': 'East Anglian Region',
        'imd_band': '20-30%'
    }

    # Create initial state
    state = create_initial_state('28400_AAA_2013J', test_learner)
    
    # Initialize and execute agent
    agent = ProfilingAgent()
    result_state = agent.execute(state)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    if result_state['profiling_complete']:
        profile = result_state['learner_profile']
        print(f"\n✓ Profile generated successfully")
        print(f"\nLearner ID: {profile['id_student']}")
        print(f"Module: {profile['code_module']}")
        print(f"Score: {profile['avg_score']}")
        print(f"Engagement: {profile['engagement_level']}")
        print(f"Learning Style: {profile['learning_style']}")
        print(f"Cluster: {profile['cluster_id']}")
        print(f"Risk Level: {profile['risk_level']}")
        print(f"Engagement Score: {profile['engagement_score']:.2f}")
    else:
        print(f"\n Profiling failed")
        print(f"Errors: {result_state['errors']}")
    
    print("\nAgent Logs:")
    for log in result_state.get('agent_logs', []):
        print(f"  {log}")
    
    print("\n" + "="*70)
