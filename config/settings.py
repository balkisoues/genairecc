"""
config/settings.py
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """LLM and embedding model configuration"""
    
    
    llm_model: str = "qwen3:8b"  
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_provider: str = "ollama"  
    
    
    embedding_model: str = "all-MiniLM-L6-v2" 
    embedding_dimension: int = 384
    
 


@dataclass
class ProfilingConfig:
    """profiling agent configuration"""
    
    # clustering config 
    n_clusters: int = 5
    clustering_algorithm: str = "kmeans"
    random_state: int = 42
    
    # feature eng
    feature_names: List[str] = None
    
    def __post_init__(self):
        if self.feature_names is None:
            self.feature_names = [
                'avg_score',
                'total_clicks', 
                'num_prev_attempts',
                'engagement_level',
                'final_result'
            ]
    
    # similartiy search
    similarity_top_k: int = 5
    similarity_threshold: float = 0.7


@dataclass
class PathPlanningConfig:
    """path planning agent configuration"""
    

    max_path_length: int = 10
    include_prerequisites: bool = True
    

    difficulty_levels: List[str] = None
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ['beginner', 'intermediate', 'advanced']
    

    default_unit_duration: int = 30
    

    beginner_threshold: float = 50.0
    intermediate_threshold: float = 70.0
    advanced_threshold: float = 85.0


@dataclass
class ContentGenerationConfig:
    """ccontent generation agent config"""
    
    # RAG settings
    rag_top_k: int = 3
    rag_similarity_threshold: float = 0.6
    
    # Content generation
    max_explanation_length: int = 500  # words
    include_examples: bool = True
    
    # Quiz generation
    questions_per_unit: int = 3
    question_types: List[str] = None
    
    def __post_init__(self):
        if self.question_types is None:
            self.question_types = ['multiple_choice', 'true_false', 'short_answer']
    
    # content adaptation
    adapt_to_learning_style: bool = True
    adapt_to_difficulty: bool = True


@dataclass
class RecommendationConfig:
    """recommendation agent configuration"""
    
    # Rank
    top_n_recommendations: int = 5
    diversity_weight: float = 0.3
    relevance_weight: float = 0.7
    
    # Filter
    use_collaborative_filtering: bool = True
    use_content_based_filtering: bool = True
    use_llm_reranking: bool = True
    
 
    cold_start_strategy: str = "popularity"  # or "random", "diverse"


@dataclass
class XAIConfig:
    """XAI Agent configuration """
    
    # XAI methods
    use_shap: bool = True
    use_lime: bool = True
    use_counterfactuals: bool = True
    
    # SHAP settings
    shap_max_samples: int = 100
    shap_feature_importance_threshold: float = 0.1
    
    # LIME settings
    lime_num_features: int = 5
    lime_num_samples: int = 5000
    
    # counterfactual settings
    num_counterfactuals: int = 3
    
    # explanation generation
    explanation_style: str = "pedagogical"  # or "technical"
    max_explanation_length: int = 300  # words
    
    # multi-stakeholder support
    generate_learner_view: bool = True
    generate_instructor_view: bool = True


@dataclass
class VectorDBConfig:
    """chromadb config"""
    
    # collections
    learner_collection: str = "learners"
    content_collection: str = "educational_content"
    
    # storage
    persist_directory: str = "./chroma_db"
    
    # search
    search_type: str = "similarity"  # or "mmr"
    distance_metric: str = "cosine"  # or "l2", "ip"


@dataclass
class SystemConfig:
    """WHOLE system configuration"""
    
    # data paths
    data_path: str = "data/processed/learner_profiles.csv"
    output_path: str = "outputs/"
    
    # log
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_agent_traces: bool = True
    
    # performance
    batch_size: int = 32
    enable_caching: bool = True
    
    # feedback loop
    enable_feedback_loop: bool = True
    feedback_update_frequency: int = 10  # Update after N interactions




class Config:
    """
    Main configuratio
    """
    
    model = ModelConfig()
    profiling = ProfilingConfig()
    path_planning = PathPlanningConfig()
    content_generation = ContentGenerationConfig()
    recommendation = RecommendationConfig()
    xai = XAIConfig()
    vector_db = VectorDBConfig()
    system = SystemConfig()
    
    @classmethod
    def load_from_file(cls, config_path: str):
        """Load configuration from YAML/JSON file"""
        
        pass
    
    @classmethod
    def save_to_file(cls, config_path: str):
        """Save current configuration to file"""
       
        pass


# Example usage
if __name__ == "__main__":
    print("System Configuration:")
    print(f"  LLM Model: {Config.model.llm_model}")
    print(f"  Embedding Model: {Config.model.embedding_model}")
    print(f"  Number of Clusters: {Config.profiling.n_clusters}")
    print(f"  RAG Top K: {Config.content_generation.rag_top_k}")
    print(f"  Use SHAP: {Config.xai.use_shap}")
    print(f"  Data Path: {Config.system.data_path}")
