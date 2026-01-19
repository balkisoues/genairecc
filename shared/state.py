# shared/state.py
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LearnerState:
    """State representing a learner throughout the pipeline"""
    learner_id: str
    query: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    learning_path: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    generated_content: Optional[Dict[str, Any]] = None
    explanations: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Agent execution tracking
    profiling_complete: bool = False
    path_planning_complete: bool = False
    content_generation_complete: bool = False
    recommendation_complete: bool = False
    xai_complete: bool = False
    
    # Error tracking
    errors: List[str] = field(default_factory=list)


class SystemState(TypedDict, total=False):
    """
    System-wide state for LangGraph orchestration
    Compatible with LangGraph's state management
    """
    # Learner identification
    learner_id: str
    learner_data: Optional[Dict[str, Any]]  # ADD THIS LINE
    query: Optional[str]
    
    # Agent outputs
    profile: Optional[Dict[str, Any]]
    learner_profile: Optional[Dict[str, Any]]  # ADD THIS (alias for profile)
    learning_path: Optional[List[Dict[str, Any]]]
    recommendations: Optional[Dict[str, Any]]
    generated_content: Optional[List[Dict[str, Any]]]  # Changed to List
    explanations: Optional[Dict[str, Any]]
    
    # Feedback and metadata
    feedback: Optional[Dict[str, Any]]
    timestamp: str
    
    # Execution tracking
    profiling_complete: bool
    path_planning_complete: bool
    content_generation_complete: bool
    recommendation_complete: bool
    xai_complete: bool
    
    # Error handling
    errors: List[str]
    
    # Agent messages (for LangGraph)
    messages: List[Dict[str, Any]]
    agent_logs: List[str]  # ADD THIS LINE


def create_initial_state(learner_id: str, learner_data: Dict[str, Any]) -> SystemState:
    """
    Create initial state for a learner entering the system
    
    Args:
        learner_id: Unique identifier for the learner
        learner_data: Dictionary containing learner information from CSV
        
    Returns:
        SystemState object with default values
    """
    return SystemState(
        learner_id=learner_id,
        learner_data=learner_data,  # ADD THIS LINE
        query=learner_data.get('query', None),
        profile=None,
        learning_path=None,
        recommendations=None,
        generated_content=None,
        explanations=None,
        feedback=None,
        timestamp=datetime.now().isoformat(),
        profiling_complete=False,
        path_planning_complete=False,
        content_generation_complete=False,
        recommendation_complete=False,
        xai_complete=False,
        errors=[],
        messages=[],
        agent_logs=[]  # ADD THIS LINE for the display_results function
    )


def update_state(state: SystemState, **kwargs) -> SystemState:
    """
    Update state with new values
    
    Args:
        state: Current SystemState
        **kwargs: Fields to update
        
    Returns:
        Updated SystemState
    """
    updated_state = state.copy()
    for key, value in kwargs.items():
        if key in SystemState.__annotations__:
            updated_state[key] = value
    return updated_state


def learner_state_to_system_state(learner_state: LearnerState) -> SystemState:
    """
    Convert LearnerState to SystemState for LangGraph compatibility
    
    Args:
        learner_state: LearnerState object
        
    Returns:
        SystemState object
    """
    return SystemState(
        learner_id=learner_state.learner_id,
        query=learner_state.query,
        profile=learner_state.profile,
        learning_path=learner_state.learning_path,
        recommendations=learner_state.recommendations,
        generated_content=learner_state.generated_content,
        explanations=learner_state.explanations,
        feedback=learner_state.feedback,
        timestamp=learner_state.timestamp.isoformat(),
        profiling_complete=learner_state.profiling_complete,
        path_planning_complete=learner_state.path_planning_complete,
        content_generation_complete=learner_state.content_generation_complete,
        recommendation_complete=learner_state.recommendation_complete,
        xai_complete=learner_state.xai_complete,
        errors=learner_state.errors,
        messages=[]
    )


def system_state_to_learner_state(system_state: SystemState) -> LearnerState:
    """
    Convert SystemState back to LearnerState
    
    Args:
        system_state: SystemState object
        
    Returns:
        LearnerState object
    """
    return LearnerState(
        learner_id=system_state['learner_id'],
        query=system_state.get('query'),
        profile=system_state.get('profile'),
        learning_path=system_state.get('learning_path'),
        recommendations=system_state.get('recommendations'),
        generated_content=system_state.get('generated_content'),
        explanations=system_state.get('explanations'),
        feedback=system_state.get('feedback'),
        timestamp=datetime.fromisoformat(system_state.get('timestamp', datetime.now().isoformat())),
        profiling_complete=system_state.get('profiling_complete', False),
        path_planning_complete=system_state.get('path_planning_complete', False),
        content_generation_complete=system_state.get('content_generation_complete', False),
        recommendation_complete=system_state.get('recommendation_complete', False),
        xai_complete=system_state.get('xai_complete', False),
        errors=system_state.get('errors', [])
    )
