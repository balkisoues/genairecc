"""
shared.py 
"""
from typing import TypedDict, Annotated, List, Dict, Any
import operator


class SystemState(TypedDict):
    """
    shared state across all agents in the system.
    
    flow:
    1. Input Layer: learner_id, learner_data, learner_query etc
    2. Processing: Each agent reads inputs, writes outputs
    3. Output Layer: final_output with all results
    4. Feedback Layer: user_interactions, performance_metrics
    
    Communication:
    - Profiling Agent: writes learner_profile
    - Path Planning reads: learner_profile & writes: learning_path
    - Content Gen reads: learning_path, learner_profile & writes: generated_content
    - Recommendation reads: generated_content, learner_profile & writes: recommendations
    - XAI reads: ALL & writes: explanations
    """
    
    # input layer
    learner_id: str
    """Unique identifier for the learner (e.g., '28400_AAA_2013J')"""
    
    learner_data: Dict[str, Any]
    """raw learner data """
    
    learner_query: str
    """learning goal or whatever"""
    
    
    #orchestration
    agent_logs: Annotated[List[str], operator.add]
    """
    comm logs
    """
    
    next_agent: str
    """
    which agent to execute next
    """
    
    
    # agents output
    learner_profile: Dict[str, Any]
    """
    output from 1st agent it coontains: embedding, features, cluster_id, learning_style, preferences
    """
    
    learning_path: List[Dict[str, Any]]
    """
    output from 2nd afent list of learning units with: concept, difficulty, duration, prerequisites
    """
    
    generated_content: List[Dict[str, Any]]
    """
    output from gen agent w/ personalized materials: explanations, quizzes, examples
    """
    
    recommendations: Dict[str, Any]
    """
    output of recc agent w/ ranked resources, next steps, external resources
    """
    
    explanations: Dict[str, Any]
    """
    output XAI Agent, fature importance, reasoning traces, counterfactuals, explanations
    """
    
    
    # output layerrrrrrrrrrr
    final_output: Dict[str, Any]
    """
    final output
    """
    
    
    # feedback layer
    user_interactions: List[Dict[str, Any]]
    """
   capture user interactions
    """
    
    performance_metrics: Dict[str, float]
    """
	
	performance tracking accruacy etc etc    """


def create_initial_state(learner_id: str, learner_data: Dict[str, Any]) -> SystemState:
    """
    creating an inittial state for learners  w/ arguemnts:
        learner_id
        learner_data
    and it returns:
        initialized SystemState ready for agent processing
    """
    return {
        # input
        'learner_id': learner_id,
        'learner_data': learner_data,
        'learner_query': learner_data.get('query', ''),
        
        # orchestration
        'agent_logs': [],
        'next_agent': 'profiling',  # start with profiling
        
        # agent outputs (empty initially)
        'learner_profile': {},
        'learning_path': [],
        'generated_content': [],
        'recommendations': {},
        'explanations': {},
        
        # output
        'final_output': {},
        
        # d´feedback
        'user_interactions': [],
        'performance_metrics': {}
    }


#exp
if __name__ == "__main__":
    # test state creation
    sample_data = {
        'id_student': '28400',
        'code_module': 'AAA',
        'avg_score': 75.5,
        'engagement_level': 'high',
        'total_clicks': 4500
    }
    
    state = create_initial_state('28400_AAA_2013J', sample_data)
    
    print("Initial State Created:")
    print(f"  Learner ID: {state['learner_id']}")
    print(f"  Next Agent: {state['next_agent']}")
    print(f"  Agent Logs: {state['agent_logs']}")
