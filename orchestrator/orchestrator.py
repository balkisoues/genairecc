"""
orchestrator/orchestrator.py

Coordinates the execution of all agents using LangGraph.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END

from shared.state import SystemState
from agents.profiling_agent import ProfilingAgent
from agents.path_planning_agent import PathPlanningAgent
from agents.content_generation_agent import ContentGenerationAgent
from agents.recommendation_agent import RecommendationAgent
from agents.xai_agent import XAIAgent


class Orchestrator:
    """Orchestrates multi-agent workflow using LangGraph"""
    
    def __init__(self):
        """Initialize Orchestrator with all agents"""
        print("="*70)
        print(" ORCHESTRATOR: Initializing Multi-Agent System")
        print("="*70)
        
        # Initialize agents
        self.profiling_agent = ProfilingAgent()
        self.path_planning_agent = PathPlanningAgent()
        self.content_generation_agent = ContentGenerationAgent()
        self.recommendation_agent = RecommendationAgent()
        self.xai_agent = XAIAgent()
        
        print("✓ All agents initialized")
        
        # Build LangGraph workflow
        self._build_workflow()
        print("✓ LangGraph workflow compiled")
    
    def _build_workflow(self):
        """Build LangGraph workflow connecting all agents"""
        # Create workflow graph
        workflow = StateGraph(SystemState)
        
        # Add nodes for each agent
        workflow.add_node("profiling", self._profiling_node)
        workflow.add_node("path_planning", self._path_planning_node)
        workflow.add_node("content_generation", self._content_generation_node)
        workflow.add_node("recommendation", self._recommendation_node)
        workflow.add_node("xai", self._xai_node)
        
        # Define workflow edges (sequential flow)
        workflow.set_entry_point("profiling")
        workflow.add_edge("profiling", "path_planning")
        workflow.add_edge("path_planning", "content_generation")
        workflow.add_edge("content_generation", "recommendation")
        workflow.add_edge("recommendation", "xai")
        workflow.add_edge("xai", END)
        
        # Compile the workflow
        self.app = workflow.compile()
    
    def _profiling_node(self, state: SystemState) -> SystemState:
        """Profiling agent node"""
        try:
            result = self.profiling_agent.execute(state)
            print(f"  ✓ Profiling node completed")
            return result
        except Exception as e:
            print(f"   Profiling node failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Profiling error: {str(e)}']
            state['profiling_complete'] = False
            return state
    
    def _path_planning_node(self, state: SystemState) -> SystemState:
        """Path planning agent node"""
        try:
            result = self.path_planning_agent.execute(state)
            print(f"  ✓ Path planning node completed")
            return result
        except Exception as e:
            print(f"   Path planning node failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Path planning error: {str(e)}']
            state['path_planning_complete'] = False
            return state
    
    def _content_generation_node(self, state: SystemState) -> SystemState:
        """Content generation agent node"""
        try:
            result = self.content_generation_agent.execute(state)
            print(f"  ✓ Content generation node completed")
            return result
        except Exception as e:
            print(f"   Content generation node failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Content generation error: {str(e)}']
            state['content_generation_complete'] = False
            return state
    
    def _recommendation_node(self, state: SystemState) -> SystemState:
        """Recommendation agent node"""
        try:
            result = self.recommendation_agent.execute(state)
            print(f"  ✓ Recommendation node completed")
            return result
        except Exception as e:
            print(f"   Recommendation node failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Recommendation error: {str(e)}']
            state['recommendation_complete'] = False
            return state
    
    def _xai_node(self, state: SystemState) -> SystemState:
        """XAI agent node"""
        try:
            result = self.xai_agent.execute(state)
            print(f"  ✓ XAI node completed")
            return result
        except Exception as e:
            print(f"   XAI node failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'XAI error: {str(e)}']
            state['xai_complete'] = False
            return state
    
    def run(self, state: SystemState) -> SystemState:
        """
        Run the multi-agent workflow
        
        Args:
            state: Initial system state
            
        Returns:
            Final system state after all agents execute
        """
        print("\n Starting multi-agent workflow...\n")
        
        try:
            # Run the workflow
            final_state = self.app.invoke(state)
            
            print("\n Workflow completed successfully!")
            return final_state
            
        except Exception as e:
            print(f"\n Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Workflow error: {str(e)}']
            raise
    
    def get_graph_visualization(self):
        """Return Mermaid diagram of the graph"""
        try:
            return self.app.get_graph().draw_mermaid()
        except Exception:
            return "Graph visualization not available"
