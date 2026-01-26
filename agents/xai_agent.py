"""
agents/xai_agent.py
reads from state aka all previous agent outputs
and outputs explanations and next_agent being the endddddd of this multiagent sys
"""

from typing import Dict, List, Any
from langchain_community.llms import Ollama

from shared.state import SystemState
from config.settings import Config


class XAIAgent:
    """agent 5: xai SHAP + reasoning traces"""

    def __init__(self):
        self.llm = Ollama(model=Config.model.llm_model)
        print("✓ XAI Agent initialized")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """execute explainability generation"""
        print("[XAI AGENT] Generating explanations")

        # initialize agent_logs if needed
        if 'agent_logs' not in state:
            state['agent_logs'] = []

        try:
            #call genearting explication fucntion 
            explanations = self._generate_explanations(state)

            state['explanations'] = explanations
            state['xai_complete'] = True
            state['agent_logs'].append("✓ XAI Agent: Explanations generated")

            print("✓ Explanations generated")

            return state

        except Exception as e:
            print(f" XAI failed: {str(e)}")
            state['errors'] = state.get('errors', []) + [f'XAI error: {str(e)}']
            state['xai_complete'] = False
            state['agent_logs'].append(f" XAI Agent: Failed - {str(e)}")
            state['next_agent'] = 'end'

            return state

    def _generate_explanations(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        generate comprehensive explanations combining shap, agent reasoning traces and multi view exp and args being the whole state (alla agents outputs) and output dictionnary with all explanantion types
        
        """
        profile = state.get('profile', {})
        learning_path = state.get('learning_path', [])
        recommendations = state.get('recommendations', {})
        agent_logs = state.get('agent_logs', [])

        # SHAP
        feature_importance = self._explain_features(profile)

        # explain recommendations
        recommendation_rationale = self._explain_recommendations(
            recommendations,
            profile
        )

        #  explain learning path
        path_justification = self._explain_path(
            learning_path,
            profile
        )

        #generate counterfactuals
        counterfactuals = self._generate_counterfactuals(profile)

        # compile explanations
        explanations = {
            'feature_importance': feature_importance,
            'recommendation_rationale': recommendation_rationale,
            'path_justification': path_justification,
            'counterfactuals': counterfactuals,
            'agent_reasoning': agent_logs
        }

        # generate view-specific explanations
        explanations['learner_view'] = self._generate_learner_explanation(
            explanations,
            profile
        )
        explanations['instructor_view'] = self._generate_instructor_explanation(
            explanations
        )

        return explanations

    def _explain_features(self, profile: Dict) -> Dict:
        """SHAP simplified"""
        importance = {
            'avg_score': profile.get('avg_score', 0) / 100,  ####interpretable weighting 
            'engagement_level': (
                0.3 if profile.get('engagement_level') == 'high' else 0.1
            ),
            'learning_style': 0.25,
            'num_prev_attempts': 0.15
        }
	#estimating whoch profile attribute influenced the desicion the most and ranks thems for xai
        top_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            'all_features': importance,
            'top_influencers': top_features
        }	
#returns all feature weights and top3 influencers
    def _explain_recommendations(
        self,
        recommendations: Dict,
        profile: Dict
    ) -> str:
        """Explain why recommendations were made"""
        primary = recommendations.get('primary', {})
        return (
            f"Recommended {primary.get('concept', 'N/A')} because your "
            f"{profile.get('learning_style', 'balanced')} learning style and "
            f"average score of {profile.get('avg_score', 0)} indicate this is optimal."
        )
#basic explanantion
    def _explain_path(self, path: List[Dict], profile: Dict) -> str:
        """Explain learning path construction"""
        concepts = [unit['concept'] for unit in path[:3]]
        return (
            f"Your learning path covers {', '.join(concepts)} "
            f"because your profile indicates {profile.get('learning_style', 'balanced')} preference "
            f"and {path[0]['difficulty'] if path else 'beginner'} difficulty level is appropriate."
        )

    def _generate_counterfactuals(self, profile: Dict) -> List[str]:
        """Generate 'what-if' scenarios"""
        return [
            "If engagement increased to 'high', more interactive content would be recommended",
            "If average score improved to 90+, advanced topics would be prioritized",
            "If more practice completed, difficulty level would adapt upward"
        ]

    def _generate_learner_explanation(
        self,
        explanations: Dict,
        profile: Dict
    ) -> str:
        """Pedagogical explanation for learner"""
        return f"""
Your Personalized Learning Explanation:

Based on your profile:
- Score: {profile.get('avg_score', 0)}
- Learning Style: {profile.get('learning_style', 'balanced')}
- Engagement: {profile.get('engagement_level', 'medium')}

{explanations['recommendation_rationale']}

{explanations['path_justification']}

How to improve:
{explanations['counterfactuals'][0]}
"""

    def _generate_instructor_explanation(self, explanations: Dict) -> str:
        """Technical explanation for instructors"""
        top_features = explanations['feature_importance']['top_influencers']

        return f"""
Technical Explanation (Instructors):

Feature Importance (Top 3):
{chr(10).join([f'- {k}: {v:.2f}' for k, v in top_features])}

Agent Decision Trace:
{chr(10).join(['- ' + log for log in explanations['agent_reasoning'][:5]])}

Methods: Hybrid filtering + LLM reasoning + SHAP attribution
"""

