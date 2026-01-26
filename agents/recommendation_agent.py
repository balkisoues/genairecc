"""
agents/recommendation_agent.py
recommendation agent: ranks and selects learning resources.

input reads fromstate and output a dict with recc 
"""

from typing import List, Dict, Any
from langchain_community.llms import Ollama

from shared.state import SystemState
from config.settings import Config


class RecommendationAgent:
    """agent 4: recommendation with hybrid filtering"""

    def __init__(self):
        self.llm = Ollama(model=Config.model.llm_model)
        print("✓ Recommendation Agent initialized")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """execute recommendation"""
        print(" [RECOMMENDATION AGENT] Generating recommendations")

        # initialize agent_logs if needed
        if 'agent_logs' not in state:
            state['agent_logs'] = []

        try:
            profile = state.get('profile')
            learning_path = state.get('learning_path')

            if not profile or not learning_path:
                state['errors'] = state.get('errors', []) + ['Missing profile or learning path']
                state['recommendation_complete'] = False
                state['agent_logs'].append(
                    " Recommendation Agent: Failed - missing data"
                )
                state['next_agent"'] = 'end'
                return state

            # generate recommendations
            recommendations = self._generate_recommendations(profile, learning_path)

            state['recommendations'] = recommendations
            state['recommendation_complete'] = True
            state['agent_logs'].append(
                "✓ Recommendation Agent: Recommendations generated"
            )

            state['next_agent"'] = 'explainability'
            print(f"✓ Recommendations generated")
            
            return state

        except Exception as e:
            print(f" Recommendation failed: {str(e)}")
            state['errors'] = state.get('errors', []) + [f'Recommendation error: {str(e)}']
            state['recommendation_complete'] = False
            state['agent_logs'].append(f" Recommendation Agent: Failed - {str(e)}")
            return state

    def _rank_content(self, content: List[Dict], profile: Dict) -> List[Dict]:
        """rank content by relevance"""
        scored_content = []
	#score starts at 0 and we add point if difficulty matches learner lvel or if learner has high engageent level
        for item in content:
            score = 0.0

            # diff match
            if item['difficulty'] == self._get_optimal_difficulty(profile):
                score += 0.5

            # engagement prediction
            if profile.get('engagement_level') == 'high':
                score += 0.3

            item['relevance_score'] = score
            scored_content.append(item)

        return sorted(scored_content, key=lambda x: x['relevance_score'], reverse=True)

    def _get_optimal_difficulty(self, profile: Dict) -> str: #maps learner scores -> diff
        """determine optimal difficulty"""
        score = profile.get('avg_score', 0)
        if score >= 85:
            return 'advanced'
        elif score >= 70:
            return 'intermediate'
        return 'beginner'

    def _generate_next_steps(self, profile: Dict, path: List[Dict]) -> List[str]:
        """Generate actionable next steps"""
        return [
            f"Complete {path[0]['concept']} module" if path else "Start learning",
            "Take practice quiz",
            "Review feedback and adapt"
        ]

    def _recommend_external(self, unit: Dict) -> List[Dict]:
        """Recommend external resources"""
        return [
            {
                'title': f"Tutorial on {unit.get('concept', 'basics')}",
                'type': 'video',
                'url': 'https://example.com',
                'duration': '15 min'
            }
        ]

    def _generate_recommendations(self, profile: Dict, learning_path: List[Dict]) -> Dict[str, Any]:
        """ core logic """
        recommendations = []

        ranked_units = self._rank_content(learning_path, profile)
	# + external ressources
        for unit in ranked_units:
            external = self._recommend_external(unit)
            recommendations.append({
                'unit': unit['concept'],
                'recommended_content': external,
                'difficulty': unit['difficulty']
            })

        next_steps = self._generate_next_steps(profile, learning_path)
        return {
            'recommendations': recommendations,
            'next_steps': next_steps
        }

