import networkx as nx
from typing import List, Dict, Any
from langchain_community.llms import Ollama

from shared.state import SystemState
from config.settings import Config


class PathPlanningAgent:
    """agent 2: Paath planninnnnnnnng """

    def __init__(self):
        self.knowledge_graph = self._build_knowledge_graph()
        self.llm = Ollama(model=Config.model.llm_model)
        print("✓ Path Planning Agent initialized")

    
    # ain entry point

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """path planning"""
        print("  [PATH PLANNING AGENT] Creating learning path")

        if 'agent_logs' not in state:
            state['agent_logs'] = []

        try:
            profile = state.get('profile')
            if not profile:
                return self._fail(state, "No profile available for path planning")

            learner_data = state.get('learner_data', {})
            module_code = learner_data.get('code_module', 'AAA')

            learning_path = self._plan_path(profile, module_code)

            state['learning_path'] = learning_path
            state['path_planning_complete'] = True
            state['next_agent'] = 'content_generation'   
            state['agent_logs'].append("✓ Path Planning Agent: Learning path created")

            print(f"✓ Learning path created with {len(learning_path)} units for {self._get_module_name(module_code)}")
            return state

        except Exception as e:
            return self._fail(state, str(e))

    # debugging

    def _fail(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        state['errors'] = state.get('errors', []) + [message]
        state['path_planning_complete'] = False
        state['agent_logs'].append(f"Path Planning Agent: Failed - {message}")
        return state

    # path planning pipeline

    def _plan_path(self, profile: Dict[str, Any], module_code: str) -> List[Dict[str, Any]]:
        subjects = self._get_module_subjects(module_code)
	#get module subjects 
        difficulty = self._determine_difficulty(profile)
        #deermine begginer itnermidiate oradvanced 
        num_subjects = self._determine_path_length(profile)
        #decide how many topics to include 
        selected = subjects[:num_subjects]
        # enforce prerequisites using knowledge graph
        ordered_subjects = self._order_by_prerequisites(selected)

        return self._generate_path(ordered_subjects, difficulty, profile, module_code)
#generates structured learning unitss 
    
    #module content
    def _get_module_subjects(self, module_code: str) -> List[str]:
        module_subjects = {
            'AAA': [
                'Network Fundamentals and Architecture',
                'Data Communication Protocols',
                'Network Security and Protection',
                'Wireless and Mobile Networks',
                'Network Management and Troubleshooting'
            ],
            'BBB': [
                'Introduction to Information Technology',
                'Computer Hardware and Software Basics',
                'Programming Fundamentals (Python)',
                'Web Technologies and Design',
                'Data Management and Databases'
            ],
            # ... etc
        }
        return module_subjects.get(module_code, [
            'Introduction to the Subject',
            'Core Concepts and Theory',
            'Advanced Applications'
        ])

    def _get_module_name(self, code: str) -> str:
        module_names = {
            'AAA': 'Understanding Computer Networks',
            'BBB': 'Introduction to Computing and Information Technology',
            'CCC': 'Developing Skills for Business Leadership',
            'DDD': 'Discovering Science',
            'EEE': 'Design and Innovation',
            'FFF': 'Understanding Physical Geography',
            'GGG': 'Introduction to Mathematics'
        }
        return module_names.get(code, f'Module {code}')



    def _determine_difficulty(self, profile: Dict) -> str:
    	
    	#uses learner score to decide the diffciulty of the module to give him
        score = profile.get('avg_score', 0)
        
        if score >= Config.path_planning.advanced_threshold:
            return 'Advanced'
        elif score >= Config.path_planning.intermediate_threshold:
            return 'Intermediate'
        else:
            return 'Beginner'

    def _determine_path_length(self, profile: Dict) -> int:
    
    	#mm chose but to decide path length
        score = profile.get('avg_score', 0)
        engagement = profile.get('engagement_level', 'medium')  # FIXED

        if score >= 80 and engagement == 'high':
            return 5
        elif score >= 60 or engagement == 'high':
            return 4
        else:
            return 3

    # knowledge graph + prerequisites
   
    def _build_knowledge_graph(self) -> nx.DiGraph: 	#creates a directected graph to explain you must learn this before this etc 
        G = nx.DiGraph()
        edges = [
            ('Introduction to Information Technology', 'Computer Hardware and Software Basics'),
            ('Computer Hardware and Software Basics', 'Programming Fundamentals (Python)'),
            ('Programming Fundamentals (Python)', 'Data Management and Databases'),
            ('Programming Fundamentals (Python)', 'Web Technologies and Design'),
            # ... other edges
        ]
        G.add_edges_from(edges)
        return G

    def _order_by_prerequisites(self, subjects: List[str]) -> List[str]:
        # Use topological sorting to enforce dependencies
        try:
            ordered = list(nx.topological_sort(self.knowledge_graph))
            return [s for s in ordered if s in subjects]
        except Exception:
            # fallback
            return subjects


    # generate path units

    def _generate_path(
        self,
        subjects: List[str],
        difficulty: str,
        profile: Dict,
        module_code: str
    ) -> List[Dict]:
        path = []
        module_name = self._get_module_name(module_code)

        for i, subject in enumerate(subjects):
            duration = self._compute_duration(i, difficulty)

            resources = self._choose_resources(profile)

            unit = {
                'sequence': i + 1,
                'concept': subject,
                'module': module_name,
                'module_code': module_code,
                'difficulty': difficulty,
                'estimated_duration': duration,
                'prerequisites': list(self.knowledge_graph.predecessors(subject)),
                'learning_objectives': [
                    f"Understand the core concepts of {subject}",
                    f"Apply {subject} principles to solve problems",
                    f"Analyze and evaluate {subject} in real-world contexts"
                ],
                'resources_needed': resources
            }
            path.append(unit)
#this is what content generation takes as input later on (next agent)
        return path

    def _compute_duration(self, index: int, difficulty: str) -> int:
    	
    	#estimates time per unit 
        base = Config.path_planning.default_unit_duration
        if index == 0:
            return base
        return base + 15 if difficulty == 'Advanced' else base

    def _choose_resources(self, profile: Dict) -> List[str]:
    
    	#chosing ressources based on elarning styles
        learning_style = profile.get('learning_style', 'balanced_learner')

        if learning_style in ['active_explorer', 'visual_learner']:
            resource = 'Video lectures'
        else:
            resource = 'Practical exercises'

        return [
            'Course materials and readings',
            'Interactive exercises',
            resource
        ]

