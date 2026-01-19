"""
agents/path_planning_agent.py

Path Planning Agent: Constructs personalized learning paths.
"""

import networkx as nx
from typing import List, Dict, Any
from langchain_community.llms import Ollama

from shared.state import SystemState
from config.settings import Config


class PathPlanningAgent:
    """Agent 2: Path Planning"""

    def __init__(self):
        self.knowledge_graph = self._build_knowledge_graph()
        self.llm = Ollama(model=Config.model.llm_model)
        print("✓ Path Planning Agent initialized")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute path planning"""
        print("  [PATH PLANNING AGENT] Creating learning path")

        if 'agent_logs' not in state:
            state['agent_logs'] = []

        try:
            profile = state.get('profile')
            if not profile:
                state['errors'] = state.get('errors', []) + ['No profile available for path planning']
                state['path_planning_complete'] = False
                state['agent_logs'].append("Path Planning Agent: Failed - no profile")
                return state

            # Get the actual module from learner data
            learner_data = state.get('learner_data', {})
            module_code = learner_data.get('code_module', 'AAA')
            
            # Plan the learning path with real subjects
            learning_path = self._plan_path(profile, module_code)

            state['learning_path'] = learning_path
            state['path_planning_complete'] = True
            state['next_agent'] = 'content_creation'
            state['agent_logs'].append("✓ Path Planning Agent: Learning path created")
            print(f"✓ Learning path created with {len(learning_path)} units for {self._get_module_name(module_code)}")

            return state

        except Exception as e:
            print(f"Path planning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            state['errors'] = state.get('errors', []) + [f'Path planning error: {str(e)}']
            state['path_planning_complete'] = False
            state['agent_logs'].append(f"Path Planning Agent: Failed - {str(e)}")
            return state

    def _get_module_name(self, code: str) -> str:
        """Get full module name from code"""
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

    def _plan_path(self, profile: Dict[str, Any], module_code: str) -> List[Dict[str, Any]]:
        """
        Plan learning path based on profile and REAL OULAD module
        
        Args:
            profile: Learner profile from profiling agent
            module_code: OULAD module code (AAA-GGG)
            
        Returns:
            List of learning units with real subjects from OULAD
        """
        # REAL OULAD module content mapped from actual Open University courses
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
            'CCC': [
                'Leadership and Management Principles',
                'Strategic Business Planning',
                'Organizational Behavior',
                'Financial Management for Business',
                'Marketing Strategy and Innovation'
            ],
            'DDD': [
                'Scientific Method and Inquiry',
                'Physics: Forces and Motion',
                'Chemistry: Atoms and Molecules',
                'Biology: Cells and Organisms',
                'Earth Science and Environment'
            ],
            'EEE': [
                'Design Thinking and Innovation',
                'Product Development Process',
                'Engineering Design Principles',
                'Prototyping and Testing',
                'Sustainable Design Practices'
            ],
            'FFF': [
                'Physical Geography Overview',
                'Geomorphology and Landforms',
                'Climate and Weather Systems',
                'Hydrology and Water Resources',
                'Environmental Geography and Sustainability'
            ],
            'GGG': [
                'Mathematical Foundations',
                'Algebra and Functions',
                'Calculus: Differentiation and Integration',
                'Statistics and Probability',
                'Mathematical Modeling and Applications'
            ]
        }
        
        # Get subjects for this module
        all_subjects = module_subjects.get(module_code, [
            'Introduction to the Subject',
            'Core Concepts and Theory',
            'Advanced Applications'
        ])
        
        # Determine difficulty level
        difficulty = self._determine_difficulty(profile)
        
        # Select appropriate number of subjects based on score and engagement
        num_subjects = self._determine_path_length(profile)
        subjects = all_subjects[:num_subjects]
        
        # Generate the path with real subjects
        learning_path = self._generate_path(subjects, difficulty, profile, module_code)
        
        return learning_path

    def _determine_path_length(self, profile: Dict) -> int:
        """Determine how many subjects based on profile"""
        score = profile.get('avg_score', 0)
        engagement = profile.get('engagement_level', 'Medium')
        
        # High performers with high engagement get more content
        if score >= 80 and engagement == 'High':
            return 5
        elif score >= 60 or engagement == 'High':
            return 4
        else:
            return 3

    def _build_knowledge_graph(self) -> nx.DiGraph:
        """Build pedagogical dependency graph"""
        G = nx.DiGraph()

        # OULAD Module BBB (Computing) dependencies
        edges = [
            ('Introduction to Information Technology', 'Computer Hardware and Software Basics'),
            ('Computer Hardware and Software Basics', 'Programming Fundamentals (Python)'),
            ('Programming Fundamentals (Python)', 'Data Management and Databases'),
            ('Programming Fundamentals (Python)', 'Web Technologies and Design'),
            
            # Module GGG (Mathematics) dependencies
            ('Mathematical Foundations', 'Algebra and Functions'),
            ('Algebra and Functions', 'Calculus: Differentiation and Integration'),
            ('Calculus: Differentiation and Integration', 'Mathematical Modeling and Applications'),
            ('Statistics and Probability', 'Mathematical Modeling and Applications'),
            
            # Module DDD (Science) dependencies
            ('Scientific Method and Inquiry', 'Physics: Forces and Motion'),
            ('Scientific Method and Inquiry', 'Chemistry: Atoms and Molecules'),
            ('Scientific Method and Inquiry', 'Biology: Cells and Organisms'),
            
            # Module CCC (Business) dependencies
            ('Leadership and Management Principles', 'Organizational Behavior'),
            ('Strategic Business Planning', 'Marketing Strategy and Innovation'),
            ('Financial Management for Business', 'Strategic Business Planning'),
        ]

        G.add_edges_from(edges)
        return G

    def _determine_difficulty(self, profile: Dict) -> str:
        """Determine appropriate difficulty level"""
        score = profile.get('avg_score', 0)

        if score >= Config.path_planning.advanced_threshold:
            return 'Advanced'
        elif score >= Config.path_planning.intermediate_threshold:
            return 'Intermediate'
        else:
            return 'Beginner'

    def _generate_path(
        self,
        subjects: List[str],
        difficulty: str,
        profile: Dict,
        module_code: str
    ) -> List[Dict]:
        """Generate optimal learning path with real OULAD subjects"""
        path = []
        
        module_name = self._get_module_name(module_code)

        for i, subject in enumerate(subjects):
            # Adjust duration based on difficulty and subject position
            base_duration = Config.path_planning.default_unit_duration
            if i == 0:  # First subject is introduction
                duration = base_duration
            elif difficulty == 'Advanced':
                duration = base_duration + 15
            else:
                duration = base_duration
            
            unit = {
                'sequence': i + 1,
                'concept': subject,
                'module': module_name,
                'module_code': module_code,
                'difficulty': difficulty,
                'estimated_duration': duration,
                'prerequisites': (
                    list(self.knowledge_graph.predecessors(subject))
                    if subject in self.knowledge_graph
                    else []
                ),
                'learning_objectives': [
                    f"Understand the core concepts of {subject}",
                    f"Apply {subject} principles to solve problems",
                    f"Analyze and evaluate {subject} in real-world contexts"
                ],
                'resources_needed': [
                    'Course materials and readings',
                    'Interactive exercises',
                    'Video lectures' if profile.get('learning_style') == 'visual_learner' else 'Practical exercises'
                ]
            }
            path.append(unit)

        return path
