"""
content_generation_agent.py

"""

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from typing import List, Dict, Any

from shared.state import SystemState
from config.settings import Config


class ContentGenerationAgent:
    """agent 3: content generation with LLM + RAG"""

    def __init__(self):
        self.llm = Ollama(model=Config.model.llm_model)
        self.embedding_model = SentenceTransformer(Config.model.embedding_model)

        # RAG: Vector database for educational content
        self.chroma_client = chromadb.Client()
        self.content_db = self.chroma_client.get_or_create_collection(
            name=Config.vector_db.content_collection
        )

        self._load_educational_resources()
        print("✓ Content Generation Agent initialized")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """execute content generation"""
        print(" [CONTENT GENERATION AGENT] Generating learning materials")

        # Initialize agent_logs 
        if 'agent_logs' not in state:
            state['agent_logs'] = []

        try:
            learning_path = state.get('learning_path')
            profile = state.get('profile') or state.get('learner_profile')

            if not learning_path:
                state['errors'] = state.get('errors', []) + ['No learning path available']
                state['content_generation_complete'] = False
                state['agent_logs'].append(
                    "Content Generation Agent: Failed - no learning path"
                )
                state['next_agent"'] = 'end'
                return state

            generated_content = []

            # Generate content for each learning unit
            for unit in learning_path[:3]:  # Limit to first 3 for demo
                retrieved_docs = self._retrieve_content(unit['concept'])
                explanation = self._generate_explanation(unit, profile, retrieved_docs)
                quiz = self._generate_quiz(unit, profile)

                generated_content.append({
                    'concept': unit['concept'],
                    'explanation': explanation,
                    'quiz': quiz,
                    'difficulty': unit['difficulty'],
                    'estimated_time': unit['estimated_duration'],
                    'retrieved_sources': len(retrieved_docs)
                })

            state['generated_content'] = generated_content
            state['content_generation_complete'] = True
            state['agent_logs'].append(
                f"✓ Content Generation Agent: Generated {len(generated_content)} learning units"
            )
            state['next_agent'] = 'recommendation'

            print(f"✓ Generated {len(generated_content)} content items")
            return state

        except Exception as e:
            print(f" Content generation failed: {str(e)}")
            state['errors'] = state.get('errors', []) + [
                f'Content generation error: {str(e)}'
            ]
            state['content_generation_complete'] = False
            state['agent_logs'].append(
                f"Content Generation Agent: Failed - {str(e)}"
            )
            return state

    def _load_educational_resources(self):
        """Load educational content into RAG database"""
        resources = [
            {
                'id': 'res_1',
                'text': 'Fundamental concepts form the basis of learning',
                'topic': 'fundamentals'
            },
            {
                'id': 'res_2',
                'text': 'Practice exercises reinforce understanding',
                'topic': 'practice'
            },
            {
                'id': 'res_3',
                'text': 'Advanced topics build on foundational knowledge',
                'topic': 'advanced_topics'
            }
        ]

        for resource in resources:
            embedding = self.embedding_model.encode(resource['text'])
            self.content_db.upsert(
                ids=[resource['id']],
                documents=[resource['text']],
                embeddings=[embedding.tolist()],
                metadatas=[{'topic': resource['topic']}]
            )

    def _retrieve_content(self, concept: str) -> List[str]:
        """RAG: retrieve relevant content"""
        query_embedding = self.embedding_model.encode(concept)

        results = self.content_db.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=Config.content_generation.rag_top_k
        )

        return results['documents'][0] if results['documents'] else []

    def _generate_explanation(
        self,
        unit: Dict,
        profile: Dict,
        retrieved_docs: List[str]
    ) -> str:
        """Generate personalized explanation using LLM + RAG"""
        context = "\n".join(retrieved_docs)

        prompt = f"""Create a {unit['difficulty']} level explanation of {unit['concept']}.

Learning style: {profile.get('learning_style', 'balanced')}
Reference material: {context}

Provide a clear, concise explanation (2-3 paragraphs) with an example.
"""

        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception:
            return (
                f"Personalized {unit['difficulty']} explanation of {unit['concept']}. "
                f"Adapted for {profile.get('learning_style', 'balanced')} learners."
            )

    def _generate_quiz(self, unit: Dict, profile: Dict) -> List[Dict]:
        """Generate practice quiz"""
        return [
            {
                'question': f"Question about {unit['concept']}",
                'options': ['Option A', 'Option B', 'Option C', 'Option D'],
                'correct_answer': 'A',
                'explanation': 'Explanation of correct answer'
            }
        ]

