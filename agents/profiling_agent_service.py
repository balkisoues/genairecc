import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import json

class ProfilingAgent:
    """
    AI Agent for generating dynamic learner profile vectors from CSV data.
    """
    
    def __init__(self, collection_name="learners", n_clusters=5):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        self.n_clusters = n_clusters
        self.clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
        self.learner_profiles = {}
        self.profile_vectors = {}
        
        print(f"✓ Profiling Agent initialized (clusters: {n_clusters})")
    
    def csv_row_to_profile(self, row):
        """
        Convert CSV row to profile structure matching your data format.
        Maps your columns: code_module, code_presentation, id_student, gender, 
        region, highest_education, imd_band, age_band, num_of_prev_attempts, 
        studied_credits, disability, final_result, avg_score, total_clicks, engagement_level
        """
        # Create unique ID combining student, module, and presentation
        unique_id = f"{row['id_student']}_{row['code_module']}_{row['code_presentation']}"
        
        profile = {
            'id': unique_id,
            'id_student': row['id_student'],
            'code_module': row['code_module'],
            'code_presentation': row['code_presentation'],
            'timestamp': datetime.now().isoformat(),
            
            # Performance metrics
            'avg_score': float(row['avg_score']) if pd.notna(row['avg_score']) else 0.0,
            'final_result': row['final_result'] if pd.notna(row['final_result']) else 'Fail',
            'studied_credits': int(row['studied_credits']) if pd.notna(row['studied_credits']) else 0,
            
            # Engagement patterns
            'total_clicks': int(row['total_clicks']) if pd.notna(row['total_clicks']) else 0,
            'engagement_level': row['engagement_level'] if pd.notna(row['engagement_level']) else 'medium',
            
            # Learning history
            'num_of_prev_attempts': int(row['num_of_prev_attempts']) if pd.notna(row['num_of_prev_attempts']) else 0,
            
            # Demographics
            'gender': row['gender'],
            'region': row['region'],
            'age_band': row['age_band'],
            'highest_education': row['highest_education'],
            'imd_band': row['imd_band'],
            'disability': row['disability'],
            
            'profile_version': 1
        }
        
        return profile
    
    def profile_to_text(self, profile):
        """Convert profile to natural language for semantic embedding"""
        return (
            f"Student {profile['id_student']} in module {profile['code_module']} "
            f"presentation {profile['code_presentation']}. "
            f"A {profile['gender']} learner from {profile['region']}, "
            f"age {profile['age_band']}, education level {profile['highest_education']}, "
            f"IMD band {profile['imd_band']}. "
            f"Has {profile['num_of_prev_attempts']} previous attempts, "
            f"studied {profile['studied_credits']} credits. "
            f"Average score {profile['avg_score']:.1f}, "
            f"{profile['total_clicks']} total clicks, "
            f"engagement level {profile['engagement_level']}. "
            f"Final result: {profile['final_result']}. "
            f"Disability: {profile['disability']}."
        )
    
    def extract_numerical_features(self, profile):
        """Extract numerical features for clustering"""
        # Map engagement level to numeric
        engagement_map = {'low': 0, 'medium': 1, 'high': 2}
        engagement_level = profile['engagement_level']
        if pd.isna(engagement_level) or not isinstance(engagement_level, str):
            engagement_numeric = 1  # Default to medium
        else:
            engagement_numeric = engagement_map.get(engagement_level.lower(), 1)
        
        # Map final result to numeric
        result_map = {'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 3}
        final_result = profile['final_result']
        if pd.isna(final_result):
            result_numeric = 1  # Default to Fail
        else:
            result_numeric = result_map.get(final_result, 1)
        
        features = np.array([
            profile['avg_score'],
            profile['total_clicks'],
            profile['studied_credits'],
            profile['num_of_prev_attempts'],
            engagement_numeric,
            result_numeric,
            1 if profile['disability'] == 'Y' else 0
        ])
        return features
    
    def generate_embedding(self, profile):
        """Generate semantic embedding from profile"""
        text = self.profile_to_text(profile)
        embedding = self.model.encode(text)
        return embedding
    
    def add_learner(self, row_dict):
        """Process a CSV row and generate profile vector"""
        profile = self.csv_row_to_profile(row_dict)
        learner_id = profile['id']
        
        # Store profile
        self.learner_profiles[learner_id] = profile
        
        # Generate embedding
        embedding = self.generate_embedding(profile)
        text = self.profile_to_text(profile)
        
        # Store in ChromaDB
        self.collection.upsert(
            ids=[learner_id],
            documents=[text],
            embeddings=[embedding.tolist()],
            metadatas=[{
                'id_student': str(profile['id_student']),
                'code_module': profile['code_module'],
                'avg_score': float(profile['avg_score']),
                'engagement': profile['engagement_level'],
                'final_result': profile['final_result']
            }]
        )
        
        return profile
    
    def cluster_learners(self):
        """Apply K-means clustering"""
        if len(self.learner_profiles) < self.n_clusters:
            print(f"⚠ Need at least {self.n_clusters} learners for clustering (have {len(self.learner_profiles)})")
            return None
        
        learner_ids = list(self.learner_profiles.keys())
        
        # Extract numerical features
        features = np.array([
            self.extract_numerical_features(self.learner_profiles[lid])
            for lid in learner_ids
        ])
        
        # Normalize and cluster
        features_scaled = self.scaler.fit_transform(features)
        cluster_labels = self.clusterer.fit_predict(features_scaled)
        
        # Assign clusters
        for i, learner_id in enumerate(learner_ids):
            self.learner_profiles[learner_id]['cluster_id'] = int(cluster_labels[i])
        
        # Cluster statistics
        cluster_info = {}
        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_members = [learner_ids[i] for i in range(len(learner_ids)) if mask[i]]
            
            # Calculate cluster characteristics
            cluster_profiles = [self.learner_profiles[m] for m in cluster_members]
            
            cluster_info[cluster_id] = {
                'size': int(np.sum(mask)),
                'avg_score': np.mean([p['avg_score'] for p in cluster_profiles]),
                'avg_clicks': np.mean([p['total_clicks'] for p in cluster_profiles]),
                'centroid': self.clusterer.cluster_centers_[cluster_id].tolist(),
                'member_ids': cluster_members[:5]  # Sample members
            }
        
        return cluster_info
    
    def generate_profile_vector(self, learner_id):
        """Generate comprehensive dynamic profile vector"""
        if learner_id not in self.learner_profiles:
            return None
        
        profile = self.learner_profiles[learner_id]
        
        # Generate components
        embedding = self.generate_embedding(profile)
        features = self.extract_numerical_features(profile)
        cluster_id = profile.get('cluster_id', -1)
        
        # Combine into profile vector
        profile_vector = {
            'learner_id': learner_id,
            'id_student': profile['id_student'],
            'code_module': profile['code_module'],
            'timestamp': profile['timestamp'],
            'version': profile['profile_version'],
            
            # Vector components
            'semantic_embedding': embedding.tolist(),
            'numerical_features': features.tolist(),
            'cluster_id': cluster_id,
            
            # Combined vector
            'combined_vector': np.concatenate([
                embedding,
                features,
                [cluster_id]
            ]).tolist(),
            
            # Dimensions
            'embedding_dim': len(embedding),
            'feature_dim': len(features),
            'total_dim': len(embedding) + len(features) + 1
        }
        
        self.profile_vectors[learner_id] = profile_vector
        return profile_vector
    
    def generate_all_profile_vectors(self):
        """Generate profile vectors for all learners"""
        for learner_id in self.learner_profiles.keys():
            self.generate_profile_vector(learner_id)
        print(f"✓ Generated {len(self.profile_vectors)} profile vectors")
    
    def get_profile_vector(self, learner_id):
        """Retrieve the dynamic profile vector for a learner"""
        learner_id = str(learner_id)
        return self.profile_vectors.get(learner_id, None)
    
    def export_profile_vectors(self, filepath='learner_profile_vectors.json'):
        """Export all profile vectors to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.profile_vectors, f, indent=2)
        print(f"✓ Exported to {filepath}")
    
    def get_cluster_summary(self):
        """Get summary of clusters"""
        clusters = {}
        for learner_id, profile in self.learner_profiles.items():
            cluster_id = profile.get('cluster_id', -1)
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(learner_id)
        return clusters


# ========== MAIN: Process Your CSV ==========

if __name__ == "__main__":
    print("=" * 70)
    print("PROFILING AGENT: Processing learner_profiles.csv")
    print("=" * 70)
    
    # Initialize agent - adjust n_clusters based on your data size
    agent = ProfilingAgent(n_clusters=5)
    
    # Load CSV
    print("\n[1] Loading learner_profiles.csv...")
    df = pd.read_csv("processed/learner_profiles.csv")
    print(f"✓ Loaded {len(df)} learners from CSV")
    print(f"  Columns: {list(df.columns)}")
    
    # Add all learners
    print("\n[2] Processing learners and generating embeddings...")
    for idx, row in df.iterrows():
        agent.add_learner(row.to_dict())
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)} learners...")
    
    print(f"✓ Processed all {len(df)} learners")
    
    # Apply clustering
    print("\n[3] Applying K-means clustering...")
    cluster_info = agent.cluster_learners()
    
    if cluster_info:
        print(f"✓ Identified {len(cluster_info)} clusters:")
        for cluster_id, info in cluster_info.items():
            print(f"\n  Cluster {cluster_id}:")
            print(f"    Size: {info['size']} learners")
            print(f"    Avg Score: {info['avg_score']:.2f}")
            print(f"    Avg Clicks: {info['avg_clicks']:.0f}")
            print(f"    Sample members: {info['member_ids'][:3]}")
    
    # Generate all profile vectors
    print("\n[4] Generating dynamic profile vectors with cluster assignments...")
    agent.generate_all_profile_vectors()
    
    # Show sample profile vector
    print("\n[5] Sample Profile Vector:")
    print("-" * 70)
    sample_id = list(agent.profile_vectors.keys())[0]
    pv = agent.get_profile_vector(sample_id)
    
    print(f"  Learner ID: {pv['learner_id']}")
    print(f"  Student ID: {pv['id_student']}")
    print(f"  Module: {pv['code_module']}")
    print(f"  Semantic Embedding: {pv['embedding_dim']} dimensions")
    print(f"  Numerical Features: {pv['numerical_features']}")
    print(f"  Cluster ID: {pv['cluster_id']}")
    print(f"  Total Vector Dimension: {pv['total_dim']}")
    
    # Export results
    print("\n[6] Exporting results...")
    agent.export_profile_vectors('learner_profile_vectors.json')
    
    # Cluster distribution
    print("\n[7] Cluster Distribution:")
    cluster_summary = agent.get_cluster_summary()
    for cluster_id, members in cluster_summary.items():
        print(f"  Cluster {cluster_id}: {len(members)} learners")
    
    print("\n" + "=" * 70)
    print("✓ COMPLETE: Dynamic learner profile vectors generated!")
    print(f"  Total Learners: {len(agent.profile_vectors)}")
    print(f"  Vector Dimensions: {pv['total_dim']} (embedding + features + cluster)")
    print(f"  Output File: learner_profile_vectors.json")
    print("=" * 70)
