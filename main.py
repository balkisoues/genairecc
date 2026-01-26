"""
main.py
"""

import pandas as pd
import json
from datetime import datetime
from typing import Dict

from shared.state import create_initial_state
from orchestrator.orchestrator import Orchestrator
from config.settings import Config


def load_data(csv_path: str) -> pd.DataFrame:
    """load data from csv"""
    print(f"LOOOOOADING data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"LOADED {len(df)} learner records")
    return df


def display_results(state: Dict):
    """DISPLAY RESULTS"""

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n Agent Execution Log:")
    for log in state.get("agent_logs", []):
        print(f"  {log}")

    # learner profile
    print(f"\n LEARNER PROFILE :")
    profile = state.get("learner_profile", {})
    if profile:
        print(f"  ID: {profile.get('id_student', 'N/A')}")
        print(f"  Score: {profile.get('avg_score', 0)}")
        print(f"  Engagement: {profile.get('engagement_level', 'N/A')}")
        print(f"  Learning Style: {profile.get('learning_style', 'N/A')}")
        print(f"  Cluster: {profile.get('cluster_id', 'N/A')}")
    else:
        print(" NO PROFILE generated")

    # learning path
    learning_path = state.get("learning_path", [])
    if learning_path:
        print(f"\nLearning Path ({len(learning_path)} units):")
        for i, unit in enumerate(learning_path, 1):
            print(
                f"  {i}. {unit.get('concept', 'N/A')} "
                f"({unit.get('difficulty', 'N/A')}, "
                f"{unit.get('estimated_duration', 'N/A')}min)"
            )
    else:
        print(f"\n learning Path: Not generated yet")

    # generated content
    generated_content = state.get("generated_content", [])
    if generated_content:
        print(f"\n generated Content ({len(generated_content)} items):")
        for content in generated_content:
            print(f"  • {content.get('concept', 'N/A')}")
            explanation = content.get("explanation", "")
            if explanation:
                print(f"    Explanation: {explanation[:80]}...")
            print(f"    Quiz: {len(content.get('quiz', []))} questions")
    else:
        print(f"\n Generated Content: Not generated yet")

    # recommendations
    print(f"\n RECOMMENFDATIONS:")
    recs = state.get("recommendations", {})
    if recs:
        if recs.get("primary"):
            print(f"  Primary: {recs['primary'].get('concept', 'N/A')}")
        next_steps = recs.get("next_steps", [])
        if next_steps:
            print(f"  Next Steps:")
            for step in next_steps:
                print(f"    - {step}")
    else:
        print("  No recommendations generated yet")

    # explanations (learner view)
    explanations = state.get("explanations", {})
    if explanations and explanations.get("learner_view"):
        print(f"\n Explanation (Learner View):")
        print(explanations.get("learner_view", "No explanation available"))
    else:
        print(f"\n Explanation: Not generated yet")

    # errors
    errors = state.get("errors", [])
    if errors:
        print(f"\n ATTENTION TO THESE ERRORS:")
        for error in errors:
            print(f"  • {error}")

    print("\n" + "=" * 70)


def export_results(state: Dict, output_path: str):
    """export results in a json file"""

    explanations = state.get("explanations") or {}

    output = {
        "timestamp": datetime.now().isoformat(),
        "learner_id": state.get("learner_id"),
        "profile": state.get("learner_profile", {}),
        "learning_path": state.get("learning_path", []),
        "generated_content": state.get("generated_content", []),
        "recommendations": state.get("recommendations", {}),
        "explanations": {
            "learner_view": explanations.get("learner_view", ""),
            "instructor_view": explanations.get("instructor_view", ""),
            "feature_importance": explanations.get("feature_importance", {}),
            "counterfactuals": explanations.get("counterfactuals", []),
        },
        "agent_logs": state.get("agent_logs", []),
        "errors": state.get("errors", []),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n RESULTS EXPORTED AVEC SUCCEESSS TO {output_path}")


def main():
    """MAIN FCT"""

    print("=" * 70)
    print("EXPLAINABLE MULTI-AGENT RECOMMENDATION SYSTEM")
    print("FOR PERSONALIZED LEARNING")
    print("=" * 70)

    df = load_data(Config.system.data_path)
    #picking one learner to test the system with it 
    sample_learner = df.iloc[0].to_dict()
    learner_id = f"{sample_learner['id_student']}_{sample_learner['code_module']}"
	
    print(f"\n PROCESSING LEARNER NUMBER: {learner_id}")
    #creating a workfwlow with clean state	
    initial_state = create_initial_state(learner_id, sample_learner)
    
    #initliaze the orchestrator and run it 
    
    orchestrator = Orchestrator()
    final_state = orchestrator.run(initial_state)

    display_results(final_state)

    output_file = (
        f"{Config.system.output_path}"
        f"results_{learner_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    export_results(final_state, output_file)

    # DEBUG BLOCK (now in correct scope)
    print("\n DEBUG - Checking agent execution:")
    print(f"  Profiling complete: {final_state.get('profiling_complete', False)}")
    print(f"  Path planning complete: {final_state.get('path_planning_complete', False)}")
    print(f"  Content generation complete: {final_state.get('content_generation_complete', False)}")
    print(f"  Recommendation complete: {final_state.get('recommendation_complete', False)}")
    print(f"  XAI complete: {final_state.get('xai_complete', False)}")

    if final_state.get("errors"):
        print(f"  Errors: {final_state['errors']}")

    print("\n Process complete!")
    print("=" * 70)


def batch_process(num_learners: int = 10):
    """PRCOSSEING 10 LEARNERS AT A TIME"""

    print(f"\n PROCESSING IN BATCH {num_learners} LEARNERS...")

    df = load_data(Config.system.data_path)
    orchestrator = Orchestrator()

    results = []

    for i, (_, row) in enumerate(df.head(num_learners).iterrows()):
        learner_data = row.to_dict()
        learner_id = f"{learner_data['id_student']}_{learner_data['code_module']}"

        print(f"\n[{i + 1}/{num_learners}] Processing {learner_id}...")

        state = create_initial_state(learner_id, learner_data)
        final_state = orchestrator.run(state)
        results.append(final_state)

    print(f"\n BATCH PROCESSING FINI {len(results)} learners processed")
    return results


if __name__ == "__main__":
    main()
    #batch_process(num_learners=10)

