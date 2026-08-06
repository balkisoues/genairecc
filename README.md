# Explainable Multi-Agent Generative Recommendation System for Personalized Learning

---

## Project Overview

This project proposes an **Explainable Multi-Agent Generative Recommendation System**
for **personalized e-learning**.  
It combines **Agentic AI, Generative AI (LLMs), and Explainable AI (XAI)** to recommend
*and generate* adaptive learning paths while providing **transparent and trustworthy explanations**.

---

## Motivation & Context

Current e-learning recommendation systems (collaborative filtering, deep learning)
suffer from major limitations:

-  **Lack of adaptability**: no dynamic reasoning or planning
-  **No content generation**: they recommend but do not create learning material
-  **Black-box models**: lack of explainability → low user trust

This project addresses these issues through a **collaborative multi-agent architecture**
capable of reasoning, generating, and explaining personalized learning pathways.

---

##  Scientific Objectives

- Design a **collaborative multi-agent architecture** (memory, planning, communication)
- Integrate **LLMs + RAG** for personalized content generation
- Provide **hybrid explanations** (post-hoc XAI + agentic reasoning)
- Evaluate **recommendation quality** and **user trust**

---

##  Multi-Agent Architecture

| Agent | Role | Technologies |
|------|-----|-------------|
| **Profiling Agent** | Learner profile & learning style analysis | Embeddings, clustering, LLM |
| **Path Planning Agent** | Pedagogical path planning | Graph search, RL, heuristics |
| **Content Generator Agent** | Generates lessons & quizzes | LLM, RAG |
| **Recommendation Agent** | Ranks and recommends resources | Hybrid filtering, LLM |
| **XAI Agent** | Explains decisions | SHAP, LIME, counterfactuals |
| **Orchestrator** | Coordinates agents | LangGraph, AutoGen |

---

##  Technical Pipeline

1. Learner interaction collection  
2. Embedding encoding  
3. Agentic planning  
4. Content generation via **LLM + RAG**  
5. Recommendation & ranking  
6. Explainability (XAI)  
7. Evaluation  

---

##  Explainable AI Methods

| Method | Example |
|------|--------|
| **SHAP / LIME** | Feature importance from learner profile |
| **Counterfactuals** | “If your score increased by +10%, resource X would be recommended” |
| **Chain-of-Thought** | Structured agent reasoning explanations |

---

##  Datasets

- **OULAD**
- **EdNet**
- **Moodle interaction logs**

---

##  Evaluation Metrics

### Recommendation
- NDCG
- MRR
- Recall@K

### Generation
- ROUGE
- BERTScore
- Human evaluation

### Explainability
- Faithfulness
- Plausibility
- User trust score

---

##  Expected Contributions

-  A unified **Agentic AI + GenAI + XAI framework** for e-learning
-  Cognitive explanation methods based on **multi-agent reasoning**
-  Empirical evaluation of **user trust and transparency**

---

##  Project Structure

```text
genai_recommender/
├── agents/            # Individual agent implementations
├── orchestrator/      # Agent coordination logic
├── utils/             # Shared utilities
├── config/            # Configuration files
├── main.py            # Main pipeline
├── demo.py            # Demo / experiments
└── README.md
