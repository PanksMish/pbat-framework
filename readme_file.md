# PBAT: Profile-Based Adaptive Testing Framework

A comprehensive implementation of the Profile-Based Adaptive Testing framework powered by Large Language Models (LLMs) for intelligent quiz generation and adaptive assessment.

## Overview

PBAT addresses key limitations in AI-driven educational assessment systems:
- **Hallucination Mitigation**: Uses Retrieval-Augmented Generation (RAG) to ensure factual accuracy
- **Adaptive Personalization**: Employs Multi-Armed Restless Bandit (MARB) algorithms for learner-specific question selection
- **Psychometric Rigor**: Integrates Item Response Theory (IRT) for statistical calibration

## System Architecture

The framework consists of three main modules:

### 1. Document Processing Module
- PDF parsing and content extraction
- Semantic chunking and embedding generation
- Knowledge base indexing for RAG

### 2. Quiz Generator Module
- LLM-powered question generation with RAG
- Pedagogical validation engine
- IRT parameter calibration
- Bloom's taxonomy alignment

### 3. Profile-Based Adaptive Module (PBAT)
- Real-time learner ability estimation
- Fisher Information-based question selection
- MARB-driven topic coverage optimization
- Adaptive stopping rules

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pbat-framework.git
cd pbat-framework

# Create virtual environment
python -m venv pbat_env
source pbat_env/bin/activate  # On Windows: pbat_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from pbat_main import ExperimentSimulator

# Initialize the simulator
simulator = ExperimentSimulator()

# Run experiment with 100 learners, 3 attempts each
results_df = simulator.run_experiment(n_learners=100, n_attempts=3)

# Save results
results_df.to_csv('experiment_results.csv', index=False)

# View summary
print(results_df.groupby('Model').agg({
    'AvgScore': ['mean', 'std'],
    'CognitiveLoad': ['mean', 'std'],
    'HallucinationRate': ['mean', 'std'],
    'Retention48h': ['mean', 'std'],
    'FeedbackScore': ['mean', 'std']
}).round(2))
```

## Experimental Setup

The framework compares PBAT against three baseline models:

1. **PBAT**: Full adaptive system with RAG and MARB
2. **IRT-3P**: Traditional Item Response Theory model
3. **RL-Adapt**: Reinforcement learning-based adaptation
4. **NoAdapt**: Static quiz generation without personalization

### Evaluation Metrics

**Linguistic Metrics:**
- BLEU-1/BLEU-4 scores
- F1 score
- Perplexity
- Vocabulary diversity

**User-Centric Metrics:**
- Hallucination rate (0-30%)
- Cognitive load (0-100)
- 48-hour retention (0-100)
- Learner feedback score (1-5 Likert scale)

## Dataset Structure

The generated dataset includes the following fields:

| Field | Description | Range |
|-------|-------------|-------|
| StudentID | Unique learner identifier | 101+ |
| Model | Assessment model used | {PBAT, IRT-3P, RL-Adapt, NoAdapt