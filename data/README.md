# Dataset Documentation

This directory contains the datasets used for PBAT framework experiments.

## Files

### sample_questions.csv
Sample quiz questions with metadata including:
- Question text
- Answer options
- Correct answer
- Topic/subject
- Difficulty level
- Bloom's taxonomy level

### student_performance.csv
Performance data for 100 students across different assessment models:
- StudentID: Unique identifier
- Ability: Student ability level
- Model: Assessment model used (PBAT, IRT-3P, RL-Adapt, NoAdapt)
- NumQuestions: Number of questions attempted
- AvgScore: Average performance score
- AvgTime: Average time per question
- Feedback: Student feedback score
- CognitiveLoad: Mental effort required
- HallucinationRate: Rate of factual errors
- Retention48h: Knowledge retention after 48 hours
- Perplexity: Question complexity measure

## Usage

Load datasets in Python:
```python
import pandas as pd

# Load sample questions
questions_df = pd.read_csv('data/sample_questions.csv')

# Load student performance data
performance_df = pd.read_csv('data/student_performance.csv')
