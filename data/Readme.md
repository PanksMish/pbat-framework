## Dataset Integration

The framework supports both synthetic and real datasets. To use your own data:

### Expected Data Structure

#### 1. Question Datasets
Create a `data/` folder with your CSV files:

**MCQ Questions (`data/mcq_questions.csv`)**:
```csv
question,option_a,option_b,option_c,option_d,correct_answer,topic,difficulty,bloom_level
"What is machine learning?","AI technique","Database system","Network protocol","Security tool","A","Machine Learning","Easy","Remember"
```

**Regular Questions (`data/regular_questions.csv`)**:
```csv
question,answer,topic,difficulty,bloom_level
"Explain neural networks","Networks of interconnected nodes that process information","Machine Learning","Medium","Understand"
```

#### 2. Student Performance Dataset (`data/student_performance.csv`)
```csv
StudentID,Ability,Model,NumQuestions,AvgScore,AvgTime,Feedback,CognitiveLoad,HallucinationRate,Retention48h,Perplexity
101,-0.5,PBAT,12,78.5,45.2,4.2,25.3,2.1,82.4,3.2
102,0.2,IRT-3P,15,72.1,52.8,3.8,35.7,8.5,75.2,4.1
```

### Data Upload Instructions

#### Option 1: GitHub Web Interface
1. In your repository, click "Create new file"
2. Type: `data/README.md` and add data documentation
3. Navigate to `data/` folder
4. Click "Add file" → "Upload files"
5. Upload your CSV files:
   - `mcq_questions.csv` (2000 MCQ questions)
   - `regular_questions.csv` (2400 regular questions)  
   - `student_performance.csv` (100 student records)

#### Option 2: Git Command Line
```bash
# Create data directory
mkdir data

# Copy your CSV files to data/
cp /path/to/your/mcq_questions.csv data/
cp /path/to/your/regular_questions.csv data/
cp /path/to/your/student_performance.csv data/

# Add and commit
git add data/
git commit -m "Add real experimental datasets"
git push origin main
```

### Using Real Data

```python
from data_loader import DataLoader, QuestionBankIntegrator
from pbat_main import ExperimentSimulator

# Load real datasets
loader = DataLoader()
validation = loader.validate_datasets()

if validation['integration_ready']:
    # Use real question bank
    integrator = QuestionBankIntegrator(loader)
    integrator.load_and_process_questions()
    
    # Run experiment with real data
    simulator = ExperimentSimulator()
    real_question_bank = integrator.convert_to_pbat_format()
    
    # Replace synthetic generation with real data
    simulator.question_bank = real_question_bank
    results_df = simulator.run_experiment(n_learners=100, n_attempts=3)
    
    # Compare with real performance data
    real_perf = loader.load_student_performance()
    print("Real vs Simulated Performance Comparison Available")
else:
    print("Validation failed - check data format and file paths")
```

### Column Name Flexibility

The data loader handles various column naming conventions:

**Questions Dataset Alternatives:**
- Question text: `question`, `text`, `question_text`
- Options: `option_a/b/c/d`, `option_1/2/3/4`, `optionA/B/C/D`
- Answer: `correct_answer`, `answer`, `correct`
- Topic: `topic`, `subject`, `category`
- Difficulty: `difficulty`, `difficulty_level`, `level`

**Performance Dataset Alternatives:**
- Student ID: `StudentID`, `student_id`, `ID`
- Model: `Model`, `model`, `assessment_type`
- Score: `AvgScore`, `score`, `average_score`# PBAT: Profile-Based Adaptive Testing Framework

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
| Model | Assessment model used | {PBAT, IRT-3P, RL-Adapt, NoAdapt} |
| AttemptNo | Quiz attempt number | {1, 2, 3} |
| AvgScore | Average quiz performance | 0-100 |
| CognitiveLoad | Mental effort required | 0-100 (lower is better) |
| HallucinationRate | Factual error rate | 0-30% (lower is better) |
| Retention48h | Knowledge retention after 48h | 0-100 |
| FeedbackScore | Learner satisfaction | 1-5 Likert scale |

## File Structure

```
pbat-framework/
├── pbat_main.py              # Main implementation
├── visualization.py          # Results visualization
├── analysis.py              # Statistical analysis tools
├── requirements.txt         # Dependencies
├── README.md               # This file
├── examples/               # Usage examples
│   └── demo_notebook.ipynb
└── results/               # Generated results
    └── pbat_experiment_results.csv
```

## Key Features

### Adaptive Question Selection
- Fisher Information maximization for optimal item selection
- Real-time ability estimation using MAP/EAP methods
- Dynamic stopping rules based on measurement precision

### RAG-Enhanced Generation
- Document parsing and semantic chunking
- TF-IDF based similarity matching
- Context-aware question generation

### Multi-Armed Restless Bandit (MARB)
- Topic-difficulty arms with evolving belief states
- Reward optimization for learning gain
- Balanced exploration-exploitation strategy

## Performance Results

Based on experimental validation, PBAT demonstrates:

- **15.6% improvement** in 48-hour retention over static methods
- **2.3% hallucination rate** (vs 20%+ for non-RAG approaches)
- **Lowest cognitive load** with maintained engagement
- **Highest learner satisfaction** scores (4.2/5.0 average)

## Usage Examples

### Basic Quiz Generation

```python
from pbat_main import DocumentProcessor, QuizGenerator

# Setup document processing
processor = DocumentProcessor()
documents = ["Your course content here..."]
chunks = processor.parse_documents(documents)
processor.extract_embeddings(chunks)

# Generate quiz questions
generator = QuizGenerator(processor)
questions = generator.generate_questions("Machine Learning", "Medium", count=10)
validated_questions = generator.validate_questions(questions)
```

### Adaptive Assessment Session

```python
from pbat_main import PBATAdapter

# Initialize adaptive system
adapter = PBATAdapter(generator)
learner_id = "student_001"
adapter.initialize_learner_profile(learner_id, initial_ability=0.0)

# Simulate adaptive session
while not adapter.check_stopping_rule(learner_id):
    next_question = adapter.select_next_question(learner_id, available_questions)
    # Present question to learner and get response
    response = get_learner_response(next_question)  # Your implementation
    adapter.update_ability_map(learner_id, response, next_question)
```

## Experimental Validation

Run the full experimental suite:

```python
python pbat_main.py
```

This will generate:
- Comparative results across all models
- Statistical summaries and significance tests
- CSV output for further analysis

## Visualization

Generate performance comparison plots:

```python
python visualization.py
```

Creates:
- Model performance comparisons
- Learning trajectory analysis
- Statistical distribution plots

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{pbat2024,
  title={PBAT: A Profile-Based Adaptive Testing Framework Powered by LLMs},
  author={Mishra, Pankaj P. and Venkataramanan, V and Patel, Keyur and others},
  journal={Educational Technology Research},
  year={2024}
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support:
- Email: pankaj.mishra@somaiya.edu
- Project Issues: [GitHub Issues](https://github.com/yourusername/pbat-framework/issues)

## Acknowledgments

- K J Somaiya School of Engineering, Mumbai
- Nagoya Institute of Technology, Japan
- Educational Data Mining Community
