import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuestionData:
    """Data class for question information"""
    id: str
    text: str
    question_type: str  # 'mcq' or 'regular'
    options: List[str]  # Empty list for regular questions
    correct_answer: Union[int, str]  # Index for MCQ, text for regular
    topic: str
    difficulty: str
    bloom_level: str
    metadata: Dict = None

class DataLoader:
    """Load and preprocess PBAT datasets"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
        self.questions_cache = {}
        self.performance_cache = None
        
    def load_sample_questions(self, question_type: str = "all") -> pd.DataFrame:
        """
        Load sample questions dataset
        
        Args:
            question_type: 'mcq', 'regular', or 'all'
        """
        try:
            if question_type == "mcq" or question_type == "all":
                mcq_path = os.path.join(self.data_dir, "mcq_questions.csv")
                if os.path.exists(mcq_path):
                    mcq_df = pd.read_csv(mcq_path)
                    mcq_df['question_type'] = 'mcq'
                    logger.info(f"Loaded {len(mcq_df)} MCQ questions")
                else:
                    mcq_df = pd.DataFrame()
                    logger.warning("MCQ questions file not found")
            
            if question_type == "regular" or question_type == "all":
                regular_path = os.path.join(self.data_dir, "regular_questions.csv")
                if os.path.exists(regular_path):
                    regular_df = pd.read_csv(regular_path)
                    regular_df['question_type'] = 'regular'
                    logger.info(f"Loaded {len(regular_df)} regular questions")
                else:
                    regular_df = pd.DataFrame()
                    logger.warning("Regular questions file not found")
            
            # Combine datasets
            if question_type == "all":
                if not mcq_df.empty and not regular_df.empty:
                    questions_df = pd.concat([mcq_df, regular_df], ignore_index=True)
                elif not mcq_df.empty:
                    questions_df = mcq_df
                elif not regular_df.empty:
                    questions_df = regular_df
                else:
                    raise FileNotFoundError("No question files found")
            elif question_type == "mcq":
                questions_df = mcq_df
            else:  # regular
                questions_df = regular_df
            
            # Cache the results
            self.questions_cache[question_type] = questions_df
            return questions_df
            
        except Exception as e:
            logger.error(f"Error loading questions: {e}")
            return pd.DataFrame()
    
    def load_student_performance(self) -> pd.DataFrame:
        """Load student performance dataset"""
        try:
            filepath = os.path.join(self.data_dir, "student_performance.csv")
            performance_df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['StudentID', 'Ability', 'Model', 'NumQuestions', 
                           'AvgScore', 'AvgTime', 'Feedback', 'CognitiveLoad',
                           'HallucinationRate', 'Retention48h', 'Perplexity']
            
            missing_cols = set(required_cols) - set(performance_df.columns)
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
            
            self.performance_cache = performance_df
            logger.info(f"Loaded performance data for {len(performance_df)} records")
            return performance_df
            
        except FileNotFoundError:
            logger.error("Student performance file not found")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading student performance: {e}")
            return pd.DataFrame()
    
    def preprocess_questions(self, questions_df: pd.DataFrame) -> List[QuestionData]:
        """Convert DataFrame to QuestionData objects"""
        questions = []
        
        for _, row in questions_df.iterrows():
            # Handle different column name variations
            question_id = str(row.get('id', row.get('question_id', f"q_{len(questions)}")))
            text = row.get('text', row.get('question_text', row.get('question', '')))
            q_type = row.get('question_type', 'regular')
            
            # Handle options for MCQ
            if q_type == 'mcq':
                options = []
                correct_idx = 0
                
                # Try different column naming patterns
                for opt_col in ['option_a', 'option_b', 'option_c', 'option_d']:
                    if opt_col in row and pd.notna(row[opt_col]):
                        options.append(row[opt_col])
                
                # Alternative naming: options_1, options_2, etc.
                if not options:
                    for i in range(1, 5):
                        opt_col = f'option_{i}'
                        if opt_col in row and pd.notna(row[opt_col]):
                            options.append(row[opt_col])
                
                # Get correct answer
                correct_answer = row.get('correct_answer', row.get('answer', 0))
                if isinstance(correct_answer, str):
                    # If answer is 'A', 'B', etc., convert to index
                    if correct_answer.upper() in ['A', 'B', 'C', 'D']:
                        correct_idx = ord(correct_answer.upper()) - ord('A')
                    else:
                        correct_idx = 0
                else:
                    correct_idx = int(correct_answer) if pd.notna(correct_answer) else 0
            else:
                options = []
                correct_idx = row.get('correct_answer', row.get('answer', ''))
            
            # Extract metadata
            topic = row.get('topic', row.get('subject', 'General'))
            difficulty = row.get('difficulty', row.get('difficulty_level', 'Medium'))
            bloom_level = row.get('bloom_level', row.get('taxonomy', 'Understand'))
            
            question_obj = QuestionData(
                id=question_id,
                text=text,
                question_type=q_type,
                options=options,
                correct_answer=correct_idx,
                topic=topic,
                difficulty=difficulty,
                bloom_level=bloom_level,
                metadata=dict(row)  # Store original row data
            )
            
            questions.append(question_obj)
        
        logger.info(f"Preprocessed {len(questions)} questions")
        return questions
    
    def get_question_bank_structure(self) -> Dict:
        """Analyze question bank structure"""
        questions_df = self.load_sample_questions()
        
        if questions_df.empty:
            return {}
        
        structure = {
            'total_questions': len(questions_df),
            'by_type': questions_df.get('question_type', pd.Series(['regular'] * len(questions_df))).value_counts().to_dict(),
            'by_topic': questions_df.get('topic', questions_df.get('subject', pd.Series(['General'] * len(questions_df)))).value_counts().to_dict(),
            'by_difficulty': questions_df.get('difficulty', questions_df.get('difficulty_level', pd.Series(['Medium'] * len(questions_df)))).value_counts().to_dict(),
            'by_bloom_level': questions_df.get('bloom_level', questions_df.get('taxonomy', pd.Series(['Understand'] * len(questions_df)))).value_counts().to_dict()
        }
        
        return structure
    
    def validate_datasets(self) -> Dict[str, bool]:
        """Validate both datasets and return status"""
        validation_results = {
            'questions_valid': False,
            'performance_valid': False,
            'integration_ready': False
        }
        
        try:
            # Validate questions dataset
            questions_df = self.load_sample_questions()
            if not questions_df.empty:
                required_q_cols = ['text', 'topic', 'difficulty']
                missing_q_cols = set(required_q_cols) - set(questions_df.columns)
                
                # Check for alternative column names
                alt_mappings = {
                    'text': ['question_text', 'question'],
                    'topic': ['subject', 'category'],
                    'difficulty': ['difficulty_level', 'level']
                }
                
                for req_col in list(missing_q_cols):
                    for alt_col in alt_mappings.get(req_col, []):
                        if alt_col in questions_df.columns:
                            missing_q_cols.discard(req_col)
                            break
                
                if not missing_q_cols:
                    validation_results['questions_valid'] = True
                    logger.info("Questions dataset validation passed")
                else:
                    logger.warning(f"Questions dataset missing columns: {missing_q_cols}")
            
            # Validate performance dataset
            performance_df = self.load_student_performance()
            if not performance_df.empty:
                required_p_cols = ['StudentID', 'Model', 'AvgScore']
                missing_p_cols = set(required_p_cols) - set(performance_df.columns)
                
                if not missing_p_cols:
                    validation_results['performance_valid'] = True
                    logger.info("Performance dataset validation passed")
                else:
                    logger.warning(f"Performance dataset missing columns: {missing_p_cols}")
            
            # Check integration readiness
            validation_results['integration_ready'] = (
                validation_results['questions_valid'] and 
                validation_results['performance_valid']
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
        
        return validation_results
    
    def generate_data_summary(self) -> str:
        """Generate comprehensive data summary report"""
        report = []
        report.append("="*60)
        report.append("PBAT FRAMEWORK - DATASET SUMMARY REPORT")
        report.append("="*60)
        report.append("")
        
        # Questions summary
        questions_df = self.load_sample_questions()
        if not questions_df.empty:
            report.append("QUESTIONS DATASET:")
            report.append("-" * 20)
            
            structure = self.get_question_bank_structure()
            report.append(f"Total Questions: {structure['total_questions']}")
            
            if 'by_type' in structure:
                report.append(f"By Type: {structure['by_type']}")
            if 'by_topic' in structure:
                report.append(f"By Topic: {dict(list(structure['by_topic'].items())[:5])}...")
            if 'by_difficulty' in structure:
                report.append(f"By Difficulty: {structure['by_difficulty']}")
            
            report.append("")
        
        # Performance summary
        performance_df = self.load_student_performance()
        if not performance_df.empty:
            report.append("PERFORMANCE DATASET:")
            report.append("-" * 20)
            report.append(f"Total Records: {len(performance_df)}")
            report.append(f"Unique Students: {performance_df['StudentID'].nunique()}")
            report.append(f"Models Tested: {list(performance_df['Model'].unique())}")
            
            # Performance statistics
            numeric_cols = ['AvgScore', 'CognitiveLoad', 'HallucinationRate', 'Retention48h']
            existing_cols = [col for col in numeric_cols if col in performance_df.columns]
            
            if existing_cols:
                report.append("\nPerformance Summary:")
                for col in existing_cols:
                    mean_val = performance_df[col].mean()
                    std_val = performance_df[col].std()
                    report.append(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)

class QuestionBankIntegrator:
    """Integrate real question datasets with PBAT framework"""
    
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.mcq_questions = []
        self.regular_questions = []
        
    def load_and_process_questions(self):
        """Load and process both question types"""
        try:
            # Load MCQ questions
            mcq_df = pd.read_csv(os.path.join(self.data_loader.data_dir, "mcq_questions.csv"))
            self.mcq_questions = self._process_mcq_questions(mcq_df)
            logger.info(f"Processed {len(self.mcq_questions)} MCQ questions")
            
            # Load regular questions  
            regular_df = pd.read_csv(os.path.join(self.data_loader.data_dir, "regular_questions.csv"))
            self.regular_questions = self._process_regular_questions(regular_df)
            logger.info(f"Processed {len(self.regular_questions)} regular questions")
            
        except FileNotFoundError as e:
            logger.error(f"Question file not found: {e}")
        except Exception as e:
            logger.error(f"Error processing questions: {e}")
    
    def _process_mcq_questions(self, df: pd.DataFrame) -> List[QuestionData]:
        """Process MCQ questions dataset"""
        questions = []
        
        for idx, row in df.iterrows():
            # Handle different possible column names
            question_text = self._get_column_value(row, ['question', 'text', 'question_text'])
            
            # Extract options (try different naming patterns)
            options = []
            option_patterns = [
                ['option_a', 'option_b', 'option_c', 'option_d'],
                ['option_1', 'option_2', 'option_3', 'option_4'],
                ['optionA', 'optionB', 'optionC', 'optionD'],
                ['a', 'b', 'c', 'd']
            ]
            
            for pattern in option_patterns:
                temp_options = []
                for opt_col in pattern:
                    if opt_col in row and pd.notna(row[opt_col]):
                        temp_options.append(str(row[opt_col]))
                if len(temp_options) >= 2:  # At least 2 options
                    options = temp_options
                    break
            
            # Get correct answer
            correct_answer = self._get_column_value(row, ['correct_answer', 'answer', 'correct'])
            if isinstance(correct_answer, str) and correct_answer.upper() in ['A', 'B', 'C', 'D']:
                correct_idx = ord(correct_answer.upper()) - ord('A')
            else:
                correct_idx = int(correct_answer) if pd.notna(correct_answer) else 0
            
            # Extract metadata
            topic = self._get_column_value(row, ['topic', 'subject', 'category'], 'General')
            difficulty = self._get_column_value(row, ['difficulty', 'difficulty_level', 'level'], 'Medium')
            bloom_level = self._get_column_value(row, ['bloom_level', 'taxonomy', 'cognitive_level'], 'Understand')
            
            question = QuestionData(
                id=f"mcq_{idx}",
                text=question_text,
                question_type='mcq',
                options=options,
                correct_answer=correct_idx,
                topic=topic,
                difficulty=difficulty,
                bloom_level=bloom_level,
                metadata=dict(row)
            )
            questions.append(question)
        
        return questions
    
    def _process_regular_questions(self, df: pd.DataFrame) -> List[QuestionData]:
        """Process regular questions dataset"""
        questions = []
        
        for idx, row in df.iterrows():
            question_text = self._get_column_value(row, ['question', 'text', 'question_text'])
            correct_answer = self._get_column_value(row, ['answer', 'correct_answer', 'solution'])
            
            topic = self._get_column_value(row, ['topic', 'subject', 'category'], 'General')
            difficulty = self._get_column_value(row, ['difficulty', 'difficulty_level', 'level'], 'Medium')
            bloom_level = self._get_column_value(row, ['bloom_level', 'taxonomy', 'cognitive_level'], 'Understand')
            
            question = QuestionData(
                id=f"regular_{idx}",
                text=question_text,
                question_type='regular',
                options=[],
                correct_answer=correct_answer,
                topic=topic,
                difficulty=difficulty,
                bloom_level=bloom_level,
                metadata=dict(row)
            )
            questions.append(question)
        
        return questions
    
    def _get_column_value(self, row: pd.Series, possible_cols: List[str], default=''):
        """Get value from row trying different possible column names"""
        for col in possible_cols:
            if col in row and pd.notna(row[col]):
                return row[col]
        return default
    
    def get_questions_by_criteria(self, topic: str = None, difficulty: str = None, 
                                question_type: str = None, count: int = 10) -> List[QuestionData]:
        """Filter questions by criteria"""
        all_questions = self.mcq_questions + self.regular_questions
        
        filtered_questions = all_questions
        
        if topic:
            filtered_questions = [q for q in filtered_questions if q.topic.lower() == topic.lower()]
        
        if difficulty:
            filtered_questions = [q for q in filtered_questions if q.difficulty.lower() == difficulty.lower()]
        
        if question_type:
            filtered_questions = [q for q in filtered_questions if q.question_type == question_type]
        
        # Return random sample
        if len(filtered_questions) > count:
            return random.sample(filtered_questions, count)
        return filtered_questions
    
    def convert_to_pbat_format(self) -> Dict:
        """Convert questions to PBAT framework format"""
        from pbat_main import Question
        
        pbat_questions = {}
        all_questions = self.mcq_questions + self.regular_questions
        
        # Group by topic and difficulty
        for question_data in all_questions:
            topic = question_data.topic
            difficulty = question_data.difficulty
            
            if topic not in pbat_questions:
                pbat_questions[topic] = {}
            if difficulty not in pbat_questions[topic]:
                pbat_questions[topic][difficulty] = []
            
            # Convert to PBAT Question format
            pbat_question = Question(
                id=question_data.id,
                text=question_data.text,
                options=question_data.options if question_data.question_type == 'mcq' else 
                       [str(question_data.correct_answer), "Alternative 1", "Alternative 2", "Alternative 3"],
                correct_answer=question_data.correct_answer if isinstance(question_data.correct_answer, int) else 0,
                difficulty=self._map_difficulty_to_numeric(question_data.difficulty),
                discrimination=np.random.normal(1.0, 0.2),  # Generate IRT parameters
                guessing=np.random.uniform(0.1, 0.3),
                topic=question_data.topic,
                bloom_level=question_data.bloom_level
            )
            
            pbat_questions[topic][difficulty].append(pbat_question)
        
        return pbat_questions
    
    def _map_difficulty_to_numeric(self, difficulty: str) -> float:
        """Map difficulty strings to numeric values"""
        mapping = {
            'easy': -1.0,
            'medium': 0.0,
            'hard': 1.0,
            'low': -1.0,
            'moderate': 0.0,
            'high': 1.0
        }
        return mapping.get(difficulty.lower(), 0.0) + np.random.normal(0, 0.2)

def create_data_integration_example():
    """Create example integration code"""
    example_code = '''
# Example: Integrating Real Datasets with PBAT Framework

from data_loader import DataLoader, QuestionBankIntegrator
from pbat_main import ExperimentSimulator

# 1. Load your real datasets
data_loader = DataLoader(data_dir="data/")

# Validate datasets
validation_status = data_loader.validate_datasets()
print("Dataset Validation:", validation_status)

# 2. Load and process questions
integrator = QuestionBankIntegrator(data_loader)
integrator.load_and_process_questions()

# 3. Convert to PBAT format
pbat_question_bank = integrator.convert_to_pbat_format()

# 4. Load real performance data for comparison
real_performance = data_loader.load_student_performance()

# 5. Run experiment with real question bank
simulator = ExperimentSimulator()
simulator.quiz_generator.question_bank = pbat_question_bank  # Use real questions

# Run experiment
results_df = simulator.run_experiment(n_learners=50, n_attempts=3)

# 6. Compare with real performance data
print("Simulated vs Real Performance Comparison:")
print("Simulated PBAT Average Score:", results_df[results_df['Model']=='PBAT']['AvgScore'].mean())
if 'AvgScore' in real_performance.columns:
    print("Real Average Score:", real_performance['AvgScore'].mean())
'''
    
    return example_code

def main():
    """Demo the data loading functionality"""
    # Initialize data loader
    loader = DataLoader()
    
    # Generate summary
    summary = loader.generate_data_summary()
    print(summary)
    
    # Validate datasets
    validation = loader.validate_datasets()
    print(f"\nValidation Results: {validation}")
    
    # Show integration example
    if validation['integration_ready']:
        print("\nDatasets are ready for integration!")
        integrator = QuestionBankIntegrator(loader)
        integrator.load_and_process_questions()
        
        # Show question bank structure
        structure = loader.get_question_bank_structure()
        print(f"\nQuestion Bank Structure: {structure}")
    else:
        print("\nDatasets need attention before integration.")
        print("Please check file paths and column names.")

if __name__ == "__main__":
    main()