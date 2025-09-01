import pandas as pd
import numpy as np
from typing import Tuple, Optional

class DataLoader:
    """Load and preprocess PBAT datasets"""
    
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = data_dir
    
    def load_sample_questions(self) -> pd.DataFrame:
        """Load sample questions dataset"""
        filepath = f"{self.data_dir}/sample_questions.csv"
        return pd.read_csv(filepath)
    
    def load_student_performance(self) -> pd.DataFrame:
        """Load student performance dataset"""
        filepath = f"{self.data_dir}/student_performance.csv"
        return pd.read_csv(filepath)
    
    def validate_datasets(self) -> bool:
        """Validate dataset formats and completeness"""
        try:
            questions_df = self.load_sample_questions()
            performance_df = self.load_student_performance()
            
            # Validate required columns
            required_perf_cols = ['StudentID', 'Ability', 'Model', 'NumQuestions', 
                                'AvgScore', 'AvgTime', 'Feedback', 'CognitiveLoad',
                                'HallucinationRate', 'Retention48h', 'Perplexity']
            
            missing_cols = set(required_perf_cols) - set(performance_df.columns)
            if missing_cols:
                print(f"Missing columns in performance data: {missing_cols}")
                return False
            
            print("Dataset validation passed!")
            return True
            
        except Exception as e:
            print(f"Dataset validation failed: {e}")
            return False
