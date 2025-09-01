import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import normaltest, levene, kruskal
import warnings
from typing import Dict, List, Tuple
warnings.filterwarnings('ignore')

class PBATAnalyzer:
    """Statistical analysis tools for PBAT experiment results"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.metrics = ['AvgScore', 'CognitiveLoad', 'HallucinationRate', 'Retention48h', 'FeedbackScore']
        self.models = self.df['Model'].unique()
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run complete statistical analysis suite"""
        results = {
            'descriptive_stats': self.descriptive_statistics(),
            'normality_tests': self.test_normality(),
            'homogeneity_tests': self.test_homogeneity(),
            'anova_results': self.perform_anova(),
            'pairwise_comparisons': self.pairwise_comparisons(),
            'effect_sizes': self.calculate_effect_sizes(),
            'learning_progression': self.analyze_learning_progression(),
            'correlation_analysis': self.correlation_analysis()
        }
        return results
    
    def descriptive_statistics(self) -> pd.DataFrame:
        """Calculate descriptive statistics for all metrics by model"""
        desc_stats = self.df.groupby('Model')[self.metrics].agg([
            'count', 'mean', 'std', 'min', 'max', 'median',
            lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)
        ]).round(3)
        
        # Rename percentile columns
        desc_stats.columns = [f'{col[0]}_{col[1]}' if col[1] not in ['<lambda_0>', '<lambda_1>'] 
                             else f'{col[0]}_{"Q1" if col[1] == "<lambda_0>" else "Q3"}' 
                             for col in desc_stats.columns]
        
        return desc_stats
    
    def test_normality(self) -> Dict:
        """Test normality assumption for each metric and model"""
        normality_results = {}
        
        for metric in self.metrics:
            normality_results[metric] = {}
            for model in self.models:
                data = self.df[self.df['Model'] == model][metric]
                
                # D'Agostino-Pearson test
                stat, p_value = normaltest(data)
                normality_results[metric][model] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
        
        return normality_results
    
    def test_homogeneity(self) -> Dict:
        """Test homogeneity of variances (Levene's test)"""
        homogeneity_results = {}
        
        for metric in self.metrics:
            groups = [self.df[self.df['Model'] == model][metric] for model in self.models]
            stat, p_value = levene(*groups)
            
            homogeneity_results[metric] = {
                'statistic': stat,
                'p_value': p_value,
                'equal_variances': p_value > 0.05
            }
        
        return homogeneity_results
    
    def perform_anova(self) -> Dict:
        """Perform one-way ANOVA for each metric"""
        anova_results = {}
        
        for metric in self.metrics:
            groups = [self.df[self.df['Model'] == model][metric] for model in self.models]
            
            # Check if ANOVA assumptions are met
            normality = self.test_normality()[metric]
            homogeneity = self.test_homogeneity()[metric]
            
            all_normal = all([normality[model]['is_normal'] for model in self.models])
            equal_vars = homogeneity['equal_variances']
            
            if all_normal and equal_vars:
                # Use parametric ANOVA
                f_stat, p_value = stats.f_oneway(*groups)
                test_type = 'One-way ANOVA'
            else:
                # Use non-parametric Kruskal-Wallis test
                f_stat, p_value = kruskal(*groups)
                test_type = 'Kruskal-Wallis'
            
            anova_results[metric] = {
                'test_type': test_type,
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'assumptions_met': all_normal and equal_vars
            }
        
        return anova_results
    
    def pairwise_comparisons(self) -> Dict:
        """Perform pairwise comparisons between PBAT and other models"""
        pairwise_results = {}
        pbat_data = self.df[self.df['Model'] == 'PBAT']
        
        for metric in self.metrics:
            pairwise_results[metric] = {}
            pbat_metric_data = pbat_data[metric]
            
            for model in self.models:
                if model != 'PBAT':
                    model_data = self.df[self.df['Model'] == model][metric]
                    
                    # Two-sample t-test
                    t_stat, p_value = stats.ttest_ind(pbat_metric_data, model_data)
                    
                    # Mann-Whitney U test (non-parametric alternative)
                    u_stat, u_p_value = stats.mannwhitneyu(pbat_metric_data, model_data, 
                                                          alternative='two-sided')
                    
                    pairwise_results[metric][model] = {
                        't_test': {
                            'statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        },
                        'mann_whitney': {
                            'statistic': u_stat,
                            'p_value': u_p_value,
                            'significant': u_p_value < 0.05
                        }
                    }
        
        return pairwise_results
    
    def calculate_effect_sizes(self) -> Dict:
        """Calculate Cohen's d effect sizes for PBAT vs other models"""
        effect_sizes = {}
        pbat_data = self.df[self.df['Model'] == 'PBAT']
        
        for metric in self.metrics:
            effect_sizes[metric] = {}
            pbat_metric_data = pbat_data[metric]
            
            for model in self.models:
                if model != 'PBAT':
                    model_data = self.df[self.df['Model'] == model][metric]
                    
                    # Cohen's d
                    pooled_std = np.sqrt((pbat_metric_data.var() + model_data.var()) / 2)
                    cohens_d = (pbat_metric_data.mean() - model_data.mean()) / pooled_std
                    
                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        interpretation = 'negligible'
                    elif abs(cohens_d) < 0.5:
                        interpretation = 'small'
                    elif abs(cohens_d) < 0.8:
                        interpretation = 'medium'
                    else:
                        interpretation = 'large'
                    
                    effect_sizes[metric][model] = {
                        'cohens_d': cohens_d,
                        'interpretation': interpretation,
                        'pbat_mean': pbat_metric_data.mean(),
                        'other_mean': model_data.mean(),
                        'difference': pbat_metric_data.mean() - model_data.mean()
                    }
        
        return effect_sizes
    
    def analyze_learning_progression(self) -> Dict:
        """Analyze learning progression across attempts"""
        progression_results = {}
        
        for metric in self.metrics:
            progression_results[metric] = {}
            
            for model in self.models:
                model_data = self.df[self.df['Model'] == model]
                
                # Calculate progression across attempts
                attempt_means = model_data.groupby('AttemptNo')[metric].mean()
                attempt_stds = model_data.groupby('AttemptNo')[metric].std()
                
                # Linear trend test
                attempts = attempt_means.index
                values = attempt_means.values
                slope, intercept, r_value, p_value, std_err = stats.linregress(attempts, values)
                
                # Improvement from attempt 1 to 3
                if 1 in attempt_means.index and 3 in attempt_means.index:
                    improvement = attempt_means[3] - attempt_means[1]
                    improvement_pct = (improvement / attempt_means[1]) * 100
                else:
                    improvement = np.nan
                    improvement_pct = np.nan
                
                progression_results[metric][model] = {
                    'attempt_means': attempt_means.to_dict(),
                    'attempt_stds': attempt_stds.to_dict(),
                    'linear_trend': {
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'significant_trend': p_value < 0.05
                    },
                    'improvement_1_to_3': improvement,
                    'improvement_percent': improvement_pct
                }
        
        return progression_results
    
    def correlation_analysis(self) -> Dict:
        """Analyze correlations between metrics"""
        correlation_results = {}
        
        # Overall correlations
        corr_matrix = self.df[self.metrics].corr()
        
        # Correlations by model
        model_correlations = {}
        for model in self.models:
            model_data = self.df[self.df['Model'] == model]
            model_correlations[model] = model_data[self.metrics].cor