#!/usr/bin/env python3
"""
LLM Calibration Analysis Script

This script analyzes parsed CSV outputs from LLM calibration experiments.
It computes comprehensive metrics and generates visualization plots for each model/dataset combination.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
import json
from scipy import stats
from sklearn.metrics import mean_absolute_error

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Suppress warnings
warnings.filterwarnings('ignore')

def extract_model_and_dataset(filename: str) -> Tuple[str, str]:
    """Extract model name and dataset name from filename."""
    # Handle deepseek files
    if 'deepseek' in filename.lower():
        if 'deepseek-reasoner' in filename:
            model = 'deepseek-reasoner'
        elif 'deepseek-chat' in filename:
            model = 'deepseek-chat'
        else:
            model = 'deepseek'
    elif 'gpt-4o' in filename:
        model = 'gpt-4o'
    elif 'claude' in filename.lower():
        model = 'claude'
    elif 'gemini' in filename.lower():
        model = 'gemini'
    else:
        model = 'unknown'
    
    # Extract dataset name
    if 'lsat_ar_test' in filename:
        dataset = 'LSAT_AR_Test'
    elif 'sat_en' in filename:
        dataset = 'SAT_EN'
    elif 'sciq_test' in filename:
        dataset = 'SciQ_Test'
    elif 'boolq_valid' in filename:
        dataset = 'BoolQ_Valid'
    elif 'halu_eval_qa' in filename:
        dataset = 'HaluEval_QA'
    elif 'life_eval' in filename:
        dataset = 'Life_Eval'
    else:
        dataset = 'unknown'
    
    return model, dataset

def compute_metrics(df: pd.DataFrame) -> Dict:
    """Compute all metrics for a given dataframe."""
    metrics = {}
    
    # Basic counts
    metrics['questions_asked'] = len(df)
    
    # Check if correct_format column exists
    if 'correct_format' in df.columns:
        metrics['responses_formatted_correctly'] = len(df[df['correct_format'] == True])
        metrics['responses_formatted_correctly_pct'] = (metrics['responses_formatted_correctly'] / metrics['questions_asked']) * 100
    else:
        metrics['responses_formatted_correctly'] = len(df)  # Assume all are formatted correctly
        metrics['responses_formatted_correctly_pct'] = 100.0
    
    # Extract stated confidence and token probabilities
    stated_confidences = []
    token_probs = []
    correct_answers = 0
    
    # Check if required columns exist
    has_answer_col = 'answer' in df.columns
    has_answer_col_cap = 'Answer' in df.columns
    # Check if at least some confidence columns exist (A, B, C, D are required, E is optional)
    has_confidence_cols = all(col in df.columns for col in ['A', 'B', 'C', 'D'])
    has_token_prob_col = 't1_prob' in df.columns
    
    for _, row in df.iterrows():
        # Check if answer is correct
        if has_answer_col and has_answer_col_cap:
            try:
                if str(row['answer']).strip() == str(row['Answer']).strip():
                    correct_answers += 1
            except:
                pass
        
        # Extract stated confidence from A, B, C, D columns (E is optional)
        if has_confidence_cols:
            try:
                answer_letter = str(row.get('Answer', '')).strip()
                if answer_letter in ['A', 'B', 'C', 'D']:
                    confidence_val = row[answer_letter]
                    if pd.notna(confidence_val):
                        confidence = float(confidence_val)
                        stated_confidences.append(confidence)
            except (ValueError, KeyError, TypeError):
                pass
        
        # Extract token probability from t1_prob column
        if has_token_prob_col:
            try:
                token_prob_val = row['t1_prob']
                if pd.notna(token_prob_val):
                    token_prob = float(token_prob_val)
                    if token_prob > 0:  # Filter out zero probabilities
                        token_probs.append(token_prob)
            except (ValueError, KeyError, TypeError):
                pass
    
    metrics['correct_answers'] = correct_answers
    metrics['accuracy'] = (correct_answers / metrics['questions_asked']) * 100
    
    # Stated confidence metrics
    if stated_confidences:
        metrics['min_confidence'] = min(stated_confidences)
        metrics['max_confidence'] = max(stated_confidences)
        metrics['avg_confidence'] = np.mean(stated_confidences)
        metrics['std_confidence'] = np.std(stated_confidences)
    else:
        metrics['min_confidence'] = np.nan
        metrics['max_confidence'] = np.nan
        metrics['avg_confidence'] = np.nan
        metrics['std_confidence'] = np.nan
    
    # Token probability metrics
    if token_probs:
        metrics['avg_token_prob'] = np.mean(token_probs)
        metrics['std_token_prob'] = np.std(token_probs)
    else:
        metrics['avg_token_prob'] = np.nan
        metrics['std_token_prob'] = np.nan
    
    # Expected Calibration Error (ECE)
    if stated_confidences and has_answer_col and has_answer_col_cap:
        ece = compute_ece(stated_confidences, df, correct_answers)
        metrics['ece'] = ece
    else:
        metrics['ece'] = np.nan
    
    # Overconfidence
    if stated_confidences and has_confidence_cols:
        max_confidences = []
        for _, row in df.iterrows():
            try:
                confidences = [float(row[col]) for col in ['A', 'B', 'C', 'D'] 
                             if col in row and pd.notna(row[col])]
                if confidences:
                    max_confidences.append(max(confidences))
            except (ValueError, KeyError, TypeError):
                pass
        
        if max_confidences:
            avg_max_confidence = np.mean(max_confidences)
            metrics['overconfidence'] = avg_max_confidence - metrics['accuracy']
        else:
            metrics['overconfidence'] = np.nan
    else:
        metrics['overconfidence'] = np.nan
    
    # Correlations
    if stated_confidences and has_answer_col and has_answer_col_cap and len(stated_confidences) == len(df):
        correctness = []
        for _, row in df.iterrows():
            try:
                if str(row['answer']).strip() == str(row['Answer']).strip():
                    correctness.append(1)
                else:
                    correctness.append(0)
            except:
                correctness.append(0)
        
        if len(correctness) == len(stated_confidences):
            try:
                corr = np.corrcoef(stated_confidences, correctness)[0, 1]
                metrics['confidence_correctness_corr'] = corr if not np.isnan(corr) else np.nan
            except:
                metrics['confidence_correctness_corr'] = np.nan
        else:
            metrics['confidence_correctness_corr'] = np.nan
    else:
        metrics['confidence_correctness_corr'] = np.nan
    
    if token_probs and has_answer_col and has_answer_col_cap and len(token_probs) == len(df):
        correctness = []
        for _, row in df.iterrows():
            try:
                if str(row['answer']).strip() == str(row['Answer']).strip():
                    correctness.append(1)
                else:
                    correctness.append(0)
            except:
                correctness.append(0)
        
        if len(correctness) == len(token_probs):
            try:
                corr = np.corrcoef(token_probs, correctness)[0, 1]
                metrics['token_prob_correctness_corr'] = corr if not np.isnan(corr) else np.nan
            except:
                metrics['token_prob_correctness_corr'] = np.nan
        else:
            metrics['token_prob_correctness_corr'] = np.nan
    else:
        metrics['token_prob_correctness_corr'] = np.nan
    
    # Mean absolute difference between stated confidence and token probability
    if stated_confidences and token_probs and len(stated_confidences) == len(token_probs):
        try:
            mae = mean_absolute_error(stated_confidences, token_probs)
            metrics['confidence_token_prob_mae'] = mae
        except:
            metrics['confidence_token_prob_mae'] = np.nan
    else:
        metrics['confidence_token_prob_mae'] = np.nan
    
    # Gini coefficient for probability distributions
    gini_coefficients = []
    if has_confidence_cols:
        for _, row in df.iterrows():
            try:
                probs = [float(row[col]) for col in ['A', 'B', 'C', 'D'] 
                        if col in row and pd.notna(row[col])]
                if probs and sum(probs) > 0:
                    # Normalize to sum to 1
                    probs = np.array(probs) / sum(probs)
                    gini = compute_gini(probs)
                    gini_coefficients.append(gini)
            except (ValueError, KeyError, TypeError):
                pass
    
    if gini_coefficients:
        metrics['avg_gini'] = np.mean(gini_coefficients)
    else:
        metrics['avg_gini'] = np.nan
    
    return metrics

def compute_ece(confidences: List[float], df: pd.DataFrame, correct_answers: int) -> float:
    """Compute Expected Calibration Error using 11 bins."""
    if not confidences:
        return np.nan
    
    # Create 11 bins: [0-10), [10-20), ..., [100]
    bins = np.linspace(0, 100, 12)
    bin_indices = np.digitize(confidences, bins) - 1
    
    ece = 0
    for i in range(11):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_confidences = np.array(confidences)[mask]
            bin_correctness = []
            
            for idx, row in df[mask].iterrows():
                try:
                    if str(row['answer']).strip() == str(row['Answer']).strip():
                        bin_correctness.append(1)
                    else:
                        bin_correctness.append(0)
                except:
                    bin_correctness.append(0)
            
            if bin_correctness:
                avg_confidence = np.mean(bin_confidences)
                accuracy = np.mean(bin_correctness)
                bin_size = len(bin_correctness)
                
                ece += (bin_size / len(df)) * abs(avg_confidence - accuracy)
    
    return ece

def compute_gini(probs: np.ndarray) -> float:
    """Compute Gini coefficient for a probability distribution."""
    if len(probs) == 0:
        return 0
    
    # Sort probabilities in ascending order
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    
    # Compute Gini coefficient
    cumsum = np.cumsum(sorted_probs)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

def generate_plots(df: pd.DataFrame, model: str, dataset: str, output_dir: str):
    """Generate all plots for a given model/dataset combination."""
    # Create output directory
    plot_dir = Path(output_dir) / model / dataset
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    stated_confidences = []
    token_probs = []
    correctness = []
    probability_distributions = []
    
    # Check if required columns exist
    has_answer_col = 'answer' in df.columns
    has_answer_col_cap = 'Answer' in df.columns
    # Check if at least some confidence columns exist (A, B, C, D are required, E is optional)
    has_confidence_cols = all(col in df.columns for col in ['A', 'B', 'C', 'D'])
    has_token_prob_col = 't1_prob' in df.columns
    
    for _, row in df.iterrows():
        # Correctness
        if has_answer_col and has_answer_col_cap:
            try:
                is_correct = 1 if str(row['answer']).strip() == str(row['Answer']).strip() else 0
                correctness.append(is_correct)
            except:
                correctness.append(0)
        else:
            correctness.append(0)
        
        # Stated confidence
        if has_confidence_cols:
            try:
                answer_letter = str(row.get('Answer', '')).strip()
                if answer_letter in ['A', 'B', 'C', 'D']:
                    confidence = float(row[answer_letter])
                    stated_confidences.append(confidence)
                else:
                    stated_confidences.append(np.nan)
            except (ValueError, KeyError, TypeError):
                stated_confidences.append(np.nan)
        else:
            stated_confidences.append(np.nan)
        
        # Token probability
        if has_token_prob_col:
            try:
                token_prob_val = row['t1_prob']
                if pd.notna(token_prob_val):
                    token_prob = float(token_prob_val)
                    token_probs.append(token_prob)
                else:
                    token_probs.append(np.nan)
            except (ValueError, KeyError, TypeError):
                token_probs.append(np.nan)
        else:
            token_probs.append(np.nan)
        
        # Probability distribution
        if has_confidence_cols:
            try:
                probs = [float(row[col]) for col in ['A', 'B', 'C', 'D'] 
                        if col in row and pd.notna(row[col])]
                if probs and sum(probs) > 0:
                    probs = np.array(probs) / sum(probs)
                    probability_distributions.append(probs)
                else:
                    probability_distributions.append(np.nan)
            except (ValueError, KeyError, TypeError):
                probability_distributions.append(np.nan)
        else:
            probability_distributions.append(np.nan)
    
    # Convert to numpy arrays
    stated_confidences = np.array(stated_confidences)
    token_probs = np.array(token_probs)
    correctness = np.array(correctness)
    
    # 1. Calibration Plot - Stated Confidence
    if has_confidence_cols and not np.all(np.isnan(stated_confidences)):
        try:
            plt.figure(figsize=(10, 8))
            plot_calibration(stated_confidences, correctness, 'Stated Confidence', 'Accuracy')
            plt.title(f'Calibration Plot - {model} on {dataset}\n(Stated Confidence)')
            plt.tight_layout()
            plt.savefig(plot_dir / 'calibration_stated_confidence.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Error generating stated confidence calibration plot: {e}")
    
    # 2. Calibration Plot - Token Probability
    if has_token_prob_col and not np.all(np.isnan(token_probs)):
        try:
            plt.figure(figsize=(10, 8))
            plot_calibration(token_probs, correctness, 'Token Probability', 'Accuracy')
            plt.title(f'Calibration Plot - {model} on {dataset}\n(Token Probability)')
            plt.tight_layout()
            plt.savefig(plot_dir / 'calibration_token_probability.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Error generating token probability calibration plot: {e}")
    
    # 3. Histogram - Stated Confidence
    if has_confidence_cols and not np.all(np.isnan(stated_confidences)):
        try:
            plt.figure(figsize=(10, 6))
            valid_confidences = stated_confidences[~np.isnan(stated_confidences)]
            if len(valid_confidences) > 0:
                plt.hist(valid_confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Stated Confidence')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Stated Confidence - {model} on {dataset}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'histogram_stated_confidence.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"    Error generating stated confidence histogram: {e}")
    
    # 4. Histogram - Token Probability
    if has_token_prob_col and not np.all(np.isnan(token_probs)):
        try:
            plt.figure(figsize=(10, 6))
            valid_token_probs = token_probs[~np.isnan(token_probs)]
            if len(valid_token_probs) > 0:
                plt.hist(valid_token_probs, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('Token Probability')
                plt.ylabel('Frequency')
                plt.title(f'Distribution of Token Probability - {model} on {dataset}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'histogram_token_probability.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"    Error generating token probability histogram: {e}")
    
    # 5. Scatter Plot - Stated Confidence vs Token Probability
    if has_confidence_cols and has_token_prob_col and not np.all(np.isnan(stated_confidences)) and not np.all(np.isnan(token_probs)):
        try:
            # Remove rows with NaN values
            mask = ~(np.isnan(stated_confidences) | np.isnan(token_probs))
            if np.any(mask):
                valid_confidences = stated_confidences[mask]
                valid_token_probs = token_probs[mask]
                
                plt.figure(figsize=(10, 8))
                plt.scatter(valid_confidences, valid_token_probs, alpha=0.6)
                
                # Add correlation line
                if len(valid_confidences) > 1:
                    corr = np.corrcoef(valid_confidences, valid_token_probs)[0, 1]
                    if not np.isnan(corr):
                        plt.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
                                transform=plt.gca().transAxes, fontsize=12,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.xlabel('Stated Confidence')
                plt.ylabel('Token Probability')
                plt.title(f'Stated Confidence vs Token Probability - {model} on {dataset}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'scatter_confidence_vs_token_prob.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"    Error generating scatter plot: {e}")
    
    # 6. Distribution Plot - Sum of Probabilities
    if has_confidence_cols and probability_distributions and not all(pd.isna(probability_distributions)):
        try:
            valid_dists = [dist for dist in probability_distributions if not pd.isna(dist)]
            if valid_dists:
                plt.figure(figsize=(12, 6))
                
                # Plot individual distributions
                for i, dist in enumerate(valid_dists[:100]):  # Limit to first 100 for clarity
                    plt.plot(range(len(dist)), dist, alpha=0.1, color='blue')
                
                # Plot mean distribution
                mean_dist = np.mean(valid_dists, axis=0)
                plt.plot(range(len(mean_dist)), mean_dist, 'r-', linewidth=2, label='Mean Distribution')
                
                plt.xlabel('Option Index')
                plt.ylabel('Probability')
                plt.title(f'Probability Distributions - {model} on {dataset}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plot_dir / 'distribution_probabilities.png', dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"    Error generating probability distribution plot: {e}")

def plot_calibration(confidences: np.ndarray, correctness: np.ndarray, xlabel: str, ylabel: str):
    """Helper function to create calibration plots."""
    # Create 11 bins: [0-10), [10-20), ..., [100]
    bins = np.linspace(0, 100, 12)
    bin_indices = np.digitize(confidences, bins) - 1
    
    bin_accuracies = []
    bin_confidences = []
    bin_sizes = []
    
    for i in range(11):
        mask = (bin_indices == i)
        if np.any(mask):
            bin_conf = confidences[mask]
            bin_corr = correctness[mask]
            
            if len(bin_corr) > 0:
                bin_accuracies.append(np.mean(bin_corr))
                bin_confidences.append(np.mean(bin_conf))
                bin_sizes.append(len(bin_corr))
    
    if bin_accuracies:
        # Plot calibration curve
        plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=8, label='Calibration')
        
        # Plot perfect calibration line
        plt.plot([0, 100], [0, 1], '--', color='red', alpha=0.7, label='Perfect Calibration')
        
        # Add bin sizes as text
        for i, (conf, acc, size) in enumerate(zip(bin_confidences, bin_accuracies, bin_sizes)):
            plt.annotate(f'n={size}', (conf, acc), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()

def find_csv_files() -> List[Tuple[str, str, str]]:
    """Find all CSV files in the data directories."""
    csv_files = []
    
    # Search in data/parsed
    parsed_dir = Path('data/parsed')
    if parsed_dir.exists():
        for csv_file in parsed_dir.rglob('*.csv'):
            if csv_file.is_file():
                model, dataset = extract_model_and_dataset(csv_file.name)
                csv_files.append((str(csv_file), model, dataset))
    
    # Search in data/otherllms
    otherllms_dir = Path('data/otherllms')
    if otherllms_dir.exists():
        for csv_file in otherllms_dir.rglob('*.csv'):
            if csv_file.is_file() and not csv_file.name.startswith('.'):
                model, dataset = extract_model_and_dataset(csv_file.name)
                csv_files.append((str(csv_file), model, dataset))
    
    return csv_files

def main():
    """Main function to run the analysis."""
    print("Starting LLM Calibration Analysis...")
    
    # Find all CSV files
    csv_files = find_csv_files()
    print(f"Found {len(csv_files)} CSV files to analyze")
    
    # Create output directory for results
    output_dir = Path('analysis/analysis_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store results
    all_results = []
    
    # Process each CSV file
    for csv_path, model, dataset in csv_files:
        print(f"\nProcessing: {model} on {dataset}")
        print(f"File: {csv_path}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            print(f"  Loaded {len(df)} rows")
            
            # Compute metrics
            metrics = compute_metrics(df)
            metrics['model'] = model
            metrics['dataset'] = dataset
            
            # Round numeric values to 4 decimal places
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics[key] = round(value, 4)
            
            all_results.append(metrics)
            
            # Skip plot generation for now - focus on metrics only
            # print(f"  Generating plots...")
            # generate_plots(df, model, dataset, output_dir)
            # print(f"  Plots saved to analysis/analysis_results/plots/{model}/{dataset}/")
            
        except Exception as e:
            print(f"  Error processing {csv_path}: {str(e)}")
            continue
    
    # Create summary DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns for better readability
        column_order = [
            'model', 'dataset', 'questions_asked', 'responses_formatted_correctly_pct',
            'correct_answers', 'accuracy', 'min_confidence', 'max_confidence',
            'avg_confidence', 'std_confidence', 'avg_token_prob', 'std_token_prob',
            'ece', 'overconfidence', 'confidence_correctness_corr', 'token_prob_correctness_corr',
            'confidence_token_prob_mae', 'avg_gini'
        ]
        
        # Only include columns that exist
        existing_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[existing_columns]
        
        # Save summary CSV to analysis_results directory
        summary_path = Path('analysis/analysis_results/analysis_summary.csv')
        results_df.to_csv(summary_path, index=False)
        print(f"\nAnalysis complete! Summary saved to {summary_path}")
        
        # Display summary
        print("\nSummary of Results:")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
    else:
        print("No results to display.")

if __name__ == "__main__":
    main()
