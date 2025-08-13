#!/usr/bin/env python3
"""
Processing script to create final tables with correct answers.

This script:
1. Loads parsed outputs from data/parsed/ and data/otherllms/
2. Loads original data from data/full/ to get correct answers
3. Merges them to create final processed tables
4. Saves results in data/processed/ with model subfolders
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

# GitHub repository base URL for downloading formatted benchmarks with correct answers
GITHUB_PROMPT_BASE = "https://raw.githubusercontent.com/NoamMichael/Comparing-Confidence-in-LLMs/refs/heads/main/Formatted%20Benchmarks"

def download_correct_answers(dataset_name: str) -> Dict[int, str]:
    """
    Download correct answers from GitHub repository.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'sat_en', 'lsat_ar_test')
    
    Returns:
        Dictionary mapping question_id to correct answer
    """
    correct_answers = {}
    
    try:
        url = f"{GITHUB_PROMPT_BASE}/{dataset_name}_formatted.csv"
        print(f"    Downloading {dataset_name}_formatted.csv from GitHub...")
        
        response = requests.get(url)
        if response.status_code != 200:
            print(f"    Warning: Failed to download {url} (status: {response.status_code})")
            return correct_answers
        
        # Parse the CSV
        df = pd.read_csv(StringIO(response.text))
        print(f"    Downloaded {len(df)} questions from {dataset_name}")
        
        # Extract correct answers based on dataset type
        if 'Question ID' not in df.columns:
            print(f"    Warning: Question ID column not found in {dataset_name}")
            print(f"    Available columns: {list(df.columns)}")
            return correct_answers
        
        # Handle different column structures for different datasets
        if 'Correct Answer Letter' in df.columns:
            # Standard multiple choice format (sat_en, lsat_ar_test, sciq_test)
            for _, row in df.iterrows():
                question_id = row['Question ID']
                correct_answer = row['Correct Answer Letter']
                if pd.notna(question_id) and pd.notna(correct_answer):
                    correct_answers[int(question_id)] = str(correct_answer).strip().upper()
            
            print(f"    Found {len(correct_answers)} correct answers (multiple choice)")
            
        elif 'right_answer' in df.columns:
            # halu_eval_qa format
            for _, row in df.iterrows():
                question_id = row['Question ID']
                correct_answer = row['right_answer']
                if pd.notna(question_id) and pd.notna(correct_answer):
                    correct_answers[int(question_id)] = str(correct_answer).strip()
            
            print(f"    Found {len(correct_answers)} correct answers (halu_eval_qa)")
            
        elif 'True Lifespan' in df.columns:
            # life_eval format (numeric answers)
            for _, row in df.iterrows():
                question_id = row['Question ID']
                correct_answer = row['True Lifespan']
                if pd.notna(question_id) and pd.notna(correct_answer):
                    correct_answers[int(question_id)] = str(correct_answer)
            
            print(f"    Found {len(correct_answers)} correct answers (life_eval)")
            
        elif 'Correct Answer' in df.columns:
            # boolq_valid format (True/False answers)
            for _, row in df.iterrows():
                question_id = row['Question ID']
                correct_answer = row['Correct Answer']
                if pd.notna(question_id) and pd.notna(correct_answer):
                    correct_answers[int(question_id)] = str(correct_answer).strip()
            
            print(f"    Found {len(correct_answers)} correct answers (boolq_valid)")
            
        else:
            print(f"    Warning: No recognized answer column found in {dataset_name}")
            print(f"    Available columns: {list(df.columns)}")
    
    except Exception as e:
        print(f"    Error downloading {dataset_name}: {e}")
    
    return correct_answers

def load_original_data(dataset_name: str, model_type: str = "deepseek") -> Dict[int, str]:
    """
    Load correct answers by downloading from GitHub repository.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'sat_en', 'lsat_ar_test')
        model_type: Type of model data to load ('deepseek' or 'other')
    
    Returns:
        Dictionary mapping question_id to correct answer
    """
    # Download correct answers from GitHub
    return download_correct_answers(dataset_name)

def process_parsed_file(parsed_file_path: str, dataset_name: str, model_name: str) -> Optional[pd.DataFrame]:
    """
    Process a parsed CSV file and add correct answers.
    
    Args:
        parsed_file_path: Path to the parsed CSV file
        dataset_name: Name of the dataset
        model_name: Name of the model
    
    Returns:
        Processed DataFrame with correct answers, or None if processing fails
    """
    try:
        # Load parsed data
        df = pd.read_csv(parsed_file_path)
        print(f"    Loaded {len(df)} rows from {parsed_file_path}")
        
        # Load correct answers from original data
        correct_answers = load_original_data(dataset_name, "deepseek" if "deepseek" in model_name.lower() else "other")
        
        if not correct_answers:
            print(f"    Warning: No correct answers found for {dataset_name}")
            return None
        
        # Add correct answer column
        if dataset_name == 'halu_eval_qa':
            # For halu_eval_qa, strip the _r/_h suffix and convert to int when mapping question IDs
            df['correct_answer'] = df['Question ID'].str.replace(r'_[rh]$', '', regex=True).astype(int).map(correct_answers)
        else:
            # For other datasets, use question IDs as-is
            df['correct_answer'] = df['Question ID'].map(correct_answers)
        
        # Check how many correct answers we found
        found_answers = df['correct_answer'].notna().sum()
        print(f"    Found correct answers for {found_answers}/{len(df)} questions")
        
        # Only proceed if we have at least some correct answers
        if found_answers == 0:
            print(f"    Warning: No correct answers found for {dataset_name}, skipping...")
            return None
        
        # Add model and dataset columns for clarity
        df['model'] = model_name
        df['dataset'] = dataset_name
        
        return df
        
    except Exception as e:
        print(f"    Error processing {parsed_file_path}: {e}")
        return None

def find_and_process_files():
    """
    Find all parsed files and process them to create final tables.
    """
    # Create output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process DeepSeek files
    deepseek_dir = Path("data/parsed")
    if deepseek_dir.exists():
        print("Processing DeepSeek files...")
        
        for model_dir in deepseek_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                print(f"  Processing {model_name}...")
                
                # Create output subdirectory
                model_output_dir = output_dir / model_name
                model_output_dir.mkdir(exist_ok=True)
                
                for csv_file in model_dir.glob("*.csv"):
                    if csv_file.is_file():
                        # Extract dataset name from filename
                        filename = csv_file.stem
                        dataset_match = re.search(r'(.+?)_(?:deepseek|results)', filename)
                        if dataset_match:
                            dataset_name = dataset_match.group(1)
                        else:
                            dataset_name = filename
                        
                        print(f"    Processing {dataset_name}...")
                        
                        # Process the file
                        processed_df = process_parsed_file(str(csv_file), dataset_name, model_name)
                        
                        if processed_df is not None:
                            # Save processed file
                            output_file = model_output_dir / f"{dataset_name}_processed.csv"
                            processed_df.to_csv(output_file, index=False)
                            print(f"    Saved to {output_file}")
    
    # Process other LLM files
    otherllms_dir = Path("data/otherllms/Parsed Results")
    if otherllms_dir.exists():
        print("\nProcessing other LLM files...")
        
        for provider_dir in otherllms_dir.iterdir():
            if provider_dir.is_dir():
                provider_name = provider_dir.name
                print(f"  Processing {provider_name}...")
                
                for model_dir in provider_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        print(f"    Processing {model_name}...")
                        
                        # Create output subdirectory with the specific model name
                        model_output_dir = output_dir / model_name
                        model_output_dir.mkdir(exist_ok=True)
                        
                        for csv_file in model_dir.glob("*.csv"):
                            if csv_file.is_file() and not csv_file.name.startswith('.'):
                                # Extract dataset name from filename
                                filename = csv_file.stem
                                dataset_match = re.search(r'(.+?)_(?:gpt|claude|gemini)', filename)
                                if dataset_match:
                                    dataset_name = dataset_match.group(1)
                                else:
                                    dataset_name = filename
                                
                                print(f"      Processing {dataset_name}...")
                                
                                # Process the file
                                processed_df = process_parsed_file(str(csv_file), dataset_name, model_name)
                                
                                if processed_df is not None:
                                    # Save processed file
                                    output_file = model_output_dir / f"{dataset_name}_processed.csv"
                                    processed_df.to_csv(output_file, index=False)
                                    print(f"        Saved to {output_file}")

def main():
    """Main function to run the processing."""
    print("Starting processing to create final tables with correct answers...")
    
    find_and_process_files()
    
    print("\nProcessing complete!")
    print(f"Results saved in: {Path('data/processed').absolute()}")

if __name__ == "__main__":
    main()
