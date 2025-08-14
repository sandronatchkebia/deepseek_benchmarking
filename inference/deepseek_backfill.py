# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import asyncio
from pathlib import Path
import sys
sys.path.append('.')
from inference.deepseek import (
    client, 
    _call_with_retry, 
    MODEL_NAME, 
    SAVE_DIR
)

# Create backfill directory
BACKFILL_DIR = "data/full/backfill"
os.makedirs(BACKFILL_DIR, exist_ok=True)

async def identify_incomplete_responses(parsed_csv_path, dataset_name):
    """
    Identify questions that had incomplete responses due to token limits.
    
    Args:
        parsed_csv_path (str): Path to the parsed CSV file
        dataset_name (str): Name of the dataset
        
    Returns:
        tuple: (incomplete_df, original_prompts_df)
    """
    print(f"🔍 Analyzing {parsed_csv_path} for incomplete responses...")
    
    # Read the parsed CSV
    df = pd.read_csv(parsed_csv_path)
    
    # Check if is_incomplete column exists, if not, check finish_reason
    if 'is_incomplete' in df.columns:
        incomplete_mask = df['is_incomplete'] == True
        incomplete_df = df[incomplete_mask].copy()
        print(f"   Found {len(incomplete_df)} incomplete responses out of {len(df)} total")
    elif 'finish_reason' in df.columns:
        # Check for length (token limit) finish reasons
        incomplete_mask = df['finish_reason'] == 'length'
        incomplete_df = df[incomplete_mask].copy()
        print(f"   Found {len(incomplete_df)} responses that hit token limits out of {len(df)} total")
    else:
        print(f"   ⚠️  No completion status columns found, skipping dataset")
        return None, None
    
    if len(incomplete_df) == 0:
        print("   ✅ No incomplete responses found!")
        return None, None
    
    # Use the passed dataset_name parameter instead of extracting from path
    
    # Map dataset names to their original prompt files
    dataset_prompt_map = {
        'life_eval': 'life_eval_formatted.csv',
        'boolq_valid': 'boolq_valid_formatted.csv',
        'halu_eval_qa': 'halu_eval_qa_formatted.csv',
        'lsat_ar_test': 'lsat_ar_test_formatted.csv',
        'sat_en': 'sat_en_formatted.csv',
        'sciq_test': 'sciq_test_formatted.csv'
    }
    
    if dataset_name not in dataset_prompt_map:
        print(f"   ❌ Unknown dataset: {dataset_name}")
        return None, None
    
    # Download original prompts
    from inference.deepseek import download_prompt_csv, functions_map
    
    try:
        original_df = download_prompt_csv(dataset_name)
        format_function = functions_map[dataset_name]
        prompts_df = format_function(original_df)
        
        # Filter prompts to only include incomplete questions
        incomplete_prompts = prompts_df[prompts_df['Question ID'].isin(incomplete_df['Question ID'])].copy()
        
        print(f"   📝 Retrieved {len(incomplete_prompts)} original prompts for incomplete questions")
        
        return incomplete_df, incomplete_prompts
        
    except Exception as e:
        print(f"   ❌ Error retrieving original prompts: {e}")
        return None, None


async def backfill_incomplete_responses(dataset_name, incomplete_prompts_df, system_prompt, model=MODEL_NAME):
    """
    Re-run inference on incomplete responses with higher token limits.
    
    Args:
        dataset_name (str): Name of the dataset
        incomplete_prompts_df (pd.DataFrame): DataFrame with prompts for incomplete questions
        system_prompt (str): System prompt for the dataset
        model (str): Model name to use
        
    Returns:
        list: List of response results
    """
    if len(incomplete_prompts_df) == 0:
        print(f"   ⚠️  No incomplete prompts to process for {dataset_name}")
        return []
    
    print(f"   🔄 Re-running inference on {len(incomplete_prompts_df)} incomplete questions...")
    
    results = []
    semaphore = asyncio.Semaphore(8)  # Lower concurrency for backfill
    total_count = len(incomplete_prompts_df)
    
    async def process_incomplete_row(row):
        qid = row["Question ID"]
        user_prompt = row["Full Prompt"]
        
        # Retry mechanism for token limit issues
        max_retries = 3
        attempt = 1
        
        while attempt <= max_retries:
            try:
                print(f"     🔄 QID {qid} - Attempt {attempt}/{max_retries}")
                
                # Use higher token limit for backfill, increasing on retries
                max_tokens = 8192 * (2 ** (attempt - 1))  # 8192, 16384, 32768
                print(f"     📝 Using max_tokens: {max_tokens}")
                
                response = await _call_with_retry(
                    system_prompt, 
                    user_prompt, 
                    model, 
                    request_timeout=180,  # Longer timeout
                    semaphore=semaphore, 
                    qid=qid,
                    max_tokens=max_tokens
                )
                
                # Check if still hitting token limit
                if response.choices[0].finish_reason == "length":
                    if attempt < max_retries:
                        print(f"     ⚠️  QID {qid} hit token limit (attempt {attempt}), retrying...")
                        attempt += 1
                        continue
                    else:
                        print(f"     ❌ QID {qid} still hitting token limit after {max_retries} attempts")
                        return {
                            "question_id": qid,
                            "prompt": user_prompt,
                            "response": response.choices[0].message.content,
                            "raw_response": response.model_dump(),
                            "still_incomplete": True,
                            "attempts": max_retries
                        }
                
                # Success! Question completed without hitting token limit
                print(f"     ✅ QID {qid} successfully completed on attempt {attempt}")
                return {
                    "question_id": qid,
                    "prompt": user_prompt,
                    "response": response.choices[0].message.content,
                    "raw_response": response.model_dump(),
                    "still_incomplete": False,
                    "attempts": attempt
                }
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"     ⚠️  QID {qid} error on attempt {attempt}: {e}, retrying...")
                    attempt += 1
                    continue
                else:
                    print(f"     ❌ QID {qid} failed after {max_retries} attempts: {e}")
                    return {
                        "question_id": qid,
                        "error": str(e),
                        "attempts": max_retries
                    }
    
    # Process all incomplete questions
    tasks = [process_incomplete_row(row) for _, row in incomplete_prompts_df.iterrows()]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    print(f"   ✅ Completed backfill for {len(results)} questions")
    return results


async def main():
    """
    Main function to process all datasets and backfill incomplete responses.
    """
    print("🚀 Starting DeepSeek Backfill Process")
    print("=" * 50)
    
    # Get all parsed CSV files
    parsed_dir = Path("data/parsed/deepseek_r1")
    csv_files = list(parsed_dir.glob("*_wide.csv"))
    
    print(f"Found {len(csv_files)} parsed datasets to check:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    print("\n" + "=" * 50)
    
    for csv_file in csv_files:
        # Extract dataset name more carefully - handle underscores properly
        filename = csv_file.stem
        
        if 'life_eval' in filename:
            dataset_name = 'life_eval'
        elif 'boolq_valid' in filename:
            dataset_name = 'boolq_valid'
        elif 'halu_eval_qa' in filename:
            dataset_name = 'halu_eval_qa'
        elif 'lsat_ar_test' in filename:
            dataset_name = 'lsat_ar_test'
        elif 'sciq_test' in filename:
            dataset_name = 'sciq_test'
        elif 'sat_en' in filename:
            dataset_name = 'sat_en'
        else:
            dataset_name = filename.split('_')[0]  # fallback
        
        # Process all datasets that have incomplete responses
        # No skipping - process everything
        
        print(f"\n📊 Processing dataset: {dataset_name}")
        print(f"   🔍 Checking for incomplete responses...")
        
        # Identify incomplete responses
        incomplete_df, incomplete_prompts = await identify_incomplete_responses(str(csv_file), dataset_name)
        
        if incomplete_prompts is None or len(incomplete_prompts) == 0:
            continue
        
        # Get system prompt from the prompts
        system_prompt = incomplete_prompts["System Prompt"].iloc[0]
        
        print(f"   🚀 Starting backfill for {len(incomplete_prompts)} questions on {MODEL_NAME}...")
        print(f"   📝 Dataset: {dataset_name}")
        print(f"   🤖 Model: {MODEL_NAME}")
        
        # Run backfill
        backfill_results = await backfill_incomplete_responses(
            dataset_name, 
            incomplete_prompts, 
            system_prompt
        )
        
        if backfill_results:
            # Save backfill results
            output_file = os.path.join(BACKFILL_DIR, f"{dataset_name}_backfill_results.json")
            with open(output_file, "w") as f:
                json.dump(backfill_results, f, indent=2)
            
            print(f"   💾 Saved backfill results to: {output_file}")
            
            # Summary
            successful = sum(1 for r in backfill_results if 'error' not in r and not r.get('still_incomplete', False))
            still_incomplete = sum(1 for r in backfill_results if r.get('still_incomplete', False))
            errors = sum(1 for r in backfill_results if 'error' in r)
            
            print(f"   📊 Backfill Summary:")
            print(f"      Successfully completed: {successful}")
            print(f"      Still incomplete: {still_incomplete}")
            print(f"      Errors: {errors}")
    
    print("\n" + "=" * 50)
    print("✅ Backfill process completed!")
    print(f"📁 Results saved in: {BACKFILL_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
