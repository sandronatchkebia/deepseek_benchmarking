# -*- coding: utf-8 -*-
import os
import pandas as pd
from pathlib import Path
import shutil

def merge_backfill_with_original():
    """
    Merge backfill data with original parsed files, replacing incomplete rows with complete ones.
    """
    print("🔄 Starting Backfill Data Merge Process")
    print("=" * 50)
    
    # Define paths
    original_parsed_dir = Path("data/parsed/deepseek_r1")
    backfill_parsed_dir = Path("data/parsed/backfill")
    backup_dir = Path("data/parsed/deepseek_r1_backup")
    
    # Create backup directory
    backup_dir.mkdir(exist_ok=True)
    print(f"📁 Backup directory: {backup_dir}")
    
    # Get all backfill files
    backfill_files = list(backfill_parsed_dir.glob("*_backfill_results_wide.csv"))
    print(f"Found {len(backfill_files)} backfill files to merge")
    
    for backfill_file in backfill_files:
        # Extract dataset name from backfill filename
        dataset_name = backfill_file.stem.replace('_backfill_results_wide', '')
        print(f"\n📊 Processing dataset: {dataset_name}")
        
        # Find corresponding original file (handle different naming patterns)
        possible_names = [
            f"{dataset_name}_wide.csv",
            f"{dataset_name}_deepseek-reasoner_reasoning_results_wide.csv",
            f"{dataset_name}_deepseek-reasoner_reasoning_results__incomplete_wide.csv"
        ]
        
        original_file = None
        for name in possible_names:
            if (original_parsed_dir / name).exists():
                original_file = original_parsed_dir / name
                break
        
        if not original_file:
            print(f"   ⚠️  Original file not found for {dataset_name}")
            print(f"   🔍 Tried: {possible_names}")
            continue
        
        if not original_file.exists():
            print(f"   ⚠️  Original file not found: {original_file}")
            continue
        
        print(f"   📖 Reading original file: {original_file.name}")
        print(f"   📖 Reading backfill file: {backfill_file.name}")
        
        # Read both files
        try:
            original_df = pd.read_csv(original_file)
            backfill_df = pd.read_csv(backfill_file)
            
            print(f"   📊 Original: {len(original_df)} rows")
            print(f"   📊 Backfill: {len(backfill_df)} rows")
            
            # Create backup of original file
            backup_file = backup_dir / f"{dataset_name}_wide_backup.csv"
            shutil.copy2(original_file, backup_file)
            print(f"   💾 Backup created: {backup_file.name}")
            
            # Check if original has incomplete responses
            if 'is_incomplete' in original_df.columns:
                incomplete_mask = original_df['is_incomplete'] == True
                incomplete_count = incomplete_mask.sum()
                print(f"   🔍 Found {incomplete_count} incomplete responses in original")
                
                if incomplete_count > 0:
                    # Get question IDs of incomplete responses
                    incomplete_qids = set(original_df[incomplete_mask]['Question ID'])
                    print(f"   🔍 Incomplete QIDs: {sorted(incomplete_qids)}")
                    
                    # Get question IDs from backfill
                    backfill_qids = set(backfill_df['Question ID'])
                    print(f"   🔍 Backfill QIDs: {sorted(backfill_qids)}")
                    
                    # Verify all incomplete QIDs are covered by backfill
                    missing_qids = incomplete_qids - backfill_qids
                    if missing_qids:
                        print(f"   ⚠️  Warning: Some incomplete QIDs not in backfill: {sorted(missing_qids)}")
                    
                    # Remove incomplete rows from original
                    original_df_clean = original_df[~incomplete_mask].copy()
                    print(f"   🧹 Removed {incomplete_count} incomplete rows from original")
                    
                    # Add backfill data
                    merged_df = pd.concat([original_df_clean, backfill_df], ignore_index=True)
                    print(f"   🔗 Merged: {len(merged_df)} total rows")
                    
                    # Sort by Question ID to maintain order
                    merged_df = merged_df.sort_values('Question ID').reset_index(drop=True)
                    
                    # Save merged file
                    merged_df.to_csv(original_file, index=False)
                    print(f"   💾 Saved merged file: {original_file.name}")
                    
                    # Verify merge
                    final_df = pd.read_csv(original_file)
                    print(f"   ✅ Final file: {len(final_df)} rows")
                    
                    # Check for any remaining incomplete responses
                    if 'is_incomplete' in final_df.columns:
                        remaining_incomplete = (final_df['is_incomplete'] == True).sum()
                        print(f"   🔍 Remaining incomplete: {remaining_incomplete}")
                    
                else:
                    print(f"   ✅ No incomplete responses found, appending backfill data")
                    # No incomplete responses, just append backfill data
                    merged_df = pd.concat([original_df, backfill_df], ignore_index=True)
                    merged_df = merged_df.sort_values('Question ID').reset_index(drop=True)
                    merged_df.to_csv(original_file, index=False)
                    print(f"   💾 Saved merged file: {original_file.name}")
            
            elif 'finish_reason' in original_df.columns:
                # Check for length (token limit) finish reasons
                incomplete_mask = original_df['finish_reason'] == 'length'
                incomplete_count = incomplete_mask.sum()
                print(f"   🔍 Found {incomplete_count} responses that hit token limits in original")
                
                if incomplete_count > 0:
                    # Get question IDs of incomplete responses
                    incomplete_qids = set(original_df[incomplete_mask]['Question ID'])
                    print(f"   🔍 Incomplete QIDs: {sorted(incomplete_qids)}")
                    
                    # Get question IDs from backfill
                    backfill_qids = set(backfill_df['Question ID'])
                    print(f"   🔍 Backfill QIDs: {sorted(backfill_qids)}")
                    
                    # Remove incomplete rows from original
                    original_df_clean = original_df[~incomplete_mask].copy()
                    print(f"   🧹 Removed {incomplete_count} incomplete rows from original")
                    
                    # Add backfill data
                    merged_df = pd.concat([original_df_clean, backfill_df], ignore_index=True)
                    print(f"   🔗 Merged: {len(merged_df)} total rows")
                    
                    # Sort by Question ID to maintain order
                    merged_df = merged_df.sort_values('Question ID').reset_index(drop=True)
                    
                    # Save merged file
                    merged_df.to_csv(original_file, index=False)
                    print(f"   💾 Saved merged file: {original_file.name}")
                    
                    # Verify merge
                    final_df = pd.read_csv(original_file)
                    print(f"   ✅ Final file: {len(final_df)} rows")
                    
                    # Check for any remaining token limit responses
                    remaining_incomplete = (final_df['finish_reason'] == 'length').sum()
                    print(f"   🔍 Remaining token limit hits: {remaining_incomplete}")
                
                else:
                    print(f"   ✅ No token limit responses found, appending backfill data")
                    # No incomplete responses, just append backfill data
                    merged_df = pd.concat([original_df, backfill_df], ignore_index=True)
                    merged_df = merged_df.sort_values('Question ID').reset_index(drop=True)
                    merged_df.to_csv(original_file, index=False)
                    print(f"   💾 Saved merged file: {original_file.name}")
            
            else:
                print(f"   ⚠️  No completion status columns found, appending backfill data")
                # No completion status columns, just append backfill data
                merged_df = pd.concat([original_df, backfill_df], ignore_index=True)
                merged_df = merged_df.sort_values('Question ID').reset_index(drop=True)
                merged_df.to_csv(original_file, index=False)
                print(f"   💾 Saved merged file: {original_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("✅ Backfill merge process completed!")
    print(f"📁 Original files updated in: {original_parsed_dir}")
    print(f"📁 Backup files saved in: {backup_dir}")
    print("\n📋 Summary:")
    print("   - Incomplete rows replaced with complete backfill data")
    print("   - All question IDs preserved")
    print("   - Original files backed up before modification")
    print("   - Files sorted by Question ID for consistency")


if __name__ == "__main__":
    merge_backfill_with_original()
