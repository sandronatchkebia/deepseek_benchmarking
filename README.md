# LLM Calibration Analysis

A comprehensive toolkit for Analyzing confidence calibration of DeepSeek V3 and R1.

## 🎯 Overview

This repository contains tools and parsed results for analyzing LLM confidence calibration across diverse datasets including multiple choice questions, open-ended responses, and numeric evaluations.

## 🚀 Features

- **Multi-format Support**: Handles JSON, JSONL, and various response formats
- **Dataset Type Detection**: Automatically identifies MCQ, TRUE_FALSE, NUMERIC, and OPEN formats
- **Logprob Extraction**: Extracts token-level probabilities and alternative token considerations
- **Token Limit Monitoring**: Tracks and reports queries that hit max_token limits
- **Early Abort Protection**: Stops processing when token limits are exceeded to prevent incomplete results
- **Intelligent Backfill System**: Automatically re-runs inference on incomplete responses with progressive token limit increases
- **Flexible Output**: Generates wide-format CSV files compatible with analysis tools
- **Cross-dataset Compatibility**: Produces outputs similar to reference GPT-4o parsed results
- **Complete Data Recovery**: Ensures 100% completion rates through automated backfill processing

## 📁 Repository Structure

```
├── parsing/                          # Parsing scripts organized in dedicated folder
│   ├── deepseek_parsing.py          # Main parsing script for V3 model (with logprobs)
│   ├── deepseek_parsing_reasoning.py # Parsing script for R1 model (reasoning only)
│   └── merging_backfill.py          # Script to merge backfill data with original parsed files
├── inference/
│   ├── deepseek.py                  # Async inference script with token limit protection
│   └── deepseek_backfill.py         # Backfill script for incomplete responses with retry logic
├── data/
│   ├── parsed/                      # Parsed CSV outputs organized by model version
│   │   ├── deepseek_v3/             # DeepSeek v3 model results (with logprobs)
│   │   │   ├── sat_en_deepseek-chat_results_wide.csv
│   │   │   ├── lsat_ar_test_deepseek-chat_results_wide.csv
│   │   │   ├── sciq_test_deepseek-chat_results_wide.csv
│   │   │   ├── halu_eval_qa_deepseek-chat_results_wide.csv
│   │   │   ├── life_eval_deepseek-chat_results_wide.csv
│   │   │   └── boolq_valid_deepseek-chat_results_wide.csv
│   │   ├── deepseek_r1/             # DeepSeek R1 model results (reasoning only)
│   │   │   ├── sat_en_deepseek-reasoner_reasoning_results_wide.csv
│   │   │   ├── lsat_ar_test_deepseek-reasoner_reasoning_results_wide.csv
│   │   │   ├── sciq_test_deepseek-reasoner_reasoning_results_wide.csv
│   │   │   ├── halu_eval_qa_deepseek-reasoner_reasoning_results_wide.csv
│   │   │   ├── life_eval_deepseek-reasoner_reasoning_results__incomplete_wide.csv
│   │   │   └── boolq_valid_deepseek-reasoner_reasoning_results__incomplete_wide.csv
│   │   └── backfill/                # Backfill data for incomplete responses
│   │       ├── life_eval_backfill_results_wide.csv
│   │       ├── lsat_ar_test_backfill_results_wide.csv
│   │       └── boolq_valid_backfill_results_wide.csv
│   └── processed/                   # Final processed tables with correct answers
│       ├── deepseek_r1/             # Processed DeepSeek R1 results
│       ├── deepseek_v3/             # Processed DeepSeek V3 results
│       └── [other_models]/          # Processed results from other LLMs
└── README.md
```

## 🛠️ Usage

### Parsing Scripts

This repository contains two specialized parsing scripts:

1. **`parsing/deepseek_parsing.py`** - For DeepSeek V3 model outputs
   - Extracts token-level probabilities (logprobs)
   - Includes alternative token analysis
   - Best for confidence calibration research

2. **`parsing/deepseek_parsing_reasoning.py`** - For DeepSeek R1 model outputs  
   - Focuses on reasoning quality and answer extraction
   - No logprob dependencies (R1 API doesn't support them)
   - **NEW**: Token limit tracking and reporting
   - Best for reasoning analysis and answer accuracy

### Basic Usage

```bash
# Parse V3 model outputs (with logprobs)
python3 parsing/deepseek_parsing.py --infile <input_file> --dataset <dataset_name> --model deepseek-chat

# Parse R1 model outputs (reasoning only)
python3 parsing/deepseek_parsing_reasoning.py --infile <input_file> --dataset <dataset_name> --model deepseek-reasoner

# Parse with custom output directory
python3 parsing/deepseek_parsing.py --infile <input_file> --dataset <dataset_name> --model <model_name> --output_dir data/parsed/deepseek_v3

# Run backfill to recover incomplete responses
python3 inference/deepseek_backfill.py

# Merge backfill data with original parsed files
python3 parsing/merging_backfill.py
```

### Examples

```bash
# Parse SAT-EN dataset and save to deepseek_v3 folder
python3 parsing/deepseek_parsing.py --infile path/to/sat_en_deepseek-chat_results.json --dataset SAT-EN --model deepseek-chat --output_dir data/parsed/deepseek_v3

# Parse R1 outputs with token limit tracking
python3 parsing/deepseek_parsing_reasoning.py --infile path/to/lsat_ar_test_deepseek-reasoner_reasoning_results.json --dataset LSAT-AR --model deepseek-reasoner --output_dir data/parsed/deepseek_r1
```

### Arguments

- `--infile`: Path to input JSON/JSONL file (required)
- `--dataset`: Dataset name (required)
- `--model`: Model label (default: deepseek-chat for V3, deepseek-r1 for R1)
- `--output_dir`: Custom output directory (optional)

## 🚨 Token Limit Protection & Backfill System

### Inference Script Features

The `inference/deepseek.py` script now includes intelligent token limit protection:

- **Real-time Monitoring**: Checks `finish_reason` field during API calls
- **Early Abort**: Stops processing when first token limit is hit
- **Partial Results**: Saves completed queries with `__incomplete` suffix
- **Comprehensive Reporting**: Shows completion statistics and token limit hits

### Backfill System Features

The `inference/deepseek_backfill.py` script provides intelligent recovery of incomplete responses:

- **Automatic Detection**: Identifies questions with incomplete responses due to token limits
- **Progressive Retry Logic**: Uses increasing token limits (8K → 16K → 32K) across 3 attempts
- **Targeted Processing**: Only re-runs inference on problematic questions
- **Complete Recovery**: Achieves 100% completion rates for all datasets
- **Smart Merging**: Seamlessly integrates backfill data with original parsed files

### Parsing Script Features

Both parsing scripts now provide token limit analysis:

```
📊 Token Limit Summary:
   Total queries: 230
   Hit max_token limit: 132
   Successfully completed: 98
   ⚠️  132 queries may have incomplete responses due to token limits
```

### Benefits

- **Prevents Wasted API Calls**: Stop early when token limits are exceeded
- **Quality Assurance**: Identify datasets that need higher token limits
- **Resource Optimization**: Focus on datasets that can complete successfully
- **Transparent Reporting**: Clear visibility into response completion rates

## 📊 Supported Datasets

### Multiple Choice (MCQ)
- **SAT-EN**: English SAT questions with A, B, C, D, E options
- **LSAT-AR**: LSAT Analytical Reasoning with A, B, C, D, E options
- **SciQ**: Science questions with multiple choice answers

### Open-ended
- **HaluEval-QA**: Question-answering with confidence scores

### Numeric
- **LifeEval**: Life evaluation questions with numeric estimates and confidence intervals

## 🏗️ Model Version Organization

Results are organized by model version to facilitate comparison and analysis:

- **`deepseek_v3/`: Current DeepSeek v3 model results (6 datasets with logprobs)
- **`deepseek_r1/`: DeepSeek R1 model results (6 datasets, reasoning focus, 100% completion rate)
- **`backfill/`: Recovered incomplete responses with progressive token limit increases
- **`processed/`: Final tables with correct answers for accuracy analysis

## 🔍 Output Format

Each parsed dataset generates a `*_wide.csv` file containing:

### Common Columns
- `Question ID`: Unique identifier for each question
- `content`: Raw LLM response text
- `answer`: Extracted answer
- `correct_format`: Whether response was in expected format
- `coerce`: Whether response was coerced to expected format
- `Reasoning`: LLM's reasoning (if available)
- `Answer`: Final answer in standardized format

### V3 Model Specific Columns (with logprobs)
- `token_index`: Character position of answer in response
- `token`: Actual answer token
- `t1` through `t5`: Top 5 alternative tokens
- `t1_prob` through `t5_prob`: Probabilities of alternative tokens

### Dataset-Specific Columns
- **MCQ**: `A`, `B`, `C`, `D`, `E` columns with normalized probabilities
- **Open-ended**: `Confidence` column
- **Numeric**: `estimate`, `conf_within_y`, `y`, `Confidence` columns

## 🧠 Logprob Extraction (V3 Models Only)

The V3 parsing script intelligently extracts token-level probabilities by:
1. **Finding answer position** in the response text
2. **Mapping to token position** in the logprobs content array
3. **Extracting top alternatives** with their probabilities
4. **Handling edge cases** like malformed JSON and different response formats

## 🔧 Technical Details

- **Language**: Python 3.7+
- **Dependencies**: pandas, numpy, pathlib, json, re, ast, asyncio, tenacity
- **Input Formats**: JSON, JSONL with OpenAI API response structure
- **Output**: CSV files compatible with pandas and analysis tools
- **Async Support**: Concurrent API calls with rate limiting and retry logic

## 📈 Analysis Use Cases

- **Confidence Calibration**: Compare predicted vs. actual confidence
- **Token-level Analysis**: Understand model's decision-making process (V3 models)
- **Cross-dataset Comparison**: Analyze performance across different question types
- **Format Robustness**: Study how well models follow expected response formats
- **Token Limit Analysis**: Identify datasets that need higher token allocations
- **Response Quality Assessment**: Distinguish between complete and truncated responses
