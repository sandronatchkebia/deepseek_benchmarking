# LLM Calibration Analysis

A comprehensive toolkit for parsing and analyzing Large Language Model (LLM) response data across multiple datasets and formats.

## 🎯 Overview

This repository contains tools and parsed results for analyzing LLM confidence calibration across diverse datasets including multiple choice questions, open-ended responses, and numeric evaluations.

## 🚀 Features

- **Multi-format Support**: Handles JSON, JSONL, and various response formats
- **Dataset Type Detection**: Automatically identifies MCQ, TRUE_FALSE, NUMERIC, and OPEN formats
- **Logprob Extraction**: Extracts token-level probabilities and alternative token considerations
- **Flexible Output**: Generates wide-format CSV files compatible with analysis tools
- **Cross-dataset Compatibility**: Produces outputs similar to reference GPT-4o parsed results

## 📁 Repository Structure

```
├── deepseek_parsing.py          # Main parsing script
├── data/
│   └── parsed/                  # Parsed CSV outputs
│       ├── sat_en_deepseek-chat_results_wide.csv
│       ├── lsat_ar_test_deepseek-chat_results_wide.csv
│       ├── sciq_test_deepseek-chat_results_wide.csv
│       ├── halu_eval_qa_deepseek-chat_results_wide.csv
│       └── life_eval_deepseek-chat_results_wide.csv
└── README.md
```

## 🛠️ Usage

### Basic Usage

```bash
python3 deepseek_parsing.py --infile <input_file> --dataset <dataset_name> --model <model_name>
```

### Examples

```bash
# Parse SAT-EN dataset
python3 deepseek_parsing.py --infile path/to/sat_en_deepseek-chat_results.json --dataset SAT-EN --model deepseek-chat

# Parse with custom output directory
python3 deepseek_parsing.py --infile path/to/lsat_ar_test_deepseek-chat_results.json --dataset LSAT-AR --model deepseek-chat --output_dir data/parsed
```

### Arguments

- `--infile`: Path to input JSON/JSONL file (required)
- `--dataset`: Dataset name (required)
- `--model`: Model label (default: deepseek-chat)
- `--output_dir`: Custom output directory (optional)

## 📊 Supported Datasets

### Multiple Choice (MCQ)
- **SAT-EN**: English SAT questions with A, B, C, D, E options
- **LSAT-AR**: LSAT Analytical Reasoning with A, B, C, D, E options
- **SciQ**: Science questions with multiple choice answers

### Open-ended
- **HaluEval-QA**: Question-answering with confidence scores

### Numeric
- **LifeEval**: Life evaluation questions with numeric estimates and confidence intervals

## 🔍 Output Format

Each parsed dataset generates a `*_wide.csv` file containing:

### Common Columns
- `Question ID`: Unique identifier for each question
- `content`: Raw LLM response text
- `answer`: Extracted answer
- `token_index`: Character position of answer in response
- `token`: Actual answer token
- `t1` through `t5`: Top 5 alternative tokens
- `t1_prob` through `t5_prob`: Probabilities of alternative tokens
- `correct_format`: Whether response was in expected format
- `coerce`: Whether response was coerced to expected format
- `Reasoning`: LLM's reasoning (if available)
- `Answer`: Final answer in standardized format

### Dataset-Specific Columns
- **MCQ**: `A`, `B`, `C`, `D`, `E` columns with normalized probabilities
- **Open-ended**: `Confidence` column
- **Numeric**: `estimate`, `conf_within_y`, `y`, `Confidence` columns

## 🧠 Logprob Extraction

The script intelligently extracts token-level probabilities by:
1. **Finding answer position** in the response text
2. **Mapping to token position** in the logprobs content array
3. **Extracting top alternatives** with their probabilities
4. **Handling edge cases** like malformed JSON and different response formats

## 🔧 Technical Details

- **Language**: Python 3.7+
- **Dependencies**: pandas, numpy, pathlib, json, re, ast
- **Input Formats**: JSON, JSONL with OpenAI API response structure
- **Output**: CSV files compatible with pandas and analysis tools

## 📈 Analysis Use Cases

- **Confidence Calibration**: Compare predicted vs. actual confidence
- **Token-level Analysis**: Understand model's decision-making process
- **Cross-dataset Comparison**: Analyze performance across different question types
- **Format Robustness**: Study how well models follow expected response formats
