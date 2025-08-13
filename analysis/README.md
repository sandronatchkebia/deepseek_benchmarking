# LLM Calibration Analysis

This directory contains the analysis script for evaluating LLM calibration across different models and datasets.

## Overview

The `analysis.py` script processes parsed CSV outputs from LLM calibration experiments and computes comprehensive metrics and generates visualization plots for each model/dataset combination.

## Features

### Metrics Computed

For each model/dataset combination, the script computes:

- **Basic Statistics**: Questions asked, responses formatted correctly, correct answers, accuracy
- **Confidence Metrics**: Min/max/average/standard deviation of stated confidence values
- **Token Probability Metrics**: Average and standard deviation of token probabilities (if available)
- **Calibration Metrics**: Expected Calibration Error (ECE) using 11 bins
- **Overconfidence**: Average max confidence minus accuracy
- **Correlations**: Between stated confidence and correctness, token probability and correctness
- **Distribution Metrics**: Mean absolute difference between confidence and token probability, Gini coefficient

### Plots Generated

For each model/dataset combination, the script generates:

1. **Calibration Plot** - Confidence vs Accuracy with 11 bins (for both stated confidence and token probability)
2. **Histogram** - Distribution of stated confidence values
3. **Histogram** - Distribution of token probability values (if available)
4. **Scatter Plot** - Stated confidence vs token probability with correlation coefficient
5. **Distribution Plot** - Probability distributions across options with mean distribution
6. **Gini Coefficient** - Measure of probability distribution concentration

## Usage

### Prerequisites

Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
python3 analysis/analysis.py
```

The script will:
1. Automatically discover CSV files in `data/parsed/` and `data/otherllms/`
2. Process each file and compute metrics
3. Generate plots in `plots/{model}/{dataset}/`
4. Save a summary CSV as `analysis_summary.csv`

### Input Data Structure

The script expects CSV files with the following columns:

**Required columns:**
- `Question ID`: Unique identifier for each question
- `answer`: The correct answer
- `Answer`: The model's predicted answer

**Optional columns:**
- `A`, `B`, `C`, `D`, `E`: Confidence values for each option (A, B, C, D are required, E is optional)
- `t1_prob`: Token probability for the first token
- `correct_format`: Boolean indicating if response was properly formatted

### Output Structure

```
plots/
├── {model}/
│   ├── {dataset}/
│   │   ├── calibration_stated_confidence.png
│   │   ├── calibration_token_probability.png
│   │   ├── histogram_stated_confidence.png
│   │   ├── histogram_token_probability.png
│   │   ├── scatter_confidence_vs_token_prob.png
│   │   └── distribution_probabilities.png
```

## Supported Models

The script automatically detects and categorizes:
- **DeepSeek**: deepseek-reasoner, deepseek-chat
- **GPT**: gpt-4o
- **Claude**: claude-3-7-sonnet, claude-sonnet-4, claude-3-haiku
- **Gemini**: gemini-2.5-pro, gemini-2.5-flash

## Supported Datasets

- **LSAT_AR_Test**: Logical reasoning questions
- **SAT_EN**: English language questions
- **SciQ_Test**: Science questions
- **BoolQ_Valid**: Boolean questions
- **HaluEval_QA**: Hallucination evaluation
- **Life_Eval**: Life evaluation questions

## Error Handling

The script gracefully handles:
- Missing confidence columns (E column is optional)
- Missing token probability data
- Malformed or inconsistent data
- Files with different column structures

## Output Files

### Summary CSV (`analysis_summary.csv`)

Contains one row per model/dataset combination with all computed metrics, rounded to 4 decimal places.

### Plots

High-quality PNG plots (300 DPI) suitable for publication, with clear titles and labels.

## Troubleshooting

### Common Issues

1. **"Item wrong length" errors**: Some CSV files have inconsistent row lengths due to data parsing issues
2. **"The truth value of an array" errors**: Usually indicates malformed data in specific rows
3. **Missing confidence metrics**: Files without A, B, C, D columns won't have confidence-related metrics

### Solutions

- The script automatically skips problematic files and continues processing others
- Check the console output for specific error messages
- Verify CSV file integrity if many files are failing

## Customization

The script is modular and can be easily modified:
- `compute_metrics()`: Add new metrics
- `generate_plots()`: Add new plot types
- `extract_model_and_dataset()`: Add new model/dataset detection logic

## Performance

- Processes ~40 CSV files in a few minutes
- Generates plots efficiently using matplotlib
- Memory usage scales linearly with file sizes
- Handles files with thousands of rows efficiently
