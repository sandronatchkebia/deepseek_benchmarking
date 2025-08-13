#!/usr/bin/env python3
"""
Parse DeepSeek R1 model outputs without logprobs.
Focuses on extracting reasoning, answers, and confidence scores.
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


def extract_answer_letter_from_text(text: str) -> Optional[str]:
    """Extract answer letter (A, B, C, D, E) from text."""
    # Look for patterns like "Answer: A", "Answer: B", etc.
    patterns = [
        r'Answer["\s]*:["\s]*([A-E])',
        r'answer["\s]*:["\s]*([A-E])',
        r'["\s]([A-E])["\s]*$',
        r'["\s]([A-E])["\s]*[.,]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return None


def coerce_float(value) -> Optional[float]:
    """Coerce value to float, return None if impossible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def normalize_dict_probs(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalize probabilities to sum to 1.0."""
    total = sum(probs.values())
    if total > 0:
        return {k: v / total for k, v in probs.items()}
    return probs


def detect_dataset_type(dataset_name: str) -> str:
    """Detect dataset type based on name."""
    dataset_name = dataset_name.upper()
    
    if any(name in dataset_name for name in ["SAT", "LSAT", "SCI"]):
        return "MCQ"
    elif "BOOL" in dataset_name:
        return "TRUE_FALSE"
    elif "LIFE" in dataset_name:
        return "NUMERIC"
    else:
        return "OPEN"


def parse_entry(entry: Dict[str, Any], dataset_name: str, model_name: str) -> Dict[str, Any]:
    """Parse a single entry without logprobs."""
    qid = entry.get("question_id", entry.get("id", "unknown"))
    resp_text = entry.get("response", "") or ""
    raw_content = entry.get("raw_response", {})
    
    # Try to parse the response as JSON
    obj = {}
    parse_ok = False
    coerced = False
    
    try:
        # First try direct parsing
        obj = json.loads(resp_text)
        parse_ok = True
        coerced = False
    except json.JSONDecodeError:
        # Try to fix common formatting issues
        try:
            # Look for JSON content between braces
            start = resp_text.find('{')
            end = resp_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = resp_text[start:end + 1]
                obj = json.loads(json_str)
                parse_ok = False
                coerced = True
        except json.JSONDecodeError:
            obj = {}
            parse_ok = False
            coerced = False
    
    # Extract basic information
    answer = obj.get("Answer", obj.get("answer", None))
    reasoning = obj.get("Reasoning", obj.get("reasoning", None))
    
    # If no structured answer found, try to extract from text
    if not answer:
        answer = extract_answer_letter_from_text(resp_text)
    
    # Determine dataset type
    task_type = detect_dataset_type(dataset_name)
    
    # Create base row
    wide_row = {
        "Question ID": qid,
        "content": resp_text,
        "answer": answer,
        "correct_format": parse_ok,
        "coerce": coerced,
        "Reasoning": reasoning,
        "Answer": answer,
    }
    
    # Add dataset-specific columns
    if task_type == "MCQ":
        if isinstance(answer, str):
            answer = answer.strip().upper()
            wide_row["Answer"] = answer
        options = [k for k in ["A", "B", "C", "D", "E"] if k in obj]
        probs = {k: coerce_float(obj.get(k)) for k in options}
        probs_norm = normalize_dict_probs(probs) if options else {}
        # write A, B, C, D, E columns (GPT-4o format)
        for k in ["A", "B", "C", "D", "E"]:
            wide_row[k] = probs_norm.get(k, 0.0)

    elif task_type == "TRUE_FALSE":
        # prefer explicit True/False keys
        p_true = obj.get("True")
        p_false = obj.get("False")
        conf = obj.get("Confidence")
        # normalize / derive
        p_true_f = coerce_float(p_true)
        p_false_f = coerce_float(p_false)
        conf_f = coerce_float(conf)
        # If only one side present, derive the other
        if p_true_f is None and p_false_f is None and conf_f is not None and isinstance(answer, str):
            # answer is "True"/"False" string or boolean
            ans_bool = str(answer).strip().lower() == "true"
            if ans_bool:
                p_true_f, p_false_f = conf_f, 1.0 - conf_f
            else:
                p_true_f, p_false_f = 1.0 - conf_f, conf_f
        # Add Confidence column (GPT-4o format)
        wide_row["Confidence"] = conf_f

    elif task_type == "NUMERIC":
        # LifeEval: numeric answer, confidence = prob within ±y; y may be in entry or prompt metadata
        est = coerce_float(answer)
        conf_within_y = coerce_float(obj.get("Confidence"))
        y = obj.get("y") or obj.get("Y") or None  # if you encode y into JSON upstream
        y_val = coerce_float(y)
        wide_row["estimate"] = est
        wide_row["conf_within_y"] = conf_within_y
        wide_row["y"] = y_val
        # Add Confidence column (GPT-4o format)
        wide_row["Confidence"] = conf_within_y

    else:  # OPEN
        # HaluEval-QA: free-text answer + confidence if present
        conf = coerce_float(obj.get("Confidence"))
        wide_row["Confidence"] = conf

    return wide_row


def load_entries(infile: Path) -> List[Dict[str, Any]]:
    """Load entries from JSON or JSONL file."""
    if infile.suffix == '.jsonl':
        entries = []
        with open(infile, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
    else:
        with open(infile, 'r') as f:
            return json.load(f)


def parse_file(infile: Path, dataset_name: str, model_name: str):
    """Parse file and return wide format DataFrame."""
    entries = load_entries(infile)
    wide_rows = []
    for e in entries:
        w = parse_entry(e, dataset_name=dataset_name, model_name=model_name)
        wide_rows.append(w)
    wide_df = pd.DataFrame(wide_rows)
    return wide_df


def main():
    import argparse
    p = argparse.ArgumentParser(description="Parse DeepSeek R1 results without logprobs.")
    p.add_argument("--infile", required=True, help="Path to results JSON/JSONL")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., SAT-EN, BoolQ, LifeEval, HaluEval-QA)")
    p.add_argument("--model", default="deepseek-r1", help="Model label")
    p.add_argument("--output_dir", help="Output directory for CSV files (default: same as input file)")
    args = p.parse_args()

    in_path = Path(args.infile)
    wide_df = parse_file(in_path, dataset_name=args.dataset, model_name=args.model)

    # Determine output path
    if args.output_dir:
        # Use specified output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        filename = in_path.stem
        wide_path = output_dir / f"{filename}_wide.csv"
    else:
        # Use same directory as input file (original behavior)
        base = in_path.with_suffix("")
        wide_path = Path(str(base) + "_wide.csv")

    wide_df.to_csv(wide_path, index=False)

    print(f"Wrote: {wide_path}")


if __name__ == "__main__":
    main()
