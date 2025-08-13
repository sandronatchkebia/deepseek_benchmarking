#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLMC Multi-Dataset Parser
- Handles multiple dataset types: MCQ (A..E), True/False (BoolQ), Numeric Estimation (LifeEval), Open-Ended (HaluEval-QA)
- Accepts DeepSeek/OpenAI-like result structures:
    {
      "question_id": ...,
      "response": "<text that includes a JSON object (possibly fenced with ```json ... ```)>",
      "raw_response": { ... optional, with choices[0].logprobs.content ... }
    }
- Also supports JSONL/JSON arrays where each line/object is that entry.

Outputs:
- *_wide.csv: one row per question with normalized fields and type-specific columns

Notes:
- Token top-logprobs at the answer-letter position are kept ONLY for exploratory work.
- For MCQ, we normalize any present subset of A..E to sum to 1 (after clamping to [0,1]).
- For True/False, we prefer explicit "True" and "False" keys. If missing, we derive the opposite prob as 1 - Confidence.
- For Numeric (LifeEval), we store "estimate" (float) and "conf_within_y" (float). If "y" provided upstream, include it.
- For Open-Ended (HaluEval-QA), we keep "answer" (str) and "stated_confidence" (float if present).
"""

import json
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_entries(path: Path) -> List[Dict[str, Any]]:
    txt = path.read_text(encoding="utf-8")
    t = txt.strip()
    if t.startswith("["):
        return json.loads(t)
    # JSONL fallback
    out = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_last_json_obj(text: str) -> Dict[str, Any]:
    s = strip_code_fence(text or "")
    last_obj = None
    depth = 0
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    last_obj = s[start:i+1]
    if last_obj is None:
        try:
            return json.loads(s)
        except Exception:
            raise ValueError("No JSON object found in response text.")
    return json.loads(last_obj)


def coerce_float(x: Any) -> Optional[float]:
    try:
        f = float(x)
        if math.isfinite(f):
            return f
    except Exception:
        return None
    return None


def normalize_dict_probs(d: Dict[str, Optional[float]]) -> Dict[str, float]:
    keys = list(d.keys())
    vals = []
    for k in keys:
        v = d[k]
        vals.append(0.0 if v is None else max(0.0, min(1.0, float(v))))
    s = sum(vals)
    if s > 0:
        vals = [v / s for v in vals]
    else:
        n = len(vals)
        vals = [1.0 / n] * n if n else []
    return {k: v for k, v in zip(keys, vals)}


def detect_task_type(obj: Dict[str, Any], dataset_hint: str = "") -> str:
    # Use dataset name hints first
    lower = (dataset_hint or "").lower()
    if any(k in lower for k in ["lsat", "sciq", "sat"]):
        # likely MCQ
        return "MCQ"
    if "boolq" in lower:
        return "TRUE_FALSE"
    if "life" in lower:
        return "NUMERIC"
    if "halu" in lower:
        return "OPEN"

    # Fallback by keys present
    option_keys = [k for k in ["A", "B", "C", "D", "E"] if k in obj]
    if option_keys:
        return "MCQ"
    if any(k in obj for k in ["True", "False"]):
        return "TRUE_FALSE"
    # numeric if Answer looks numeric
    ans = obj.get("Answer")
    try:
        float(str(ans))
        return "NUMERIC"
    except Exception:
        pass
    return "OPEN"


def extract_answer_letter_from_text(text: str) -> Optional[str]:
    m = re.search(r'"Answer"\s*:\s*"([A-E])"', text or "")
    return m.group(1) if m else None


def find_answer_token_position(resp_text: str, answer: str) -> Tuple[int, str]:
    """Find the position and token of the answer in the response text"""
    if not answer or not resp_text:
        return -1, ""
    
    # Try to find the answer in the response text
    answer_str = str(answer).strip()
    
    # Look for patterns like "Answer": "B" or "Answer": "True"
    patterns = [
        rf'"Answer"\s*:\s*"({re.escape(answer_str)})"',
        rf'"Answer"\s*:\s*({re.escape(answer_str)})',
        rf'Answer\s*:\s*"({re.escape(answer_str)})"',
        rf'Answer\s*:\s*({re.escape(answer_str)})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, resp_text, re.IGNORECASE)
        if match:
            # Find the position of this match
            pos = match.start()
            return pos, answer_str
    
    # If no pattern match, try simple text search
    pos = resp_text.find(answer_str)
    if pos != -1:
        return pos, answer_str
    
    return -1, answer_str


def token_top5_at_answer_letter(entry: Dict[str, Any]) -> Tuple[List[Optional[str]], List[Optional[float]]]:
    tokens = [None]*5
    probs = [None]*5
    try:
        raw = entry.get("raw_response") or {}
        
        # Handle case where raw_response might be a JSON string
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return tokens, probs
        
        choices = raw.get("choices", [])
        if not choices:
            return tokens, probs
        content = choices[0].get("logprobs", {}).get("content", []) or []
        resp_text = entry.get("response", "") or ""
        letter = extract_answer_letter_from_text(resp_text)
        if not letter:
            return tokens, probs
        
        # Use the GPT parser approach: find exact answer position in text
        pattern = r'"(' + re.escape(letter) + r')"'
        match = re.search(pattern, resp_text)
        if not match:
            return tokens, probs
        
        answer_index = match.start()
        
        # Find the corresponding token position using character counting
        position = 0
        str_char = 0
        while str_char < answer_index and position < len(content):
            token_info = content[position]
            # Use bytes length if available, otherwise token length
            if 'bytes' in token_info and token_info['bytes']:
                str_char += len(token_info['bytes'])
            else:
                str_char += len(token_info.get('token', ''))
            position += 1
        
        # Check if we found the right position
        if position >= len(content):
            return tokens, probs
        
        # Get the top logprobs at the found position
        top = content[position].get("top_logprobs", [])
        for i, t in enumerate(top[:5]):
            tokens[i] = t.get("token")
            lp = t.get("logprob")
            if lp is None:
                p = t.get("prob")
                probs[i] = float(p) if p is not None else None
            else:
                # Maintain precision: don't convert -9999.0 to 0.0
                if lp < -100:  # Very negative logprob (essentially impossible)
                    probs[i] = 0.0
                else:
                    try:
                        probs[i] = float(np.exp(lp))
                    except Exception:
                        probs[i] = None
        return tokens, probs
    except Exception:
        return tokens, probs


def parse_entry(e: Dict[str, Any], dataset_name: str, model_name: str) -> Dict[str, Any]:
    qid = e.get("question_id")
    resp_text = e.get("response", "") or ""
    
    # Store the raw content for the 'content' column
    raw_content = resp_text
    
    # Parse primary JSON payload
    parse_ok = True
    coerced = True
    try:
        obj = extract_last_json_obj(resp_text)
    except Exception:
        parse_ok = False
        coerced = False
        obj = {}
        # try raw_response.message.content
        raw = e.get("raw_response") or {}
        try:
            m = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            if m:
                obj = extract_last_json_obj(m)
                parse_ok = True
                coerced = True
        except Exception:
            pass

    task_type = detect_task_type(obj, dataset_hint=dataset_name)
    reasoning = obj.get("Reasoning")
    answer = obj.get("Answer")
    
    # Find answer token position and token
    token_index, token = find_answer_token_position(resp_text, answer)

    # Exploratory tokens
    tks, tps = token_top5_at_answer_letter(e)

    # Base wide row with GPT-4o compatible columns
    wide_row = {
        "Question ID": qid,
        "content": raw_content,
        "answer": answer,
        "token_index": token_index,
        "token": token,
        "t1": tks[0], "t1_prob": tps[0],
        "t2": tks[1], "t2_prob": tps[1],
        "t3": tks[2], "t3_prob": tps[2],
        "t4": tks[3], "t4_prob": tps[3],
        "t5": tks[4], "t5_prob": tps[4],
        "correct_format": parse_ok,
        "coerce": coerced,
        "Reasoning": reasoning,
        "Answer": answer,
    }

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


def parse_file(infile: Path, dataset_name: str, model_name: str):
    entries = load_entries(infile)
    wide_rows = []
    for e in entries:
        w = parse_entry(e, dataset_name=dataset_name, model_name=model_name)
        wide_rows.append(w)
    wide_df = pd.DataFrame(wide_rows)
    return wide_df


def main():
    import argparse
    p = argparse.ArgumentParser(description="Parse LLMC results across dataset types.")
    p.add_argument("--infile", required=True, help="Path to results JSON/JSONL")
    p.add_argument("--dataset", required=True, help="Dataset name (e.g., SAT-EN, BoolQ, LifeEval, HaluEval-QA)")
    p.add_argument("--model", default="deepseek-chat", help="Model label")
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
