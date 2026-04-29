"""
BioMistral vLLM Benchmark — corrected version (script form).

Generated from vllm_benchmark_fixed.ipynb. Run as:
    pip install vllm rouge-score bert-score scikit-learn
    python vllm_benchmark_fixed.py

Notes:
  - Designed for Google Colab; the Drive-mount block is wrapped in try/except
    so it is a no-op outside Colab.
  - Set BASE_DIR below to point to your dataset directory.
  - USE_CHAT_TEMPLATE defaults to False (raw zero-shot prompts).
  - medcalc uses range scoring via the dataset's Lower / Upper Limit.
  - medec accepts 'NA' when error_flag is false.
"""


# ------------------------------------------------------------------------
# # BioMistral vLLM Benchmark — Corrected Version
#
# This notebook fixes the issues identified in the original `vllm.ipynb`:
#
# 1. **Chat template** — wrap prompts in Mistral's `[INST] ... [/INST]` format (BioMistral is a Mistral-Instruct fine-tune).
# 2. **Single results list** — the original used `results_list` for charts but appended to `all_results_list`, so charts were empty.
# 3. **`medec` normalization** — cast non-string fields before concatenating.
# 4. **Task routing** — `medcalc` / `medec` are no longer forced through the MCQ pipeline; they get exact-match scoring on numeric / ID references.
# 5. **File loading** — branches on `.jsonl` vs `.json`; warns on unsupported extensions instead of silently failing.
# 6. **`BERTScorer` cached** at module scope (was re-instantiated per dataset, loading the model into GPU each time).
# 7. **`enforce_eager=False`** — re-enables CUDA graphs for higher throughput.
# 8. **Per-task `max_tokens`** — short for MCQ, longer for generation.
# 9. **`f1_score(... zero_division=0)`** — silences warnings and gives a defined value when extraction returns empty.
# 10. **Cleaner `extract_answer`** — drops the redundant `.upper() == .upper()` check and returns `""` (counted as wrong) instead of guessing a random word when nothing matches.
# 11. **Per-example logging** — saves a CSV of every prompt → prediction → extraction so you can audit failures.
# ------------------------------------------------------------------------


# ------------------------------------------------------------------------
# ## Setup
# ------------------------------------------------------------------------

# !pip install vllm rouge-score bert-score -q

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import json
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
assert torch.cuda.is_available(), "GPU not found. Switch runtime to a GPU (T4, A100, etc.)."

# Mount Drive (Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


# ------------------------------------------------------------------------
# ## 1. Map datasets to tasks and metrics
# ------------------------------------------------------------------------

BASE_DIR = '/content/drive/MyDrive/biomedical_datasets'

# Explicit per-dataset configuration — much safer than substring heuristics.
# task_type drives prompt format; metric drives how predictions are scored.
DATASET_CONFIG = {
    'medqa':              {'task': 'mcq_letter',     'metric': 'classification'},
    'medmcqa':            {'task': 'mcq_letter',     'metric': 'classification'},
    'headqa':             {'task': 'mcq_letter',     'metric': 'classification'},  # options are A-E
    'medbullets':         {'task': 'mcq_letter',     'metric': 'classification'},  # answer_idx is a letter
    'pubmedqa':           {'task': 'yes_no',         'metric': 'classification'},
    'pqa':                {'task': 'yes_no',         'metric': 'classification'},
    'medhallu':           {'task': 'yes_no',         'metric': 'classification'},
    'medcalc':            {'task': 'numeric',        'metric': 'numeric_range'},  # uses Lower/Upper Limit
    'medec':              {'task': 'sentence_id',   'metric': 'exact_match'},
    'aci_bench':          {'task': 'generation',    'metric': 'generation'},
    'aci-bench':          {'task': 'generation',    'metric': 'generation'},
    # Default for free-text generation tasks (summarization etc.)
    '__default__':        {'task': 'generation',     'metric': 'generation'},
}

def detect_dataset_kind(filename):
    name = filename.lower()
    for key in DATASET_CONFIG:
        if key != '__default__' and key in name:
            return key
    return '__default__'

dataset_mapping = {}
assert os.path.exists(BASE_DIR), f"Dataset directory not found: {BASE_DIR}"

for root, _, files in os.walk(BASE_DIR):
    for fname in files:
        if not fname.endswith(('.jsonl', '.json')):
            # CSV / unknown — skip, but warn so we don't fail silently
            if fname.endswith('.csv'):
                print(f"[skip] CSV not supported by this loader: {fname}")
            continue

        # Use validation split for medmcqa (test labels are hidden), test split otherwise
        lower = fname.lower()
        if 'medmcqa' in lower:
            if 'validation' not in lower:
                continue
        elif 'test' not in lower:
            continue

        path = os.path.join(root, fname)
        kind = detect_dataset_kind(fname)
        cfg = DATASET_CONFIG[kind]
        dataset_mapping[path] = {
            'category': os.path.basename(root),
            'kind': kind,
            'task': cfg['task'],
            'metric': cfg['metric'],
        }

print(f"Found {len(dataset_mapping)} dataset files:")
for p, info in dataset_mapping.items():
    print(f"  - {os.path.basename(p):40s} -> kind={info['kind']:12s} metric={info['metric']}")


# ------------------------------------------------------------------------
# ## 2. Evaluation metric functions
# ------------------------------------------------------------------------

from sklearn.metrics import accuracy_score, f1_score
from rouge_score import rouge_scorer
from bert_score import BERTScorer

# Cache heavy objects at module scope so we don't reload the BERT model
# (or rebuild the rouge tokenizer) for every dataset.
_ROUGE = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
# distilbert is fine; for biomedical text consider:
#   model_type='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
_BERT = BERTScorer(model_type='distilbert-base-uncased', lang='en', rescale_with_baseline=False)


def calculate_classification_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    return {'Accuracy': accuracy, 'F1 Score': f1}


def calculate_exact_match(true_labels, predictions):
    correct = sum(
        1 for t, p in zip(true_labels, predictions)
        if str(t).strip().lower() == str(p).strip().lower()
    )
    acc = correct / len(true_labels) if true_labels else 0.0
    return {'Accuracy': acc, 'F1 Score': float('nan')}


def calculate_numeric_range_metrics(true_labels, predictions, bounds_list):
    """Score a numeric prediction as correct iff it falls within [lower, upper].

    Falls back to exact-match (after float-cast) when bounds are missing.
    """
    correct = 0
    total = 0
    for ref, pred, bounds in zip(true_labels, predictions, bounds_list):
        total += 1
        try:
            p = float(str(pred).replace(',', '').strip())
        except (TypeError, ValueError):
            continue
        if bounds is not None:
            lo, hi = bounds
            if lo <= p <= hi:
                correct += 1
        else:
            try:
                if abs(p - float(str(ref).strip())) < 1e-6:
                    correct += 1
            except (TypeError, ValueError):
                pass
    acc = correct / total if total else 0.0
    return {'Accuracy': acc, 'F1 Score': float('nan')}


def calculate_generation_metrics(references, candidates):
    refs = [r if r and r.strip() else "empty reference"  for r in references]
    cands = [c if c and c.strip() else "empty prediction" for c in candidates]

    rouge = [_ROUGE.score(r, c)['rougeL'].fmeasure for r, c in zip(refs, cands)]
    avg_rouge = float(np.mean(rouge)) if rouge else 0.0

    _, _, F1 = _BERT.score(cands, refs)
    avg_bert = F1.mean().item() if len(F1) > 0 else 0.0

    return {'ROUGE': avg_rouge, 'BERTScore': avg_bert}

print("Metric functions ready.")


# ------------------------------------------------------------------------
# ## 3. Prompt formatting (with BioMistral chat template)
# ------------------------------------------------------------------------

# Set to True to wrap prompts in Mistral-Instruct chat template ([INST] ... [/INST]).
# Default: False — raw zero-shot prompts (the convention used in published
# BioMistral evaluations and lm-evaluation-harness). Prompts end with
# "Answer:" so the model continues with the answer.
# Set True to wrap in [INST] ... [/INST] for instruction-mode behavior.
USE_CHAT_TEMPLATE = False


def chat_wrap(user_msg):
    if USE_CHAT_TEMPLATE:
        return f"<s>[INST] {user_msg.strip()} [/INST]"
    return user_msg.strip()


def format_mcq_prompt(example, choice_style='letter'):
    question = example.get('question', '')
    options = example.get('options', {}) or {}
    body = f"Question: {question}\nChoices:\n"
    if isinstance(options, dict):
        for key, value in options.items():
            if value:
                body += f"{key}: {value}\n"
    elif isinstance(options, list):
        for i, opt in enumerate(options):
            label = chr(65 + i) if choice_style == 'letter' else str(i + 1)
            body += f"{label}: {opt}\n"
    if choice_style == 'letter':
        body += "\nAnswer with the letter of the correct choice (A, B, C, ...).\nAnswer:"
    else:
        body += "\nAnswer with the number of the correct choice (1, 2, 3, ...).\nAnswer:"
    return chat_wrap(body)


def format_yes_no_prompt(example):
    ctx = example.get('context', '')
    q = example.get('question', '')
    body = ""
    if ctx:
        body += f"Context: {ctx}\n"
    body += f"Question: {q}\n\nAnswer with yes or no.\nAnswer:"
    return chat_wrap(body)


def format_numeric_prompt(example):
    ctx = example.get('context', '')
    q = example.get('question', '')
    body = ""
    if ctx:
        body += f"Patient Note: {ctx}\n"
    body += f"Calculation Task: {q}\n\nGive only the final numeric answer.\nAnswer:"
    return chat_wrap(body)


def format_sentence_id_prompt(example):
    ctx = example.get('context', '')
    body = (
        "Below is a clinical text with numbered sentences. "
        "Identify the sentence ID that contains an error.\n\n"
        f"{ctx}\n\nGive only the integer sentence ID.\nAnswer:"
    )
    return chat_wrap(body)


def format_generation_prompt(example):
    ctx = example.get('context', '')
    q = example.get('question', '')
    body = ""
    if ctx:
        body += f"Context: {ctx}\n"
    if q:
        body += f"Task: {q}\n"
    body += "\nResponse:"
    return chat_wrap(body)


PROMPT_BUILDERS = {
    'mcq_letter':   lambda ex: format_mcq_prompt(ex, choice_style='letter'),
    'mcq_number':   lambda ex: format_mcq_prompt(ex, choice_style='number'),
    'yes_no':       format_yes_no_prompt,
    'numeric':      format_numeric_prompt,
    'sentence_id':  format_sentence_id_prompt,
    'generation':   format_generation_prompt,
}


# ------------------------------------------------------------------------
# ## 4. Example normalization
# ------------------------------------------------------------------------

def _coerce_str(v):
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return '\n'.join(_coerce_str(x) for x in v)
    if isinstance(v, dict):
        return '\n'.join(f"{k}: {_coerce_str(val)}" for k, val in v.items())
    return str(v)


def _first_nonempty(d, *keys):
    for k in keys:
        if k in d:
            v = d[k]
            if v not in (None, ''):
                return v
    return ''


def _options_from_op_fields(ex):
    out = {}
    for letter, field in zip('ABCDE', ['opa', 'opb', 'opc', 'opd', 'ope']):
        v = ex.get(field)
        if v not in (None, '', 'nan', 'NaN'):
            out[letter] = v
    return out


def _to_float_or_none(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_example(ex, dataset_kind):
    # Optional `bounds` is (lower, upper) for range-scored numeric tasks.
    norm = {'question': '', 'options': {}, 'context': '', 'reference': '', 'bounds': None}

    if dataset_kind == 'headqa':
        data = ex.get('data', {}) or {}
        norm['question']  = data.get('Question', '')
        norm['options']   = {k: v for k, v in (data.get('Options', {}) or {}).items()
                             if v not in (None, '', 'nan', 'NaN')}
        norm['reference'] = str(data.get('Correct Option', '')).strip()

    elif dataset_kind == 'medmcqa':
        norm['question'] = ex.get('question', '')
        norm['options']  = {
            'A': ex.get('opa', ''),
            'B': ex.get('opb', ''),
            'C': ex.get('opc', ''),
            'D': ex.get('opd', ''),
        }
        cop = ex.get('cop', -1)
        norm['reference'] = ['A', 'B', 'C', 'D'][cop] if cop in (0, 1, 2, 3) else ''

    elif dataset_kind == 'medcalc':
        norm['question']  = ex.get('Question', '')
        norm['context']   = ex.get('Patient Note', '')
        norm['reference'] = str(ex.get('Ground Truth Answer', '')).strip()
        # medcalc ships explicit acceptance bounds — use them for range scoring.
        lo = _to_float_or_none(ex.get('Lower Limit'))
        hi = _to_float_or_none(ex.get('Upper Limit'))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            norm['bounds'] = (lo, hi)

    elif dataset_kind == 'medec':
        # `sentences` is pre-numbered as "0 | ...\n1 | ...\n...".
        # When error_flag is False there is no error sentence — accept 'NA'.
        norm['question']  = (
            "Identify the sentence ID containing an error, "
            "or reply 'NA' if the text has no error. Reply with just the integer ID or 'NA'."
        )
        norm['context']   = _coerce_str(ex.get('sentences', ''))
        if ex.get('error_flag') is False:
            norm['reference'] = 'NA'
        else:
            norm['reference'] = str(ex.get('error_sentence_id', '')).strip()

    elif dataset_kind in ('pqa', 'pubmedqa', 'medhallu'):
        norm['question']  = ex.get('question', ex.get('QUESTION', ''))
        norm['context']   = _coerce_str(ex.get('context', ex.get('CONTEXTS', '')))
        ref = ex.get('final_decision', ex.get('answer', ex.get('label', '')))
        norm['reference'] = str(ref).strip().lower()

    elif dataset_kind in ('medbullets', 'medqa'):
        norm['question']  = ex.get('question', '')
        opts = ex.get('options')
        if isinstance(opts, dict) and opts:
            norm['options'] = {k: v for k, v in opts.items()
                               if v not in (None, '', 'nan', 'NaN')}
        else:
            norm['options'] = _options_from_op_fields(ex)
        norm['reference'] = str(ex.get('answer_idx', ex.get('answer', ''))).strip()

    elif dataset_kind == 'aci_bench':
        norm['question']  = "Generate the structured clinical note for the dialogue above."
        norm['context']   = _coerce_str(_first_nonempty(
            ex, 'dialogue', 'src', 'conversation', 'input', 'transcript'
        ))
        norm['reference'] = _coerce_str(_first_nonempty(
            ex, 'note', 'tgt', 'summary', 'reference', 'output', 'gold'
        ))

    else:
        norm['question']  = ex.get('question', ex.get('Question', ''))
        norm['context']   = _coerce_str(ex.get('context', ex.get('Context', '')))
        norm['reference'] = _coerce_str(_first_nonempty(
            ex, 'answer', 'reference', 'summary', 'note', 'tgt', 'output'
        ))

    return norm


# ------------------------------------------------------------------------
# ## 5. Answer extraction
# ------------------------------------------------------------------------

def extract_answer(prediction, reference, task_type):
    """Robust answer extraction.

    Strict-first-token match, then a lenient fallback that searches for the
    reference token anywhere in the first 80 chars (matches how
    lm-evaluation-harness grades MCQ tasks).
    """
    p = str(prediction).strip()
    r = str(reference).strip()
    if not r:
        return ""

    if task_type in ('mcq_letter', 'mcq_number'):
        head = p[:80]

        # 1. Strict: first non-trivial token after optional preamble.
        #    For single-letter answers, require an exact case match so
        #    "a patient presented..." doesn't get scored as choice A.
        m = re.search(
            r'^\s*(?:Answer\s*[:.\-]?\s*|Option\s+|Choice\s+|The\s+(?:correct\s+)?answer\s+is\s+)?'
            r'\(?\*?\*?([A-Za-z0-9]+)\*?\*?\)?',
            head,
        )
        if m:
            tok = m.group(1).strip()
            if task_type == 'mcq_letter' and len(r) == 1:
                if tok == r.upper():
                    return r.upper()
            elif task_type == 'mcq_letter':
                if tok.upper() == r.upper():
                    return r.upper()
            elif task_type == 'mcq_number':
                if tok == r:
                    return r

        # 2. Lenient: reference appears as a standalone token in head.
        if task_type == 'mcq_letter' and len(r) == 1:
            # Case-sensitive boundary search so 'a' (article) doesn't match 'A'
            if re.search(rf'(?<![A-Za-z]){re.escape(r.upper())}(?![A-Za-z])', head):
                return r.upper()
            # Common patterns: "(A)", "A:", "A." or after non-letter
            if re.search(rf'(?:^|[^A-Za-z])\(?{re.escape(r.upper())}\)?[\s:.,)]', head):
                return r.upper()
        else:
            if re.search(rf'\b{re.escape(r)}\b', head, re.IGNORECASE):
                return r

        return ""

    if task_type == 'yes_no':
        m = re.search(r'\b(yes|no)\b', p[:80], re.IGNORECASE)
        return m.group(1).lower() if m else ""

    if task_type == 'numeric':
        m = re.search(r'-?\d+(?:\.\d+)?', p[:120])
        return m.group(0) if m else ""

    if task_type == 'sentence_id':
        # 'NA' / 'no error' answer — accept any case-insensitive variant
        if re.search(r'\bna\b|\bno error\b|\bnone\b', p[:120], re.IGNORECASE):
            return 'NA'
        m = re.search(r'-?\d+', p[:120])
        return m.group(0) if m else ""

    return p


# ------------------------------------------------------------------------
# ## 6. Initialize vLLM
# ------------------------------------------------------------------------

from vllm import LLM, SamplingParams

MODEL_NAME = "BioMistral/BioMistral-7B"
print(f"Loading {MODEL_NAME} ...")

llm = LLM(
    model=MODEL_NAME,
    dtype="auto",
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
    enforce_eager=False,   # CUDA graphs ON for speed
)
print("Model loaded.")

# Per-task sampling params. MCQ tokens are higher than you might think
# because BioMistral often emits a short preamble ("The answer is X") even
# when asked for a single letter. 32 tokens is a safe budget.
SAMPLING_PARAMS = {
    'mcq_letter':  SamplingParams(temperature=0.0, max_tokens=32),
    'mcq_number':  SamplingParams(temperature=0.0, max_tokens=32),
    'yes_no':      SamplingParams(temperature=0.0, max_tokens=16),
    'numeric':     SamplingParams(temperature=0.0, max_tokens=64),
    'sentence_id': SamplingParams(temperature=0.0, max_tokens=16),
    'generation':  SamplingParams(temperature=0.0, max_tokens=256),
}


# ------------------------------------------------------------------------
# ## 7. Per-dataset evaluation
# ------------------------------------------------------------------------

def load_dataset(file_path):
    rows = []
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            rows = data if isinstance(data, list) else [data]
    return rows


def evaluate_dataset(file_path, info, model, per_example_log):
    name = os.path.basename(file_path)
    task = info['task']
    metric = info['metric']
    print(f"\n--- {name}  (task={task}, metric={metric}) ---")

    try:
        rows = load_dataset(file_path)
    except Exception as e:
        print(f"  [error] failed to load: {e}")
        return None
    if not rows:
        print("  [warn] empty file")
        return None

    builder = PROMPT_BUILDERS[task]
    prompts, references, bounds_list = [], [], []
    for ex in rows:
        norm = normalize_example(ex, info['kind'])
        prompts.append(builder(norm))
        references.append(norm['reference'])
        bounds_list.append(norm.get('bounds'))

    outputs = model.generate(prompts, SAMPLING_PARAMS[task])
    raw_preds = [o.outputs[0].text.strip() for o in outputs]

    if metric == 'classification':
        extracted = [extract_answer(p, r, task) for p, r in zip(raw_preds, references)]
        scores = calculate_classification_metrics(references, extracted)
    elif metric == 'exact_match':
        extracted = [extract_answer(p, r, task) for p, r in zip(raw_preds, references)]
        scores = calculate_exact_match(references, extracted)
    elif metric == 'numeric_range':
        extracted = [extract_answer(p, r, task) for p, r in zip(raw_preds, references)]
        scores = calculate_numeric_range_metrics(references, extracted, bounds_list)
    else:  # generation
        extracted = raw_preds
        scores = calculate_generation_metrics(references, extracted)

    for prompt, ref, raw, ext, b in zip(prompts, references, raw_preds, extracted, bounds_list):
        per_example_log.append({
            'dataset': name,
            'task': task,
            'reference': ref,
            'bounds': b,
            'raw_prediction': raw[:200],
            'extracted': ext,
        })

    print(f"  scores: {scores}")
    return {
        'Dataset': name,
        'Category': info['category'],
        'Task': task,
        **scores,
    }


# ------------------------------------------------------------------------
# ## 8. Run the benchmark
# ------------------------------------------------------------------------

results_list = []
per_example_log = []

for path, info in tqdm(dataset_mapping.items(), desc="Datasets"):
    res = evaluate_dataset(path, info, llm, per_example_log)
    if res:
        results_list.append(res)

results_df = pd.DataFrame(results_list)
if not results_df.empty:
    numeric_cols = results_df.select_dtypes(include=['float64', 'float32']).columns
    results_df[numeric_cols] = results_df[numeric_cols].round(4)

per_example_df = pd.DataFrame(per_example_log)

# Persist artifacts so failures can be audited later
results_df.to_csv('/content/results_summary.csv', index=False)
per_example_df.to_csv('/content/results_per_example.csv', index=False)

print("\n### Summary ###")
from pprint import pprint as display
display(results_df)


# ------------------------------------------------------------------------
# ## 9. Visualize
# ------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

classification_df = results_df.dropna(subset=['Accuracy']).copy() if 'Accuracy' in results_df.columns else pd.DataFrame()
generation_df     = results_df.dropna(subset=['ROUGE']).copy()    if 'ROUGE'    in results_df.columns else pd.DataFrame()

if not classification_df.empty:
    plt.figure(figsize=(14, 6))
    melted = classification_df.melt(
        id_vars=['Dataset', 'Category'],
        value_vars=[c for c in ['Accuracy', 'F1 Score'] if c in classification_df.columns],
        var_name='Metric', value_name='Score',
    ).dropna(subset=['Score'])
    sns.barplot(data=melted, x='Dataset', y='Score', hue='Metric', palette='viridis')
    plt.title('Classification & Exact-Match Tasks')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()

if not generation_df.empty:
    plt.figure(figsize=(14, 6))
    melted = generation_df.melt(
        id_vars=['Dataset', 'Category'],
        value_vars=[c for c in ['ROUGE', 'BERTScore'] if c in generation_df.columns],
        var_name='Metric', value_name='Score',
    ).dropna(subset=['Score'])
    sns.barplot(data=melted, x='Dataset', y='Score', hue='Metric', palette='magma')
    plt.title('Generation Tasks (ROUGE / BERTScore)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------------
# ## 10. Audit failures
#
# If a dataset's score looks wrong, look at `results_per_example.csv` (or `per_example_df` below).
# The most common causes are:
#
# - The dataset has a different field name than `normalize_example` expects → add a branch.
# - The model's first token isn't the choice letter (e.g. it says "The answer is..." in a way the regex misses) → tweak `extract_answer`.
# - References include explanations along with the letter → strip them in `normalize_example`.
# ------------------------------------------------------------------------

# Quick look at where extraction returned empty (i.e. counted as wrong)
if not per_example_df.empty:
    misses = per_example_df[per_example_df['extracted'] == '']
    print(f"Empty-extraction rows: {len(misses)} / {len(per_example_df)}")
    display(misses.head(20))


# ------------------------------------------------------------------------
# ## 11. (Optional) Chat-template ablation
#
# Re-run only the affected datasets with the chat template flipped, then compare side-by-side.
# ------------------------------------------------------------------------

ABLATION_DATASETS = ['headqa', 'medbullets']  # add more as needed

ablation_paths = {
    p: info for p, info in dataset_mapping.items()
    if any(name in os.path.basename(p).lower() for name in ABLATION_DATASETS)
}

ablation_results = []
for use_template in (False, True):
    globals()['USE_CHAT_TEMPLATE'] = use_template
    label = 'with_template' if use_template else 'no_template'
    print(f"\n===== {label} =====")
    for path, info in ablation_paths.items():
        log = []
        res = evaluate_dataset(path, info, llm, log)
        if res:
            res['Variant'] = label
            ablation_results.append(res)

ablation_df = pd.DataFrame(ablation_results)
display(ablation_df.pivot_table(index='Dataset', columns='Variant', values='Accuracy'))
