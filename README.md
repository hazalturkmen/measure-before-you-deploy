# Evaluation of HealthSLM

Evaluation and demonstration-sensitivity analysis of biomedical small language models (SLMs) on clinical free-text generation tasks.

## What this measures

How much few-shot **demonstration selection** affects the consistency and quality of biomedical SLM outputs — comparing two axes of variation:

- **Random-draw axis**: 5 independently seeded random draws of k=5 few-shot demonstrations, from a frozen pool of 30 candidates per task.
- **Retriever axis**: demonstrations selected by 3 different retrievers (S-PubMedBERT, BGE, TF-IDF) from a pool of 100 candidates per task.

Output **consistency** is measured via pairwise NLI (entailment/contradiction) between outputs generated under different draws/retrievers for the same input. Output **quality** is measured via ROUGE-L and BERTScore against gold references.

## Tasks

| Task | Input | Output |
|---|---|---|
| MedDialog | Patient–doctor conversation | One-sentence summary |
| MedicationQA | Consumer medication question | Free-text answer |
| MTSamples | Patient information | Treatment plan |
| MTSamples-Proc | Procedure record | Procedure note |

(ACI-Bench also appears in early generation runs but is not part of the core consistency analysis.)

## Models evaluated

Llama3-Med42-8B, Llama3-OpenBioLLM-8B, MMed-Llama-3-8B, Apollo-7B, BioMistral-7B, AlpaCare-Llama2-7B, Asclepius-7B, MedAlpaca-7B, Meditron-7B.

## Pipeline / notebook order

1. **`demo_sensitivity_runner.ipynb`** — generation only, random-draw axis (zero-shot + 5 seeded k=5 few-shot draws per task).
2. **`healthslm_eval_dynamic_fewshot.ipynb`** — generation + scoring, retriever axis (one run per retriever).
3. **`s1_reference_quality.ipynb`** — quality scoring (ROUGE-L, BERTScore F1) for the random-draw axis.
4. **`s1_dynamic_bertscore.ipynb`** — quality scoring (BERTScore F1) for the retriever axis.
5. **`s3_nli_consistency.ipynb`** — NLI-based output consistency (S3) for the random-draw axis.
6. **`s3_dynamic_nli_consistency.ipynb`** — NLI-based output consistency (S3) for the retriever axis.

`s2_bert_consistency.ipynb` is **dropped/unused** — an earlier consistency-scoring approach superseded by the NLI-based S3 method.

## Metrics

- **Quality**: ROUGE-L (`rouge_score`), BERTScore F1 (`bert_score`, distilbert-base-uncased backbone).
- **Consistency (S3)**: bidirectional NLI via `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, composite score
  $s = \tfrac{1}{2}(\bar e_{\text{fwd}}+\bar e_{\text{bwd}}) - \max(\bar c_{\text{fwd}}, \bar c_{\text{bwd}})$,
  averaged over all pairwise output comparisons per instance (10 pairs for the random-draw axis, 3 pairs for the retriever axis).

## Setup

No `requirements.txt`; dependencies are listed in `setup_a100.sh` / `setup_h100.sh` (vLLM 0.6.6, PyTorch 2.5.1, transformers 4.46.3). Generation was run on A100/H100 GPU instances.

## Status

Generation and quality/consistency scoring pipelines are implemented for all task/model/axis combinations. Statistical analysis and paper tables/figures are in progress.
