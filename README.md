# Measure Before You Deploy: Demonstration Sensitivity and Output Stability of Small Language Models in Clinical Text Generation

Code and evaluation notebooks for the paper *"Measure Before You Deploy: Demonstration Sensitivity and Output Stability of Small Language Models in Clinical Text Generation"* (hazalturkmen91, August 2026).

## Abstract

Small language models (SLMs, under 10B parameters) proposed for on-premise clinical deployment are typically judged by a single accuracy number computed once, under one demonstration configuration — despite known sensitivity of in-context learning to demonstration choice, and despite clinical NLP's reliance on free-text generation, where there is no discrete answer to read that sensitivity off from. We evaluate nine open biomedical SLMs (7–8B) on four clinical generation tasks under zero-shot, few-shot, and dynamic few-shot prompting, measuring both reference-based quality shift and output-vs-output NLI consistency.

Three findings emerge. Demonstrations help selectively, raising BERTScore F1 by up to +0.079 on note-generation tasks but at most +0.010 on dialogue and medication QA. These gains sit on unstable ground: composite consistency averages 0.324 (0.283 under retrieval), 15–17% of output pairs are flagged contradictory, and retrieval destabilizes outputs about as much as it helps quality. Consistency is also an unreliable proxy for quality — it tracks quality for six of nine models on note-generation tasks, inverts for three, and leaves roughly a quarter of instances both self-consistent and wrong, invisible to any label-free monitor.

Accuracy measured once, offline, is therefore not deployment readiness: stability under a model's actual demonstration configuration should be measured before deployment and, being reference-free, monitored after.

## Contributions

- **An evaluation of demonstration sensitivity of SLMs on clinical generation tasks.** Zero-shot, few-shot, and dynamic few-shot prompting are contrasted across nine open biomedical SLMs (7B–8B) and four clinical generation tasks, treating demonstration choice as a perturbation whose induced variance is itself the measurement, rather than a knob tuned for the best score.
- **A stability measure for free-text generation.** Agreement between a model's own outputs is scored with bidirectional NLI, computed per instance, with the same metric on both perturbation axes.
- **A label-free monitoring signal.** Needs only the outputs — no references, logits, or judge model — so it can run on live production traffic. Its limit: consistent-but-wrong outputs escape any label-free monitor, so periodic labeled audits remain necessary.

## Tasks

| Task | Input | Output | n |
|---|---|---|---|
| MedDialog | Patient–doctor conversation | One-sentence summary | 500 |
| MedicationQA | Consumer medication question | Free-text answer | 500 |
| MTSamples | Patient information | Treatment plan | 500 |
| MTSamples-Proc | Procedure record | Procedure note | 500 |

MedDialog and MedicationQA produce short outputs; MTSamples and MTSamples-Proc produce long structured text. The pairing is deliberate — output length governs both how much a demonstration can influence generation and how far two generations can drift apart, so a stability result that held only for short answers would not transfer to note drafting, where clinical deployment is concentrated.

## Models

Nine open biomedical SLMs, 7–8B parameters, openly licensed and locally deployable:

| Model | Base | Params |
|---|---|---|
| Llama3-Med42-8B | Llama-3 | 8B |
| Llama3-OpenBioLLM-8B | Llama-3 | 8B |
| MMed-Llama-3-8B | Llama-3 | 8B |
| Apollo-7B | Qwen | 7B |
| BioMistral-7B | Mistral-7B | 7B |
| AlpaCare-Llama2-7B | Llama-2 | 7B |
| Asclepius-7B | Llama | 7B |
| MedAlpaca-7B | Llama | 7B |
| Meditron-7B | Llama-2 | 7B |

The roster mixes instruction-tuned models (Med42, OpenBioLLM, AlpaCare, Asclepius, MedAlpaca) with domain-pretrained checkpoints that received no instruction tuning (MMed-Llama-3, Meditron) — the contrast itself is a finding: the checkpoints without instruction tuning are where output stability collapses.

## Prompting strategies & perturbation axes

- **Zero-shot** — no demonstrations.
- **Static few-shot (k=5)** — demonstrations drawn from a fixed pool under 5 random seeds (**random-draw axis**), yielding $\binom{5}{2}=10$ output pairs per instance.
- **Dynamic few-shot (k=5)** — demonstrations retrieved per-query by 3 different retrievers (**retriever axis**: S-PubMedBERT, BGE, TF-IDF), yielding 3 output pairs per instance.

All three strategies share one prompt scaffold and decoding configuration, so any difference between them traces to the demonstrations alone.

## Measures

- **Demonstration sensitivity (reference-based quality shift)**: BERTScore F1 and ROUGE-L, comparing each generation against the gold reference across the three prompting strategies.
- **Output stability (label-free)**: bidirectional NLI consistency via `DeBERTa-v3-base-mnli-fever-anli`, scoring every pair of outputs the same model produces for the same input under different demonstrations. Composite score:
  $s = \tfrac{1}{2}(\bar e_{\text{fwd}}+\bar e_{\text{bwd}}) - \max(\bar c_{\text{fwd}}, \bar c_{\text{bwd}})$,
  computed per instance, same metric on both perturbation axes so random-draw and retriever results are directly comparable.

## Pipeline / notebook order

1. **`demo_sensitivity_runner.ipynb`** — generation, random-draw axis (zero-shot + 5 seeded k=5 few-shot draws per task).
2. **`healthslm_eval_dynamic_fewshot.ipynb`** — generation + scoring, retriever axis (one run per retriever).
3. **`s1_reference_quality.ipynb`** — quality scoring (ROUGE-L, BERTScore F1) for the random-draw axis.
4. **`s1_dynamic_bertscore.ipynb`** — quality scoring (BERTScore F1) for the retriever axis.
5. **`s3_nli_consistency.ipynb`** — NLI-based output consistency (S3) for the random-draw axis.
6. **`s3_dynamic_nli_consistency.ipynb`** — NLI-based output consistency (S3) for the retriever axis.

`s2_bert_consistency.ipynb` is **dropped/unused** — an earlier consistency-scoring approach superseded by the NLI-based S3 method.

## Setup

No `requirements.txt`; dependencies are listed in `setup_a100.sh` / `setup_h100.sh` (vLLM 0.6.6, PyTorch 2.5.1, transformers 4.46.3). Generation was run on A100/H100 GPU instances.
