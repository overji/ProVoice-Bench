# ProVoice-Bench

<div align="center">

[![arXiv](https://img.shields.io/badge/v1.0_arXiv-2604.15037-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2604.15037)
[![Hugging Face](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/overji/ProVoice-Bench)

</div>

**ProVoice-Bench** is an evaluation framework for assessing the proactivity of voice agents in multimodal contexts, where agents must decide not only **what to say**, but also **when to intervene**.

Unlike conventional voice-agent benchmarks that focus primarily on reactive responses to explicit user instructions, ProVoice-Bench targets proactive interaction: the agent continuously monitors conversational audio, user-defined triggers, environmental sounds, and digital context such as persona and mobile application states. The benchmark evaluates whether a model can remain silent when no intervention is needed, while proactively responding when assistance, correction, or monitoring is justified.

This project provides:

* A benchmark dataset of **1,182 curated multimodal samples** for proactive voice-agent evaluation.
* Four proactive task settings:

  * **Proactive Intent Capture (PIC):** infer implicit user needs and initiate suitable tool-use requests.
  * **Latent Topic Monitor (LTM):** monitor conversations and intervene only when a user-defined topic appears.
  * **Context Fact Checking (CFC):** detect contradictions between spoken statements and digital context records.
  * **Environment Sound Sensing (ESS):** recognize user-defined acoustic events and provide timely assistance.
* Batch inference scripts for proactive interaction prediction and response/tool-call generation.
* LLM-as-a-Judge evaluation for trigger correctness, tool-call rationality, and response quality.
* Reproducible scoring scripts for proactive decision metrics and response accuracy.

ProVoice-Bench is designed to expose key limitations of current multimodal LLMs in proactive voice interaction, especially over-triggering, weak digital-context reasoning, and the gap between deciding to speak and producing a correct intervention.

## News
+ **(2026-05-20)** ProVoice-Bench has been accepted for publication at **Interspeech 2026**. The benchmark and code are released.

---

## 1. Motivation

Recent LLM-agent research is moving from reactive text assistants to proactive multimodal agents. Existing benchmarks mainly focus on reactive answering and under-evaluate proactive intervention and monitoring behavior.

**ProVoice-Bench** addresses this gap with four proactive tasks and a curated multimodal dataset (1,182 samples, including positive and negative instances), enabling rigorous analysis of:
- When a model should intervene.
- Whether the intervention action/response is actually helpful.

---

## 2. Benchmark Overview

The benchmark covers four proactive scenarios:

1. **Speech Turn-Taking** (`single_turn_taking`)
   - Intervene only when user intent for assistance/tool usage is explicit.

2. **Long-term Turn-Taking** (`long_turn_taking`)
   - Stay silent until a monitored topic is clearly discussed.

3. **Environment Sound Sensing** (`environment_sensing`)
   - Intervene only when critical acoustic events are detected.

4. **Digital Memory Verification** (`digital_memory`)
   - Intervene when user statements clearly contradict digital records.

By design, the default behavior is conservative silence unless trigger conditions are clearly satisfied.

---

## 3. Metrics

### 3.1 Proactive Interaction Prediction

We evaluate whether the model decides to intervene correctly:
- **Accuracy (Acc):** overall correctness of intervention decision.
- **False Positive Rate (FPR):** unnecessary interventions when no valid trigger exists.
- **Recall (Rec):** sensitivity to true intervention triggers.

### 3.2 Response Accuracy ($R_{acc}$)

Beyond trigger detection, we evaluate action/response quality with an LLM judge.

For sample $i$:

$$
\begin{aligned}
S_i = &\frac{\mathcal{J}(\mathcal{S}_{c,i}, T_{p,i}, T_{g,i}) + \mathcal{J}(\mathcal{S}_{c,i}, R_{p,i}, R_{g,i})}{2} \\
&\cdot \mathbb{I}(\text{pred}_i = \text{gt}_i)
\end{aligned}
$$

Where:
- $\mathcal{J}(\cdot) \in \{0, 0.5, 1.0\}$ is the **LLM-as-a-Judge** score (here implemented with a Qwen3-80B endpoint in `eval.py`).
- $T_{p,i}, T_{g,i}$ are predicted and ground-truth tool-call sequences.
- $R_{p,i}, R_{g,i}$ are predicted and ground-truth responses.
- $\mathbb{I}(\cdot)$ is an indicator for correct trigger decision.

Final score:

$$
R_{acc} = \frac{1}{N}\sum_{i=1}^{N} S_i
$$

---

## 4. Repository Structure

```text
update_github/
├── data.json                  # Benchmark data (audio path + metadata + references)
├── new_tools.json             # Tool schema injected into prompts
├── generate_response.py       # Batch generation script
├── eval.py                    # Evaluation + LLM-as-a-Judge scoring
├── requirements.txt           # Python dependencies
├── utils.py                   # Model routing and output directory mapping
└── model_runner/
    ├── BaseModel.py
    ├── Qwen3Omni.py
    ├── Qwen3_80B.py
    └── __init__.py
```

---

## 5. Environment Setup (Python 3.12.11)

> Recommended: create a clean virtual environment with Python **3.12.11**.

### Option A: `venv`

```bash
cd ProVoice-Bench
python3.12 -m venv .venv
source .venv/bin/activate
python --version   # expected: 3.12.11
pip install -U pip
pip install -r requirements.txt
```

### Option B: Conda

```bash
cd ProVoice-Bench
conda create -n provoice-bench python=3.12.11 -y
conda activate provoice-bench
python --version   # expected: 3.12.11
pip install -U pip
pip install -r requirements.txt
```

---

## 6. Model Endpoint Assumptions

Current code expects local OpenAI-compatible chat-completion endpoints:

- Generation model (`utils.get_model("qwen3omni")`):
  - `http://localhost:8947/v1/chat/completions`

- Judge model in `eval.py` (`Qwen3_80BApiInference`):
  - `http://localhost:8947/v1/chat/completions`

If your deployment uses different ports/hosts, update:
- `utils.py`
- `eval.py`

---

## 7. Run Inference

Generate predictions for all unprocessed UUIDs:

```bash
cd ProVoice-Bench
python generate_response.py --model_name qwen3omni
```

Ablation mode (persona context removed from prompts):

```bash
python generate_response.py --model_name qwen3omni --ablation
```

Outputs are saved under the mapped directory from `utils.get_save_dir(...)`, e.g.:
- `./output/qwen3omni_base/`
- `./output/qwen3omni_base_ablation/`

Each prediction file:
- `<uuid>_response.json`

---

## 8. Run Evaluation

Evaluate active triggering and response accuracy:

```bash
cd ProVoice-Bench
python eval.py --model_name qwen3omni
```

Ablation evaluation:

```bash
python eval.py --model_name qwen3omni --ablation
```

Evaluation script behavior:
- Computes proactive trigger metrics (Acc/Rec/FPR and confusion stats by category).
- Calls judge model for response/tool-call quality on eligible samples.
- Saves judge outputs to:
  - `<output_dir>/judge/<uuid>_judge_response.json`

---

## 9. Data and Output Format

### Input (`data.json`)
Each sample contains (at minimum):
- `meta.uuid`
- `meta.task_name`
- `audio_filepath`
- `user_states.datetime_info`
- `user_states.app_states`
- `scenario_cues`
- `conversation.conversation_list` (reference model turns for target tool calls/responses)

### Prediction Output
`generate_response.py` expects model output in strict JSON format:

```json
{
  "interaction_score": 0,
  "tool_calls": [],
  "agent_response": ""
}
```

If parsing fails, the sample is skipped (empty result is not written).

---

## 10. Notes and Troubleshooting

- Ensure audio file paths in `data.json` are valid on your machine.
- `soundfile` may require system audio libraries on some Linux distributions.
- If no files are generated, check endpoint availability and JSON validity of model outputs.
- If evaluation judge files are missing, ensure the judge endpoint is running and reachable.

---

## 11. Citation

If you use this benchmark setting, please cite the corresponding ProVoice-Bench paper and model technical reports referenced in your experiments.
```bibtex
@misc{xu2026reactiveproactiveassessingproactivity,
      title={From Reactive to Proactive: Assessing the Proactivity of Voice Agents via ProVoice-Bench}, 
      author={Ke Xu and Yuhao Wang and Yu Wang},
      year={2026},
      eprint={2604.15037},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.15037}, 
}
```
For questions, please feel free to submit an issue or contact Ke Xu at overji1@sjtu.edu.cn.
