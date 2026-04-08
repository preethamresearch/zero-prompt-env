---
title: Zero-Prompt Task Inference Environment
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
---

# Zero-Prompt Task Inference Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that evaluates an AI agent's ability to **infer and execute tasks with zero instructions**. The agent receives raw input data and must figure out what to do — no prompts, no task descriptions, no guidance.

## Motivation

Current LLM benchmarks provide explicit instructions. But real-world AI deployment often requires models to **understand intent from context alone**. This environment tests a fundamental capability: can an agent look at raw data and determine the correct transformation, classification, or pipeline — without being told what to do?

This directly connects to cutting-edge research in:
- **Zero-shot task inference** (Stanford, MIT)
- **In-context learning and emergent abilities**
- **RL for efficient in-context learning** (GRPO, DPO)

## Environment Overview

The environment presents three tasks of increasing difficulty. In each task, the agent receives raw input and must produce the correct output. The agent gets multiple attempts with feedback, but **never receives explicit instructions**.

### Action Space

```python
class ZeroPromptAction(Action):
    response: str  # The agent's output (format depends on inferred task)
```

### Observation Space

```python
class ZeroPromptObservation(Observation):
    task_id: str              # Task identifier
    difficulty: str           # "easy", "medium", "hard"
    input_data: Any           # Raw input — no instructions provided
    examples: List[Dict]      # Input/output examples (hard tasks only)
    feedback: str             # Feedback from previous attempt
    attempts_remaining: int   # Remaining attempts
    done: bool                # Episode complete?
    reward: float             # Score for this step (0.0–1.0)
```

## Tasks

### Task 1: Format Inference (Easy)

The agent receives raw, unstructured data and must infer the correct output format.

| Input | Expected Output |
|---|---|
| `"john,doe,25"` | `{"first": "john", "last": "doe", "age": 25}` |
| `"2024-13-01"` | `"2024-01-13"` (fix invalid date) |
| `"hello world"` | `"Hello World"` (title case) |
| `"name:John age:25"` | `{"name": "John", "age": "25"}` |

**Categories:** CSV→JSON, fix dates, title case, key-value→JSON, extract emails

### Task 2: Intent Classification (Medium)

The agent receives a raw email/message and must infer the correct action — no instructions about what actions exist or how to classify.

| Input | Expected Output |
|---|---|
| Pricing inquiry email | `{"action": "reply", "priority": "normal", "summary": "..."}` |
| Service outage complaint | `{"action": "escalate", "priority": "urgent", "summary": "..."}` |
| Newsletter notification | `{"action": "archive", "priority": "low", "summary": "..."}` |

**Actions:** reply, forward, archive, flag, escalate, delete

### Task 3: Multi-Step Composition (Hard)

The agent receives input/output examples demonstrating a hidden transformation pipeline, then must apply the inferred rule to a new input.

| Examples | Test Input | Expected |
|---|---|---|
| `"hello world"→"WORLD HELLO"` | `"alpha beta gamma"` | `"GAMMA BETA ALPHA"` |
| `"banana apple cherry"→"apple-banana-cherry"` | `"grape fig elderberry"` | `"elderberry-fig-grape"` |
| `"Hello World Program"→"HWP"` | `"Machine Learning Operations"` | `"MLO"` |

**Categories:** reverse+uppercase, sort+dash-join, acronyms, count-repeat, Caesar cipher, vowel masking, word doubling

## Reward Design

Following the RL == Efficient In-Context Learning philosophy:

- **+1.0** — Correct output (exact match)
- **+0.5–0.9** — Partially correct (right structure, wrong values; case mismatch)
- **+0.0** — Incorrect output
- **Attempt penalty** — Each retry costs -0.1 (encourages getting it right first try)

Rewards are **incremental** — partial credit is given for:
- Correct JSON structure (keys match)
- Correct action type (even if priority/summary wrong)
- Partial token overlap with expected output

## Setup

### Install Dependencies

```bash
pip install openenv-core[core] fastapi uvicorn pydantic requests openai
```

### Run Locally

```bash
# Start the environment server
cd zero-prompt-env
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run inference
export HF_TOKEN="your-token-here"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
python inference.py
```

### Run with Docker

```bash
docker build -t zero-prompt-env .
docker run -p 8000:8000 zero-prompt-env
```

### Validate

```bash
openenv validate
```

## Baseline Performance

Model: **Qwen/Qwen2.5-72B-Instruct** (via HuggingFace Inference API)

| Task | Difficulty | Score | Attempts | Notes |
|---|---|---|---|---|
| Format Inference | Easy | ~0.33 | 3/3 | Gets JSON structure but wrong key names |
| Intent Classification | Medium | ~0.95 | 2/3 | Fails first attempt, self-corrects with feedback |
| Multi-Step Composition | Hard | ~1.00 | 1/3 | In-context examples provide enough signal |

Key observations:
- The "easy" task is hardest without instructions — models don't know what format to produce
- Medium difficulty shows the value of feedback-driven RL — model improves across attempts
- Hard task benefits from in-context examples, demonstrating efficient in-context learning

## Project Structure

```
zero-prompt-env/
├── openenv.yaml                     # OpenEnv metadata
├── models.py                        # Pydantic Action/Observation/State
├── client.py                        # EnvClient implementation
├── inference.py                     # Baseline inference script
├── pyproject.toml                   # Dependencies
├── Dockerfile                       # HF Spaces deployment
├── server/
│   ├── app.py                       # FastAPI app (create_app)
│   └── zero_prompt_environment.py   # Environment implementation
└── tasks/
    ├── format_inference.py          # Easy: format transformation
    ├── intent_classification.py     # Medium: action inference
    └── multi_step_composition.py    # Hard: rule induction
```

## License

MIT
