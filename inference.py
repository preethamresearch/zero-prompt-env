"""
Inference Script — Zero-Prompt Task Inference Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import sys
import traceback
from typing import List, Optional

from openai import OpenAI

from models import ZeroPromptAction, ZeroPromptObservation
from server.zero_prompt_environment import ZeroPromptEnvironment

# --- Environment Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# --- OpenAI Client ---
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "zero_prompt_env"
MAX_STEPS = 3  # Max attempts per task

# --- Task Definitions ---
TASKS = [
    {"name": "format_inference", "difficulty": "easy"},
    {"name": "intent_classification", "difficulty": "medium"},
    {"name": "multi_step_composition", "difficulty": "hard"},
]


# --- Logging helpers (exact format required) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure action is single-line
    action_clean = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs: ZeroPromptObservation) -> str:
    """Build a prompt for the LLM based on the observation.

    The key design: we give the model the raw input with minimal guidance.
    The model must infer the correct transformation and output format.
    """
    parts = []

    # Minimal guidance — the agent must infer the task from context
    parts.append(
        "You are given raw input data with no instructions. "
        "Infer what transformation or action is needed and produce the output.\n"
        "Output structured data (JSON) when the input is structured. "
        "Output plain text when the input is plain text.\n"
    )

    if obs.examples:
        parts.append("Examples:\n")
        for ex in obs.examples:
            parts.append(f"  Input:  {ex['input']}")
            parts.append(f"  Output: {ex['output']}\n")
        parts.append("Now process this input:")

    formatted = (
        json.dumps(obs.input_data, indent=2)
        if isinstance(obs.input_data, (dict, list))
        else str(obs.input_data)
    )
    parts.append(f"Input:\n{formatted}")

    if obs.feedback:
        parts.append(f"\nYour previous attempt was wrong. Feedback: {obs.feedback}")
        parts.append("Adjust your approach based on this feedback.")

    parts.append(
        "\nRespond with ONLY the output — no explanation, no markdown, "
        "no code fences, no extra text. Just the raw output."
    )
    return "\n".join(parts)


def call_llm(prompt: str) -> str:
    """Call the LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else "error: empty response"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "error: model request failed"


def run_task(task: dict, seed: int = 42) -> tuple:
    """Run a single task episode. Returns (success, steps, score, rewards)."""
    task_name = task["name"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    env = ZeroPromptEnvironment()

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=seed, task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            prompt = build_prompt(obs)
            llm_response = call_llm(prompt)

            action = ZeroPromptAction(response=llm_response)
            obs = env.step(action)

            reward = float(obs.reward or 0.0)
            done = obs.done
            error = obs.feedback if obs.feedback else None

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step, action=llm_response, reward=reward,
                done=done, error=error,
            )

            if done:
                break

        # Score = max reward achieved (best attempt), clamped to [0, 1]
        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.9

    except Exception as exc:
        steps_taken += 1
        rewards.append(0.0)
        log_step(
            step=steps_taken, action="error", reward=0.0,
            done=True, error=str(exc),
        )
        traceback.print_exc(file=sys.stderr)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


def main():
    """Run baseline inference across all tasks."""
    for task in TASKS:
        run_task(task)


if __name__ == "__main__":
    main()
