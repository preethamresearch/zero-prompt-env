"""
Inference Script - Zero-Prompt Task Inference Environment
===================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
OPTIONAL:
    LOCAL_IMAGE_NAME   Docker image name (when using from_docker_image).

STDOUT FORMAT:
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

from client import ZeroPromptClient
from models import ZeroPromptAction, ZeroPromptObservation

# --- Environment Variables ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "zero-prompt-env")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# --- OpenAI Client ---
llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BENCHMARK = "zero_prompt_env"
MAX_STEPS = 3

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


def build_prompt(obs_data: dict) -> str:
    """Build a prompt for the LLM based on the observation data.

    Minimal guidance - the agent must infer the task from context.
    """
    input_data = obs_data.get("input_data")
    examples = obs_data.get("examples", [])
    feedback = obs_data.get("feedback", "")

    parts = []

    parts.append(
        "You are given raw input data with no instructions. "
        "Infer what transformation or action is needed and produce the output.\n"
        "Output structured data (JSON) when the input is structured. "
        "Output plain text when the input is plain text.\n"
    )

    if examples:
        parts.append("Examples:\n")
        for ex in examples:
            parts.append(f"  Input:  {ex['input']}")
            parts.append(f"  Output: {ex['output']}\n")
        parts.append("Now process this input:")

    formatted = (
        json.dumps(input_data, indent=2)
        if isinstance(input_data, (dict, list))
        else str(input_data)
    )
    parts.append(f"Input:\n{formatted}")

    if feedback:
        parts.append(f"\nYour previous attempt was wrong. Feedback: {feedback}")
        parts.append("Adjust your approach based on this feedback.")

    parts.append(
        "\nRespond with ONLY the output - no explanation, no markdown, "
        "no code fences, no extra text. Just the raw output."
    )
    return "\n".join(parts)


def call_llm(prompt: str) -> str:
    """Call the LLM and return the response text."""
    try:
        response = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else "error: empty response"
    except Exception as exc:
        return f"error: {exc}"


async def run_task_docker(task: dict, seed: int = 42) -> tuple:
    """Run a single task against the Docker environment via client."""
    task_name = task["name"]
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        env = await ZeroPromptClient.from_docker_image(LOCAL_IMAGE_NAME)

        result = await env.reset(seed=seed, task_name=task_name)
        obs_data = result.observation.model_dump() if hasattr(result, "observation") else result.model_dump()

        for step in range(1, MAX_STEPS + 1):
            done = obs_data.get("done", False)
            if done:
                break

            prompt = build_prompt(obs_data)
            llm_response = call_llm(prompt)

            action = ZeroPromptAction(response=llm_response)
            result = await env.step(action)

            obs_data = result.observation.model_dump() if hasattr(result, "observation") else result.model_dump()
            reward = float(obs_data.get("reward", 0.0) or 0.0)
            done = obs_data.get("done", False)
            feedback = obs_data.get("feedback", "")
            error = feedback if feedback else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=llm_response, reward=reward, done=done, error=error)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.9

    except Exception as exc:
        steps_taken += 1
        rewards.append(0.0)
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(exc))
        traceback.print_exc(file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


def run_task_local(task: dict, seed: int = 42) -> tuple:
    """Run a single task using the environment directly (no Docker)."""
    from server.zero_prompt_environment import ZeroPromptEnvironment

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

            obs_data = obs.model_dump()
            prompt = build_prompt(obs_data)
            llm_response = call_llm(prompt)

            action = ZeroPromptAction(response=llm_response)
            obs = env.step(action)

            reward = float(obs.reward or 0.0)
            done = obs.done
            error = obs.feedback if obs.feedback else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=llm_response, reward=reward, done=done, error=error)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.9

    except Exception as exc:
        steps_taken += 1
        rewards.append(0.0)
        log_step(step=steps_taken, action="error", reward=0.0, done=True, error=str(exc))
        traceback.print_exc(file=sys.stderr)

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, steps_taken, score, rewards


async def main_docker():
    """Run baseline inference via Docker container."""
    for task in TASKS:
        await run_task_docker(task)


def main_local():
    """Run baseline inference locally (direct import)."""
    for task in TASKS:
        run_task_local(task)


if __name__ == "__main__":
    # Use Docker client if LOCAL_IMAGE_NAME is explicitly set in env,
    # otherwise fall back to direct local execution
    if os.getenv("LOCAL_IMAGE_NAME"):
        asyncio.run(main_docker())
    else:
        main_local()
