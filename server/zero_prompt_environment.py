"""Zero-Prompt Task Inference Environment.

An OpenEnv environment where the agent receives raw inputs with ZERO
instructions and must infer what task to perform and execute it correctly.

Three difficulty levels:
  - Easy:   Format Inference (data cleaning / transformation)
  - Medium: Intent Classification (email triage / action inference)
  - Hard:   Multi-Step Composition (data pipeline / rule induction)
"""

import time
from typing import Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ZeroPromptAction, ZeroPromptObservation, ZeroPromptState
    from ..tasks import format_inference, intent_classification, multi_step_composition
except ImportError:
    from models import ZeroPromptAction, ZeroPromptObservation, ZeroPromptState
    from tasks import format_inference, intent_classification, multi_step_composition


TASK_MODULES = {
    "format_inference": format_inference,
    "intent_classification": intent_classification,
    "multi_step_composition": multi_step_composition,
}

DIFFICULTY_MAP = {
    "format_inference": "easy",
    "intent_classification": "medium",
    "multi_step_composition": "hard",
}

MAX_ATTEMPTS = {
    "easy": 3,
    "medium": 3,
    "hard": 3,
}

# Penalty weights for undesirable behavior
EMPTY_RESPONSE_PENALTY = -0.5
REPEATED_RESPONSE_PENALTY = -0.3
RETRY_PENALTY_PER_ATTEMPT = 0.05


class ZeroPromptEnvironment(
    Environment[ZeroPromptAction, ZeroPromptObservation, ZeroPromptState]
):
    """Environment for zero-prompt task inference."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._state = ZeroPromptState()
        self._task_module = None
        self._rewards: list[float] = []
        self._examples_cache: list = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: str = "format_inference",
        **kwargs,
    ) -> ZeroPromptObservation:
        """Reset the environment with a new task.

        Args:
            seed: Random seed for reproducible task selection.
            episode_id: Unique episode identifier.
            task_name: One of 'format_inference', 'intent_classification',
                       'multi_step_composition'.
        """
        if task_name not in TASK_MODULES:
            task_name = "format_inference"

        self._task_module = TASK_MODULES[task_name]
        difficulty = DIFFICULTY_MAP[task_name]
        max_att = MAX_ATTEMPTS[difficulty]

        actual_seed = seed if seed is not None else int(time.time() * 1000) % 2**31
        input_data, expected_output, category = self._task_module.pick_task(actual_seed)

        # For hard tasks, input_data is a dict with 'examples' and 'test_input'
        examples = []
        if isinstance(input_data, dict) and "examples" in input_data:
            examples = input_data["examples"]
            display_input = input_data["test_input"]
        else:
            display_input = input_data

        self._examples_cache = examples

        self._state = ZeroPromptState(
            episode_id=episode_id or f"ep_{actual_seed}",
            step_count=0,
            task_id=task_name,
            difficulty=difficulty,
            current_input=display_input,
            expected_output=expected_output,
            attempts_used=0,
            max_attempts=max_att,
            total_reward=0.0,
            history=[],
        )
        self._rewards = []

        return ZeroPromptObservation(
            task_id=task_name,
            difficulty=difficulty,
            input_data=display_input,
            examples=examples,
            feedback="",
            attempts_remaining=max_att,
            done=False,
            reward=None,
        )

    def step(
        self, action: ZeroPromptAction, timeout_s: Optional[float] = None, **kwargs
    ) -> ZeroPromptObservation:
        """Process the agent's response and return graded observation."""
        if self._task_module is None:
            return ZeroPromptObservation(
                task_id="none",
                difficulty="easy",
                input_data=None,
                feedback="Environment not initialized. Call reset() first.",
                done=True,
                reward=0.0,
            )

        self._state.step_count += 1
        self._state.attempts_used += 1

        response_text = action.response.strip()

        # --- Penalize destructive / degenerate behavior ---

        # Empty or whitespace-only response
        if not response_text:
            score = max(0.0, EMPTY_RESPONSE_PENALTY)
            feedback = "Empty response. You must produce an output based on the input."
            return self._finalize_step(score, feedback, response_text)

        # Repeated identical response (agent stuck in a loop)
        prev_responses = [h["response"] for h in self._state.history]
        if response_text[:500] in prev_responses:
            score = max(0.0, REPEATED_RESPONSE_PENALTY)
            feedback = "Repeated response. Try a different approach based on the feedback."
            return self._finalize_step(score, feedback, response_text)

        # --- Grade the response ---
        score, feedback = self._task_module.grade(
            response_text, self._state.expected_output
        )

        # Apply attempt penalty for non-first attempts
        if self._state.attempts_used > 1:
            penalty = RETRY_PENALTY_PER_ATTEMPT * (self._state.attempts_used - 1)
            score = max(0.0, score - penalty)

        return self._finalize_step(score, feedback, response_text)

    def _finalize_step(
        self, score: float, feedback: str, response_text: str
    ) -> ZeroPromptObservation:
        """Record results and build the observation."""
        score = round(score, 2)
        self._rewards.append(score)
        self._state.total_reward += score

        self._state.history.append(
            {
                "attempt": self._state.attempts_used,
                "response": response_text[:500],
                "score": score,
                "feedback": feedback,
            }
        )

        # Episode ends on success or max attempts
        done = (
            score >= 0.9
            or self._state.attempts_used >= self._state.max_attempts
        )

        remaining = self._state.max_attempts - self._state.attempts_used

        return ZeroPromptObservation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            input_data=self._state.current_input,
            examples=self._examples_cache,
            feedback=feedback,
            attempts_remaining=max(0, remaining),
            done=done,
            reward=score,
        )

    @property
    def state(self) -> ZeroPromptState:
        """Return current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up."""
        self._task_module = None
        self._rewards = []
        self._examples_cache = []
