"""Pydantic models for the Zero-Prompt Task Inference Environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class ZeroPromptAction(Action):
    """Action: the agent's response/output for the current task."""

    response: str = Field(
        ...,
        description="The agent's output — format depends on the task the agent infers.",
    )


class ZeroPromptObservation(Observation):
    """Observation returned after each step or on reset."""

    task_id: str = Field(default="", description="Identifier for the current task.")
    difficulty: str = Field(
        default="easy", description="Difficulty level: easy, medium, hard."
    )
    input_data: Any = Field(
        default=None,
        description="The raw input data. No instructions are provided — the agent must infer the task.",
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Optional input/output examples (used in harder tasks).",
    )
    feedback: str = Field(
        default="",
        description="Feedback from the previous step (hints at what went wrong).",
    )
    attempts_remaining: int = Field(
        default=3, description="Number of attempts the agent has left."
    )


class ZeroPromptState(State):
    """Internal environment state."""

    task_id: str = Field(default="")
    difficulty: str = Field(default="easy")
    current_input: Any = Field(default=None)
    expected_output: Any = Field(default=None)
    attempts_used: int = Field(default=0)
    max_attempts: int = Field(default=3)
    total_reward: float = Field(default=0.0)
    history: List[Dict[str, Any]] = Field(default_factory=list)
