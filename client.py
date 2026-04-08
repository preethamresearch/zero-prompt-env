"""Client for the Zero-Prompt Task Inference Environment."""

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

try:
    from .models import ZeroPromptAction, ZeroPromptObservation, ZeroPromptState
except ImportError:
    from models import ZeroPromptAction, ZeroPromptObservation, ZeroPromptState


class ZeroPromptClient(EnvClient[ZeroPromptAction, ZeroPromptObservation, ZeroPromptState]):
    """Client to interact with the Zero-Prompt environment server."""

    def _step_payload(self, action: ZeroPromptAction) -> dict:
        """Convert action to JSON payload for the server."""
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult[ZeroPromptObservation]:
        """Parse server response into a StepResult."""
        obs = ZeroPromptObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: dict) -> ZeroPromptState:
        """Parse server state response."""
        return ZeroPromptState(**payload)
