"""FastAPI application for the Zero-Prompt Task Inference Environment."""

from openenv.core.env_server.http_server import create_app

from .zero_prompt_environment import ZeroPromptEnvironment

try:
    from ..models import ZeroPromptAction, ZeroPromptObservation
except ImportError:
    from models import ZeroPromptAction, ZeroPromptObservation

app = create_app(
    ZeroPromptEnvironment,
    ZeroPromptAction,
    ZeroPromptObservation,
    env_name="zero_prompt_env",
    max_concurrent_envs=10,
)


@app.get("/")
def root():
    return {
        "name": "zero_prompt_env",
        "description": "Zero-Prompt Task Inference Environment - agents must infer tasks with no instructions",
        "tasks": ["format_inference", "intent_classification", "multi_step_composition"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/schema", "/docs"],
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
