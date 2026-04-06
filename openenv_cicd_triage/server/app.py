from __future__ import annotations

import os

import uvicorn

try:
    from openenv.core.env_server import create_fastapi_app
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "openenv-core is required to run the server. Install dependencies with `pip install -e .[server]`."
    ) from exc

from ..api_models import (
    APICICDTriageAction,
    APICICDTriageObservation,
    APICICDTriageStepResult,
)
from ..models import CICDTriageAction
from .environment import CICDTriageEnvironment


class ServerCICDTriageEnvironment(CICDTriageEnvironment):
    def reset(self) -> APICICDTriageObservation:
        obs = super().reset()
        return APICICDTriageObservation(
            **obs.__dict__,
            reward=None,
            done=False,
            info={},
        )

    def step(self, action: APICICDTriageAction | dict) -> APICICDTriageStepResult:
        if isinstance(action, APICICDTriageAction):
            internal_action = CICDTriageAction(
                action_type=action.action_type,
                stage_name=action.stage_name,
                query=action.query,
                line_start=action.line_start,
                line_end=action.line_end,
                category=action.category,
                summary=action.summary,
                fix_type=action.fix_type,
            )
        elif isinstance(action, dict):
            internal_action = CICDTriageAction(**action)
        else:
            internal_action = CICDTriageAction(
                action_type=action.action_type,
                stage_name=getattr(action, "stage_name", None),
                query=getattr(action, "query", None),
                line_start=getattr(action, "line_start", None),
                line_end=getattr(action, "line_end", None),
                category=getattr(action, "category", None),
                summary=getattr(action, "summary", None),
                fix_type=getattr(action, "fix_type", None),
            )

        result = super().step(internal_action)

        obs_payload = dict(result["observation"])

        # Remove duplicates if already present
        obs_payload.pop("reward", None)
        obs_payload.pop("done", None)
        obs_payload.pop("info", None)

        observation = APICICDTriageObservation(
            **obs_payload,
            reward=result.get("reward"),
            done=result.get("done", False),
            info=result.get("info", {}),
        )

        return APICICDTriageStepResult(
            observation=observation,
            reward=result.get("reward", 0.0),
            done=result.get("done", False),
            info=result.get("info", {}),
        )


app = create_fastapi_app(
    ServerCICDTriageEnvironment,
    APICICDTriageAction,
    APICICDTriageObservation,
)


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("openenv_cicd_triage.server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()