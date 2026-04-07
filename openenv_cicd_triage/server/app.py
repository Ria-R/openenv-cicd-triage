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
)
from ..models import CICDTriageAction
from .environment import CICDTriageEnvironment


class ServerCICDTriageEnvironment(CICDTriageEnvironment):
    """Wrapper that makes the environment compatible with the openenv-core framework.

    The framework's HTTPEnvServer WS handler expects:
    - reset() -> Observation subclass (APICICDTriageObservation)
    - step(action) -> Observation subclass (APICICDTriageObservation)

    The Observation must have `done`, `reward`, `metadata` fields
    (inherited from openenv.core.env_server.types.Observation).
    """

    def reset(self, **kwargs) -> APICICDTriageObservation:
        obs = super().reset()
        return APICICDTriageObservation(
            task_id=obs.task_id,
            title=obs.title,
            difficulty=obs.difficulty,
            pipeline_id=obs.pipeline_id,
            status=obs.status,
            current_stage=obs.current_stage,
            visible_summary=obs.visible_summary,
            visible_logs=obs.visible_logs,
            visible_artifacts=obs.visible_artifacts,
            action_history=obs.action_history,
            known_hypotheses=obs.known_hypotheses,
            remaining_steps=obs.remaining_steps,
            last_action_result=obs.last_action_result,
            last_action_error=obs.last_action_error,
            grader_hints=obs.grader_hints,
            reward=None,
            done=False,
            info={},
        )

    def step(self, action, **kwargs) -> APICICDTriageObservation:
        """Execute a step and return an APICICDTriageObservation.

        The framework's WS handler calls serialize_observation() on whatever
        step() returns, so this MUST return an Observation subclass — not a dict.
        """
        # Convert the framework's APICICDTriageAction to internal CICDTriageAction
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

        # CICDTriageEnvironment.step() returns a dict
        result = CICDTriageEnvironment.step(self, internal_action)

        obs_payload = result["observation"]

        # Build and return an Observation subclass (not a dict)
        return APICICDTriageObservation(
            task_id=obs_payload.get("task_id", ""),
            title=obs_payload.get("title", ""),
            difficulty=obs_payload.get("difficulty", ""),
            pipeline_id=obs_payload.get("pipeline_id", ""),
            status=obs_payload.get("status", ""),
            current_stage=obs_payload.get("current_stage"),
            visible_summary=obs_payload.get("visible_summary", {}),
            visible_logs=obs_payload.get("visible_logs", {}),
            visible_artifacts=obs_payload.get("visible_artifacts", {}),
            action_history=obs_payload.get("action_history", []),
            known_hypotheses=obs_payload.get("known_hypotheses", []),
            remaining_steps=obs_payload.get("remaining_steps", 0),
            last_action_result=obs_payload.get("last_action_result", ""),
            last_action_error=obs_payload.get("last_action_error"),
            grader_hints=obs_payload.get("grader_hints", {}),
            reward=result.get("reward"),
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
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    uvicorn.run("openenv_cicd_triage.server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    main()