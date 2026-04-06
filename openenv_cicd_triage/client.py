from __future__ import annotations

from dataclasses import asdict
from typing import Any, Generic, TypeVar

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.env_client import StepResult
except Exception:  # pragma: no cover
    TAction = TypeVar("TAction")
    TObs = TypeVar("TObs")
    TState = TypeVar("TState")

    class StepResult(Generic[TObs]):  # type: ignore
        def __init__(self, observation: Any, reward: float | None = None, done: bool = False):
            self.observation = observation
            self.reward = reward
            self.done = done
            self.info = {}

    class EnvClient(Generic[TAction, TObs, TState]):  # type: ignore
        @classmethod
        async def from_docker_image(cls, image_name: str):
            raise NotImplementedError("openenv-core is not installed, so docker-backed env client is unavailable.")

        @classmethod
        async def from_base_url(cls, base_url: str):
            raise NotImplementedError("openenv-core is not installed, so HTTP env client is unavailable.")

        async def close(self) -> None:
            return None

from .models import CICDTriageAction, CICDTriageObservation, CICDTriageState


class CICDTriageEnv(EnvClient[CICDTriageAction, CICDTriageObservation, CICDTriageState]):
    def _step_payload(self, action: CICDTriageAction) -> dict[str, Any]:
        return {
            "action_type": action.action_type,
            "stage_name": action.stage_name,
            "query": action.query,
            "line_start": action.line_start,
            "line_end": action.line_end,
            "category": action.category,
            "summary": action.summary,
            "fix_type": action.fix_type,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CICDTriageObservation]:
        observation_payload = payload.get("observation", payload)

        if isinstance(observation_payload, dict) and "observation" in observation_payload and "task_id" not in observation_payload:
            nested = observation_payload.get("observation")
            if isinstance(nested, dict):
                observation_payload = nested

        filtered_observation_payload = {
            "task_id": observation_payload.get("task_id", ""),
            "title": observation_payload.get("title", ""),
            "difficulty": observation_payload.get("difficulty", ""),
            "pipeline_id": observation_payload.get("pipeline_id", ""),
            "status": observation_payload.get("status", ""),
            "current_stage": observation_payload.get("current_stage"),
            "visible_summary": observation_payload.get("visible_summary", {}),
            "visible_logs": observation_payload.get("visible_logs", {}),
            "visible_artifacts": observation_payload.get("visible_artifacts", {}),
            "action_history": observation_payload.get("action_history", []),
            "known_hypotheses": observation_payload.get("known_hypotheses", []),
            "remaining_steps": observation_payload.get("remaining_steps", 0),
            "last_action_result": observation_payload.get("last_action_result", ""),
            "last_action_error": observation_payload.get("last_action_error"),
            "grader_hints": observation_payload.get("grader_hints", {}),
        }

        observation = CICDTriageObservation(**filtered_observation_payload)

        result = StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

        try:
            result.info = payload.get("info", {})
        except Exception:
            pass

        return result

    def _parse_state(self, payload: dict[str, Any]) -> CICDTriageState:
        return CICDTriageState(**payload)

    def _action_to_dict(self, action: CICDTriageAction) -> dict[str, Any]:
        return asdict(action)