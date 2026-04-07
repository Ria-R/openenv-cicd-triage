from __future__ import annotations

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server.types import (
        Action as OpenEnvAction,
        Observation as OpenEnvObservation,
    )
except Exception:  # pragma: no cover
    # Fallback if openenv-core is not installed
    OpenEnvAction = BaseModel  # type: ignore[misc, assignment]
    OpenEnvObservation = BaseModel  # type: ignore[misc, assignment]


ActionType = Literal[
    "view_pipeline_summary",
    "view_stage_log",
    "view_log_window",
    "search_logs",
    "view_changed_files",
    "view_dependency_file",
    "view_test_report",
    "view_env_snapshot",
    "view_recent_run_diff",
    "submit_root_cause",
    "submit_fix",
    "resolve_episode",
]


class APICICDTriageAction(OpenEnvAction):
    model_config = ConfigDict(extra="forbid")
    action_type: ActionType
    stage_name: str | None = None
    query: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    category: str | None = None
    summary: str | None = None
    fix_type: str | None = None


class APICICDTriageObservation(OpenEnvObservation):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    difficulty: str
    pipeline_id: str
    status: str
    current_stage: str | None = None
    visible_summary: dict[str, Any] = Field(default_factory=dict)
    visible_logs: dict[str, list[str]] = Field(default_factory=dict)
    visible_artifacts: dict[str, Any] = Field(default_factory=dict)
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    known_hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    remaining_steps: int = 0
    last_action_result: str = ""
    last_action_error: str | None = None
    grader_hints: dict[str, Any] = Field(default_factory=dict)

    # info is NOT part of the OpenEnv Observation base, so we add it here
    info: dict[str, Any] = Field(default_factory=dict)


class APICICDTriageStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: APICICDTriageObservation
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)