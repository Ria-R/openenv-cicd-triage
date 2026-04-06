from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


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


@dataclass
class Action:
    pass


@dataclass
class Observation:
    pass


@dataclass
class State:
    episode_id: str = ""
    step_count: int = 0


@dataclass
class RootCauseSubmission:
    category: str = ""
    summary: str = ""


@dataclass
class FixSubmission:
    fix_type: str = ""
    summary: str = ""


@dataclass
class CICDTriageAction(Action):
    action_type: ActionType
    stage_name: str | None = None
    query: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    category: str | None = None
    summary: str | None = None
    fix_type: str | None = None


@dataclass
class CICDTriageObservation(Observation):
    task_id: str
    title: str
    difficulty: str
    pipeline_id: str
    status: str
    current_stage: str | None
    visible_summary: dict[str, Any] = field(default_factory=dict)
    visible_logs: dict[str, list[str]] = field(default_factory=dict)
    visible_artifacts: dict[str, Any] = field(default_factory=dict)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    known_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    remaining_steps: int = 0
    last_action_result: str = ""
    last_action_error: str | None = None
    grader_hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class CICDTriageState(State):
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    task_index: int = -1
    max_steps: int = 0
    done: bool = False
    total_reward: float = 0.0
    visible_sections: list[str] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    root_cause_submission: RootCauseSubmission | None = None
    fix_submission: FixSubmission | None = None
    final_score: float | None = None
    last_action_error: str | None = None