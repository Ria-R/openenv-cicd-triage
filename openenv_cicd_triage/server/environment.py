from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any
from dataclasses import asdict, is_dataclass

try:
    from openenv.core.env_server import Environment
except Exception:  # pragma: no cover
    class Environment:  # type: ignore
        pass

from ..config import load_grading, load_tasks
from ..grader import EpisodeGrader
from ..api_models import APICICDTriageAction
from ..models import (
    CICDTriageAction,
    CICDTriageObservation,
    CICDTriageState,
    FixSubmission,
    RootCauseSubmission,
)

class CICDTriageEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = load_tasks()
        self.grading = load_grading()
        self._state = CICDTriageState(
            episode_id="",
            step_count=0,
            task_id="",
            task_index=-1,
            max_steps=0,
            done=False,
            total_reward=0.0,
            visible_sections=[],
            action_history=[],
            root_cause_submission=None,
            fix_submission=None,
            final_score=None,
            last_action_error=None,
        )
        self._task_index = -1
        self._task: dict[str, Any] | None = None
        self._grader: EpisodeGrader | None = None
        self._visible_logs: dict[str, list[str]] = {}
        self._visible_artifacts: dict[str, Any] = {}
        self._last_action_result = ""

    def reset(self) -> CICDTriageObservation:
        self._task_index = (self._task_index + 1) % len(self.tasks)
        self._task = self.tasks[self._task_index]
        self._grader = EpisodeGrader(self._task)
        episode_id = str(uuid.uuid4())

        self._state = CICDTriageState(
        episode_id=episode_id,
        step_count=0,
        task_id=self._task["id"],
        task_index=self._task_index,
        max_steps=int(self._task["max_steps"]),
        done=False,
        total_reward=0.0,
        visible_sections=["pipeline_summary"],
        action_history=[],
        root_cause_submission=None,
        fix_submission=None,
        final_score=None,
        last_action_error=None,
    )
        self._visible_logs = {}
        self._visible_artifacts = {}
        self._last_action_result = "Environment reset. Pipeline summary is available."
        return self._build_observation(last_error=None)

    @staticmethod
    def _serialize_observation(obs: Any) -> dict[str, Any]:
        if is_dataclass(obs):
            return asdict(obs)
        if hasattr(obs, "model_dump"):
            return obs.model_dump()
        if hasattr(obs, "__dict__"):
            return dict(obs.__dict__)
        raise TypeError(f"Unsupported observation type for serialization: {type(obs)!r}")

    def step(self, action: CICDTriageAction | APICICDTriageAction | dict[str, Any]) -> dict[str, Any]:
        if self._task is None or self._grader is None:
            obs = self.reset()
            return {
                "observation": self._serialize_observation(obs),
                "reward": 0.0,
                "done": False,
                "info": {"auto_reset": True},
            }

        if self._state.done:
            obs = self._build_observation(last_error="Episode already finished. Call reset() for a new task.")
            return {
                "observation": self._serialize_observation(obs),
                "reward": 0.0,
                "done": True,
                "info": {"reason": "already_done"},
            }
        self._state.step_count += 1
        reward = 0.0
        last_error = None
        info: dict[str, Any] = {"task_id": self._task["id"]}

        try:
            reward, info = self._apply_action(action)
        except ValueError as exc:
            reward -= float(self.grading["penalties"]["malformed_action"])
            last_error = str(exc)
            self._last_action_result = "Action rejected."
            info = {"task_id": self._task["id"], "error_type": "validation"}

        self._state.total_reward += reward
        self._state.last_action_error = last_error

        if self._state.step_count >= self._state.max_steps and not self._state.done:
            self._state.done = True
            info["max_steps_reached"] = True
            grading = self._grader.grade(self._state)
            self._state.final_score = grading["final_score"]
            info["grader"] = grading

        obs = self._build_observation(last_error=last_error)
        return {
            "observation": asdict(obs),
            "reward": round(max(0.0, min(1.0, reward)), 4),
            "done": self._state.done,
            "info": info,
        }

    @property
    def state(self) -> CICDTriageState:
        return self._state

    def _apply_action(self, action: CICDTriageAction) -> tuple[float, dict[str, Any]]:
        assert self._task is not None and self._grader is not None
        reward = 0.0
        penalties = self.grading["penalties"]
        reward_cfg = self.grading["reward_weights"]
        info: dict[str, Any] = {"task_id": self._task["id"]}
        repeated = False

        action_key = self._action_key(action)
        seen_before = {entry["action_key"] for entry in self._state.action_history}
        if action_key in seen_before:
            repeated = True
            reward -= float(penalties["repeated_action"])

        if action.action_type == "view_pipeline_summary":
            self._last_action_result = "Pipeline summary already visible."

        elif action.action_type == "view_stage_log":
            if not action.stage_name:
                raise ValueError("stage_name is required for view_stage_log")
            logs = self._task["data"]["stage_logs"].get(action.stage_name)
            if logs is None:
                raise ValueError(f"Unknown stage log: {action.stage_name}")
            self._visible_logs[action.stage_name] = logs
            self._last_action_result = f"Opened log for stage '{action.stage_name}'."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "view_log_window":
            if not action.stage_name:
                raise ValueError("stage_name is required for view_log_window")
            if action.stage_name not in self._task["data"]["stage_logs"]:
                raise ValueError(f"Unknown stage log: {action.stage_name}")
            if action.line_start is None or action.line_end is None:
                raise ValueError("line_start and line_end are required for view_log_window")
            all_lines = self._task["data"]["stage_logs"][action.stage_name]
            max_lines = int(self.grading["limits"]["max_log_window_lines"])
            start = max(0, action.line_start)
            end = min(len(all_lines), action.line_end)
            if end <= start:
                raise ValueError("line_end must be greater than line_start")
            if (end - start) > max_lines:
                end = start + max_lines
            self._visible_logs[f"{action.stage_name}:{start}-{end}"] = all_lines[start:end]
            self._last_action_result = f"Opened log window {start}:{end} for stage '{action.stage_name}'."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "search_logs":
            query = (action.query or "").strip().lower()
            if not query:
                raise ValueError("query is required for search_logs")
            results: list[str] = []
            for stage, lines in self._task["data"]["stage_logs"].items():
                for idx, line in enumerate(lines):
                    if query in line.lower():
                        results.append(f"{stage}[{idx}]: {line}")
            capped = results[: int(self.grading["limits"]["max_search_results"])]
            self._visible_artifacts.setdefault("search_results", {})[query] = capped
            if capped:
                self._last_action_result = f"Found {len(capped)} matching log lines for '{query}'."
                reward += self._evidence_reward(action_key)
            else:
                self._last_action_result = f"No log lines matched '{query}'."
                reward -= float(penalties["irrelevant_search"])

        elif action.action_type == "view_changed_files":
            self._visible_artifacts["changed_files"] = self._task["summary"]["changed_files"]
            if "changed_files_detail" in self._task["data"]:
                self._visible_artifacts["changed_files_detail"] = self._task["data"]["changed_files_detail"]
            self._last_action_result = "Changed files are now visible."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "view_dependency_file":
            if action.query:
                file_name = action.query
            else:
                deps = self._task["data"].get("dependency_files", {})
                if len(deps) == 1:
                    file_name = next(iter(deps))
                else:
                    raise ValueError("query should contain the dependency file name")
            deps = self._task["data"].get("dependency_files", {})
            if file_name not in deps:
                raise ValueError(f"Unknown dependency file: {file_name}")
            self._visible_artifacts.setdefault("dependency_files", {})[file_name] = deps[file_name]
            self._last_action_result = f"Dependency file '{file_name}' is now visible."
            reward += self._evidence_reward(f"view_dependency_file:{file_name}")
            action_key = f"view_dependency_file:{file_name}"

        elif action.action_type == "view_test_report":
            self._visible_artifacts["test_report"] = self._task["data"]["test_report"]
            self._last_action_result = "Test report is now visible."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "view_env_snapshot":
            self._visible_artifacts["env_snapshot"] = self._task["data"]["env_snapshot"]
            self._last_action_result = "Environment snapshot is now visible."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "view_recent_run_diff":
            self._visible_artifacts["recent_run_diff"] = self._task["data"]["recent_run_diff"]
            self._last_action_result = "Recent run diff is now visible."
            reward += self._evidence_reward(action_key)

        elif action.action_type == "submit_root_cause":
            if not action.category and not action.summary:
                raise ValueError("category or summary must be provided for submit_root_cause")
            self._state.root_cause_submission = RootCauseSubmission(
                category=action.category or "",
                summary=action.summary or "",
            )
            root_grade = self._grader.grade(self._state)["root_cause_score"]
            if root_grade >= 0.999:
                reward += float(reward_cfg["correct_root_cause_category"]) + float(reward_cfg["correct_root_cause_detail"])
                self._last_action_result = "Root cause submission matches ground truth."
            elif root_grade > 0:
                reward += root_grade * (
                    float(reward_cfg["correct_root_cause_category"]) + float(reward_cfg["correct_root_cause_detail"])
                )
                self._last_action_result = "Root cause submission has partial credit."
            else:
                reward -= float(penalties["incorrect_root_cause"])
                self._last_action_result = "Root cause submission is incorrect."
            info["root_cause_score"] = root_grade

        elif action.action_type == "submit_fix":
            if not action.fix_type and not action.summary:
                raise ValueError("fix_type or summary must be provided for submit_fix")
            self._state.fix_submission = FixSubmission(
                fix_type=action.fix_type or "",
                summary=action.summary or "",
            )
            fix_grade = self._grader.grade(self._state)["fix_score"]
            if fix_grade >= 0.999:
                reward += float(reward_cfg["correct_fix_type"]) + float(reward_cfg["correct_fix_detail"])
                self._last_action_result = "Fix submission matches ground truth."
            elif fix_grade > 0:
                reward += fix_grade * (float(reward_cfg["correct_fix_type"]) + float(reward_cfg["correct_fix_detail"]))
                self._last_action_result = "Fix submission has partial credit."
            else:
                reward -= float(penalties["incorrect_fix"])
                self._last_action_result = "Fix submission is incorrect."
            info["fix_score"] = fix_grade

        elif action.action_type == "resolve_episode":
            self._state.done = True
            grading = self._grader.grade(self._state)
            self._state.final_score = grading["final_score"]
            evidence_fraction = grading["evidence_score"]
            min_fraction = float(self.grading["limits"]["min_resolution_evidence_fraction"])
            if evidence_fraction < min_fraction:
                reward -= float(penalties["premature_resolution"])
                self._last_action_result = "Episode resolved with limited evidence."
            else:
                reward += float(reward_cfg["successful_resolution_bonus"]) * grading["final_score"]
                self._last_action_result = "Episode resolved."
            info["grader"] = grading

        else:
            raise ValueError(f"Unsupported action_type: {action.action_type}")

        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "action_type": action.action_type,
                "action_key": action_key,
                "payload": asdict(action),
                "repeated": repeated,
                "result": self._last_action_result,
            }
        )
        return reward, info

    def _build_observation(self, last_error: str | None) -> CICDTriageObservation:
        assert self._task is not None

        known_hypotheses = []
        if self._state.root_cause_submission is not None:
            known_hypotheses.append(
                {
                    "category": self._state.root_cause_submission.category,
                    "summary": self._state.root_cause_submission.summary,
                }
            )
        if self._state.fix_submission is not None:
            known_hypotheses.append(
                {
                    "fix_type": self._state.fix_submission.fix_type,
                    "summary": self._state.fix_submission.summary,
                }
            )

        return CICDTriageObservation(
            task_id=self._task["id"],
            title=self._task["title"],
            difficulty=self._task["difficulty"],
            pipeline_id=self._task["pipeline_id"],
            status="resolved" if self._state.done else "failed",
            current_stage=self._task["summary"].get("failed_stage"),
            visible_summary=self._task["summary"],
            visible_logs=self._visible_logs,
            visible_artifacts=self._visible_artifacts,
            action_history=self._state.action_history,
            known_hypotheses=known_hypotheses,
            remaining_steps=max(0, self._state.max_steps - self._state.step_count),
            last_action_result=self._last_action_result,
            last_action_error=last_error,
            grader_hints={
                "available_root_cause_categories": [
                    "dependency_mismatch",
                    "missing_env_var",
                    "config_syntax_error",
                    "test_regression",
                    "network_transient",
                    "migration_failure",
                    "artifact_missing",
                    "permission_error",
                ],
                "available_fix_types": [
                    "pin_dependency",
                    "restore_env_var",
                    "correct_config",
                    "rerun_transient",
                    "update_migration_image",
                    "fix_permissions",
                    "restore_artifact_path",
                    "update_code",
                ],
            },
        )

    def _evidence_reward(self, action_key: str) -> float:
        assert self._task is not None
        reward_cfg = self.grading["reward_weights"]
        evidence_cfg = self._task["truth"]["evidence"]
        if action_key in evidence_cfg.get("required_actions", []):
            return float(reward_cfg["reveal_required_evidence"])
        if action_key in evidence_cfg.get("supporting_actions", []):
            return float(reward_cfg["reveal_supporting_evidence"])
        return 0.0

    @staticmethod
    def _action_key(action: CICDTriageAction) -> str:
        if action.action_type == "view_stage_log" and action.stage_name:
            return f"view_stage_log:{action.stage_name}"
        if action.action_type == "view_dependency_file" and (action.query or ""):
            return f"view_dependency_file:{action.query}"
        return action.action_type

    def _coerce_action(self, action: CICDTriageAction | APICICDTriageAction) -> CICDTriageAction:
        if isinstance(action, CICDTriageAction):
            return action
        if isinstance(action, APICICDTriageAction):
            return CICDTriageAction(
                action_type=action.action_type,
                stage_name=action.stage_name,
                query=action.query,
                line_start=action.line_start,
                line_end=action.line_end,
                category=action.category,
                summary=action.summary,
                fix_type=action.fix_type,
            )
        if hasattr(action, "model_dump"):
            data = action.model_dump()
            return CICDTriageAction(**data)
        if isinstance(action, dict):
            return CICDTriageAction(**action)
        raise ValueError(f"Unsupported action object: {type(action)!r}")

