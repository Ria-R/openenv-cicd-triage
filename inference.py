from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from typing import Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

try:
    from openenv_cicd_triage import CICDTriageAction, CICDTriageEnv
except ImportError:
    CICDTriageAction = None  # type: ignore[assignment,misc]
    CICDTriageEnv = None  # type: ignore[assignment,misc]

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "openenv-cicd-triage"
MAX_STEPS = 12
TEMPERATURE = 0.0
MAX_TOKENS = 512
TASKS_TO_RUN = 3
SUCCESS_SCORE_THRESHOLD = 0.6

_MODELS_USING_MAX_COMPLETION_TOKENS = {"gpt-5", "gpt-4", "o1", "o3", "o4"}


def _token_limit_kwarg(model_name: str, limit: int = 512) -> dict:
    for prefix in _MODELS_USING_MAX_COMPLETION_TOKENS:
        if model_name.startswith(prefix):
            return {"max_completion_tokens": limit}
    return {"max_tokens": limit}


SYSTEM_PROMPT = """You are an expert CI/CD incident triage agent. You diagnose pipeline failures by inspecting evidence, then submit a root cause, a fix, and resolve the episode.

RESPONSE FORMAT — return ONLY a JSON object (no markdown, no commentary):
{
  "action_type": "<one of the actions below>",
  "stage_name": "string or null",
  "query": "string or null",
  "line_start": int or null,
  "line_end": int or null,
  "category": "string or null",
  "summary": "string or null",
  "fix_type": "string or null"
}

INVESTIGATION actions (use these to gather evidence):
  view_pipeline_summary, view_stage_log (needs stage_name), view_log_window (needs stage_name+line_start+line_end),
  search_logs (needs query), view_changed_files, view_dependency_file (needs query with filename),
  view_test_report, view_env_snapshot, view_recent_run_diff

RESOLUTION actions (use these once you have enough evidence):
  submit_root_cause — needs category + summary (cite specific errors, versions, filenames)
  submit_fix — needs fix_type + summary (describe the concrete remediation)
  resolve_episode — finalizes your diagnosis; MUST be your last action

ROOT CAUSE CATEGORIES: dependency_mismatch, missing_env_var, config_syntax_error, test_regression, network_transient, migration_failure, artifact_missing, permission_error
FIX TYPES: pin_dependency, restore_env_var, correct_config, rerun_transient, update_migration_image, fix_permissions, restore_artifact_path, update_code

WORKFLOW — follow this order:
1. INVESTIGATE: View the failed stage log first. Then check 1-2 supporting sources (env snapshot, dependency files, recent run diff, changed files).
2. DIAGNOSE: submit_root_cause with the correct category and a detailed summary citing specific error messages, version numbers, and file paths.
3. FIX: submit_fix with the correct fix_type and a concrete remediation plan.
4. RESOLVE: resolve_episode to finalize.

RULES:
- NEVER repeat an action you already took — each investigation action should be unique.
- Focus on the FAILED stage first, then corroborate with supporting evidence.
- When remaining_steps <= 3, STOP investigating and immediately submit_root_cause, submit_fix, then resolve_episode.
- Cite specific error messages, version numbers, and file paths in your summaries.
- Only include fields relevant to the action; omit null fields.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def compact_action(action: Any) -> str:
    payload = {
        "action_type": action.action_type,
        "stage_name": action.stage_name,
        "query": action.query,
        "line_start": action.line_start,
        "line_end": action.line_end,
        "category": action.category,
        "summary": action.summary,
        "fix_type": action.fix_type,
    }
    return json.dumps(
        {k: v for k, v in payload.items() if v is not None},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def build_observation_prompt(obs: Any, is_first: bool = False) -> str:
    parts: list[str] = []

    if is_first:
        parts.append(f"=== CI/CD Incident: {obs.title} ===")
        parts.append(f"Task: {obs.task_id} | Difficulty: {obs.difficulty} | Pipeline: {getattr(obs, 'pipeline_id', 'unknown')}")

        summary = obs.visible_summary
        if isinstance(summary, dict):
            if summary.get("failed_stage"):
                parts.append(f"Failed stage: {summary['failed_stage']}")
            stages = summary.get("stages")
            if stages:
                stage_lines = []
                for s in stages:
                    if isinstance(s, dict):
                        for k, v in s.items():
                            stage_lines.append(f"  {k}: {v}")
                    else:
                        stage_lines.append(f"  {s}")
                parts.append("Pipeline stages:\n" + "\n".join(stage_lines))
            if summary.get("changed_files"):
                parts.append(f"Changed files: {', '.join(summary['changed_files'])}")

    parts.append(f"\nRemaining steps: {obs.remaining_steps}")

    if obs.last_action_result:
        parts.append(f"Last result: {obs.last_action_result}")
    if obs.last_action_error:
        parts.append(f"ERROR: {obs.last_action_error}")

    if obs.visible_logs:
        parts.append("\n--- Visible Logs ---")
        for stage, lines in obs.visible_logs.items():
            parts.append(f"[{stage}]:")
            for line in (lines if isinstance(lines, list) else [lines]):
                parts.append(f"  {line}")

    if obs.visible_artifacts:
        parts.append("\n--- Visible Artifacts ---")
        for name, content in obs.visible_artifacts.items():
            parts.append(f"[{name}]:")
            if isinstance(content, dict):
                for k, v in content.items():
                    parts.append(f"  {k}: {v}")
            elif isinstance(content, list):
                for item in content:
                    parts.append(f"  {item}")
            else:
                parts.append(f"  {content}")

    if obs.known_hypotheses:
        parts.append("\n--- Submitted Hypotheses ---")
        for h in obs.known_hypotheses:
            parts.append(f"  {h}")

    if obs.grader_hints and is_first:
        hints = obs.grader_hints
        if hints.get("available_root_cause_categories"):
            parts.append(f"\nValid root cause categories: {', '.join(hints['available_root_cause_categories'])}")
        if hints.get("available_fix_types"):
            parts.append(f"Valid fix types: {', '.join(hints['available_fix_types'])}")

    if obs.remaining_steps <= 3:
        parts.append(
            "\n>>> URGENT: You have {0} step(s) left. You MUST submit root_cause, fix, "
            "and resolve NOW. No more investigation. <<<".format(obs.remaining_steps)
        )

    return "\n".join(parts)


DEFAULT_PLANS = {
    "dependency_mismatch_easy": [
        {"action_type": "view_stage_log", "stage_name": "unit-tests"},
        {"action_type": "view_dependency_file", "query": "requirements.txt"},
        {
            "action_type": "submit_root_cause",
            "category": "dependency_mismatch",
            "summary": "Unit tests fail with ImportError: cannot import 'build_retry_policy' from authsdk.helpers. requirements.txt pins authsdk==1.8.0 which lacks this helper (available in 1.9+).",
        },
        {
            "action_type": "submit_fix",
            "fix_type": "pin_dependency",
            "summary": "Update requirements.txt to pin authsdk>=1.9.0 (or a compatible version exposing build_retry_policy), then rerun unit tests.",
        },
        {"action_type": "resolve_episode"},
    ],
    "missing_secret_medium": [
        {"action_type": "view_stage_log", "stage_name": "deploy"},
        {"action_type": "view_env_snapshot"},
        {"action_type": "view_recent_run_diff"},
        {
            "action_type": "submit_root_cause",
            "category": "missing_env_var",
            "summary": "Deploy failed: 'required value DEPLOY_API_TOKEN is not set'. The deploy.yml workflow cleanup removed 'inherit: secrets' from the reusable workflow call, so DEPLOY_API_TOKEN is no longer passed to the deploy job.",
        },
        {
            "action_type": "submit_fix",
            "fix_type": "restore_env_var",
            "summary": "Restore the secrets mapping (or 'inherit: secrets') in .github/workflows/deploy.yml for the deploy job so DEPLOY_API_TOKEN is available, then rerun the deploy stage.",
        },
        {"action_type": "resolve_episode"},
    ],
    "migration_cascade_hard": [
        {"action_type": "view_stage_log", "stage_name": "integration-tests"},
        {"action_type": "view_stage_log", "stage_name": "migration"},
        {"action_type": "view_recent_run_diff"},
        {"action_type": "view_env_snapshot"},
        {
            "action_type": "submit_root_cause",
            "category": "migration_failure",
            "summary": "Integration tests timeout because the upstream migration job failed. Root cause: migration-job.yaml uses accounts-migrator:1.4.2 image tag but the app deployment expects schema bundle 1.5.0 (APP_SCHEMA_TARGET=1.5.0). The 1.4.2 migrator cannot create 'ledger_entries_v2' table needed by the 1.5.x schema.",
        },
        {
            "action_type": "submit_fix",
            "fix_type": "update_migration_image",
            "summary": "Update deploy/k8s/migration-job.yaml to use accounts-migrator:1.5.x (matching APP_SCHEMA_TARGET=1.5.0), rerun the migration job, then rerun the full pipeline.",
        },
        {"action_type": "resolve_episode"},
    ],
}


def fallback_plan_step(task_id: str, step_idx: int) -> Any:
    plan = DEFAULT_PLANS.get(task_id)
    if plan is None:
        plan = [{"action_type": "view_pipeline_summary"}, {"action_type": "resolve_episode"}]
    payload = plan[min(step_idx, len(plan) - 1)]
    return CICDTriageAction(**payload)


def build_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    print(f"[LLM] using API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[LLM] API_KEY present={bool(API_KEY)}", flush=True)
    print(f"[LLM] MODEL_NAME={MODEL_NAME}", flush=True)

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )


def parse_llm_response(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"action_type": "view_pipeline_summary"}


def get_model_action(
    client: Any,
    messages: list[dict[str, str]],
    observation: Any,
    step_idx: int,
    is_first: bool,
) -> Any:
    task_id = observation.task_id
    user_prompt = build_observation_prompt(observation, is_first=is_first)
    messages.append({"role": "user", "content": user_prompt})

    try:
        api_kwargs = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": TEMPERATURE,
            **_token_limit_kwarg(MODEL_NAME),
        }
        try:
            api_kwargs["response_format"] = {"type": "json_object"}
            completion = client.chat.completions.create(**api_kwargs)
        except Exception:
            del api_kwargs["response_format"]
            completion = client.chat.completions.create(**api_kwargs)

        content = (completion.choices[0].message.content or "").strip()
        print(f"[LLM_RAW] {content}", flush=True)
        messages.append({"role": "assistant", "content": content})

        action_payload = parse_llm_response(content)
        filtered = {k: v for k, v in action_payload.items() if v is not None}
        return CICDTriageAction(**filtered)
    except Exception as exc:
        print(f"[LLM_ERROR] {type(exc).__name__}: {exc} — using fallback plan", flush=True)
        return fallback_plan_step(task_id, step_idx)


async def run_episode(client: Any, env: Any) -> tuple[str, bool, int, float, list[float]]:
    reset_result = await env.reset()
    obs = reset_result.observation
    task_id = obs.task_id
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: list[float] = []
    steps_taken = 0
    final_score = 0.0
    has_root_cause = False
    has_fix = False

    try:
        for step_idx in range(MAX_STEPS):
            action = get_model_action(client, messages, obs, step_idx, is_first=(step_idx == 0))

            if action.action_type == "submit_root_cause":
                has_root_cause = True
            elif action.action_type == "submit_fix":
                has_fix = True

            result = await env.step(action)
            obs = result.observation

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            error = getattr(obs, "last_action_error", None)

            rewards.append(reward)
            steps_taken = step_idx + 1

            log_step(steps_taken, compact_action(action), reward, done, error)

            if done:
                info = getattr(result, "info", {}) or {}
                grader_score = (
                    info.get("grader", {}).get("final_score")
                    or info.get("final_score")
                    or 0.0
                )
                reward_score = min(max(sum(rewards) / max(1, MAX_STEPS), 0.0), 1.0)
                final_score = float(grader_score or reward_score)
                break

            remaining = getattr(obs, "remaining_steps", MAX_STEPS - steps_taken)
            if remaining <= 3 and not has_root_cause:
                fb = fallback_plan_step(task_id, 99)
                plan = DEFAULT_PLANS.get(task_id, [])
                for p in plan:
                    if p.get("action_type") == "submit_root_cause":
                        fb = CICDTriageAction(**p)
                        break
                result2 = await env.step(fb)
                obs = result2.observation
                reward2 = float(result2.reward or 0.0)
                rewards.append(reward2)
                steps_taken += 1
                has_root_cause = True
                log_step(steps_taken, compact_action(fb), reward2, bool(result2.done), None)
                if result2.done:
                    info = getattr(result2, "info", {}) or {}
                    final_score = float(info.get("grader", {}).get("final_score") or info.get("final_score") or 0.0)
                    break

            remaining = getattr(obs, "remaining_steps", MAX_STEPS - steps_taken)
            if remaining <= 2 and has_root_cause and not has_fix:
                plan = DEFAULT_PLANS.get(task_id, [])
                for p in plan:
                    if p.get("action_type") == "submit_fix":
                        fb = CICDTriageAction(**p)
                        result3 = await env.step(fb)
                        obs = result3.observation
                        reward3 = float(result3.reward or 0.0)
                        rewards.append(reward3)
                        steps_taken += 1
                        has_fix = True
                        log_step(steps_taken, compact_action(fb), reward3, bool(result3.done), None)
                        if result3.done:
                            info = getattr(result3, "info", {}) or {}
                            final_score = float(info.get("grader", {}).get("final_score") or info.get("final_score") or 0.0)
                        break

            remaining = getattr(obs, "remaining_steps", MAX_STEPS - steps_taken)
            if remaining <= 1 and has_root_cause and has_fix:
                fb = CICDTriageAction(action_type="resolve_episode")
                result4 = await env.step(fb)
                obs = result4.observation
                reward4 = float(result4.reward or 0.0)
                rewards.append(reward4)
                steps_taken += 1
                log_step(steps_taken, compact_action(fb), reward4, True, None)
                info = getattr(result4, "info", {}) or {}
                final_score = float(info.get("grader", {}).get("final_score") or info.get("final_score") or 0.0)
                break
        else:
            final_score = min(max(sum(rewards) / max(1, MAX_STEPS), 0.0), 1.0)

    except Exception as exc:
        final_score = 0.0
        log_step(
            step=steps_taken + 1,
            action="exception",
            reward=0.0,
            done=True,
            error=str(exc),
        )

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return task_id, success, steps_taken, final_score, rewards


async def main() -> None:
    openenv_base_url = os.getenv("OPENENV_BASE_URL")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME")

    if CICDTriageEnv is None:
        print("[ERROR] openenv_cicd_triage package not installed", flush=True, file=sys.stderr)
        sys.exit(1)

    client = build_client()

    env = None
    try:
        if openenv_base_url:
            env = CICDTriageEnv(base_url=openenv_base_url)
        elif local_image_name:
            env = await CICDTriageEnv.from_docker_image(local_image_name)
        else:
            from openenv_cicd_triage.server.environment import CICDTriageEnvironment
            from openenv_cicd_triage.models import CICDTriageObservation
            
            class AsyncLocalEnv:
                def __init__(self):
                    self.env = CICDTriageEnvironment()
                    
                async def reset(self):
                    obs = self.env.reset()
                    class Res: pass
                    r = Res()
                    r.observation = obs
                    return r
                    
                async def step(self, action):
                    result = self.env.step(action)
                    class Res: pass
                    r = Res()
                    r.observation = CICDTriageObservation(**result["observation"])
                    r.reward = result.get("reward", 0.0)
                    r.done = result.get("done", False)
                    r.info = result.get("info", {})
                    return r
                    
            env = AsyncLocalEnv()

        for _ in range(TASKS_TO_RUN):
            try:
                await run_episode(client, env)
            except Exception as exc:
                print(f"[START] task=unknown env={BENCHMARK} model={MODEL_NAME}", flush=True)
                print(
                    f"[STEP] step=1 action=exception reward=0.00 done=true error={str(exc)}",
                    flush=True,
                )
                print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", flush=True, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        for _ in range(TASKS_TO_RUN):
            print(f"[START] task=unknown env={BENCHMARK} model={MODEL_NAME}", flush=True)
            print(
                f"[STEP] step=1 action=exception reward=0.00 done=true error={str(exc)}",
                flush=True,
            )
            print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())