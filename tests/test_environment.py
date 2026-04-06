from openenv_cicd_triage.models import CICDTriageAction
from openenv_cicd_triage.server.environment import CICDTriageEnvironment


def test_reset_and_easy_path():
    env = CICDTriageEnvironment()
    obs = env.reset()
    assert obs.task_id == "dependency_mismatch_easy"
    assert obs.visible_summary["failed_stage"] == "unit-tests"

    result = env.step(CICDTriageAction(action_type="view_stage_log", stage_name="unit-tests"))
    assert result["reward"] >= 0.0
    assert "unit-tests" in result["observation"]["visible_logs"]


def test_round_robin_tasks_and_grader():
    env = CICDTriageEnvironment()
    assert env.reset().task_id == "dependency_mismatch_easy"
    assert env.reset().task_id == "missing_secret_medium"
    obs = env.reset()
    assert obs.task_id == "migration_cascade_hard"

    env.step(CICDTriageAction(action_type="view_stage_log", stage_name="integration-tests"))
    env.step(CICDTriageAction(action_type="view_stage_log", stage_name="migration"))
    final = env.step(CICDTriageAction(action_type="resolve_episode"))
    assert final["done"] is True
    assert 0.0 <= final["info"]["grader"]["final_score"] <= 1.0
