from __future__ import annotations

import re
import string
from dataclasses import asdict
from typing import Any

from .config import load_grading, normalized_weights
from .models import CICDTriageState


def _normalize_text(value: str, *, strip_punctuation: bool = True) -> str:
    value = value.lower().strip()
    if strip_punctuation:
        value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value


def _keyword_fraction(text: str, keywords: list[str], min_fraction: float) -> float:
    if not keywords:
        return 1.0
    norm = _normalize_text(text)
    hits = 0
    for kw in keywords:
        if _normalize_text(kw) in norm:
            hits += 1
    fraction = hits / len(keywords)
    return fraction if fraction >= min_fraction else fraction


class EpisodeGrader:
    def __init__(self, task: dict[str, Any]):
        self.task = task
        self.cfg = load_grading()
        self.weights = normalized_weights(self.cfg["score_weights"])
        self.min_keyword_fraction = float(self.cfg["text_matching"]["min_keyword_fraction"])
        self.symptom_partial_credit = float(self.cfg["text_matching"]["symptom_alias_partial_credit"])

    def grade(self, state: CICDTriageState) -> dict[str, Any]:
        truth = self.task["truth"]
        evidence_score = self._evidence_score(state, truth)
        root_cause_score = self._root_cause_score(state, truth)
        fix_score = self._fix_score(state, truth)
        efficiency_score = self._efficiency_score(state)

        final_score = (
            self.weights["evidence"] * evidence_score
            + self.weights["root_cause"] * root_cause_score
            + self.weights["fix"] * fix_score
            + self.weights["efficiency"] * efficiency_score
        )
        final_score = max(0.0, min(1.0, final_score))

        return {
            "evidence_score": round(evidence_score, 4),
            "root_cause_score": round(root_cause_score, 4),
            "fix_score": round(fix_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "final_score": round(final_score, 4),
            "weights": self.weights,
            "state": {
                "steps_taken": state.step_count,
                "root_cause_submission": (
                    asdict(state.root_cause_submission)
                    if state.root_cause_submission is not None
                    else None
                ),
                "fix_submission": (
                    asdict(state.fix_submission)
                    if state.fix_submission is not None
                    else None
                ),
            },
        }

    def _evidence_score(self, state: CICDTriageState, truth: dict[str, Any]) -> float:
        required = truth["evidence"].get("required_actions", [])
        supporting = truth["evidence"].get("supporting_actions", [])
        seen = {entry["action_key"] for entry in state.action_history}

        req_score = (sum(1 for item in required if item in seen) / len(required)) if required else 1.0
        sup_score = (sum(1 for item in supporting if item in seen) / len(supporting)) if supporting else 1.0

        required_weight = float(self.cfg["score_weights"].get("evidence_required", 2.0))
        supporting_weight = float(self.cfg["score_weights"].get("evidence_supporting", 1.0))
        total_weight = required_weight + supporting_weight
        if total_weight <= 0:
            return 0.0

        return max(
            0.0,
            min(1.0, (required_weight * req_score + supporting_weight * sup_score) / total_weight),
        )

    def _root_cause_score(self, state: CICDTriageState, truth: dict[str, Any]) -> float:
        submission = state.root_cause_submission
        if submission is None:
            return 0.0

        if not submission.category and not submission.summary:
            return 0.0

        cfg = self.cfg.get("grading", {}).get("root_cause", {})
        category_match_credit = float(cfg.get("category_match_credit", 0.7))
        detail_weight = float(cfg.get("detail_weight", 0.3))

        score = 0.0
        rc_truth = truth.get("root_cause", {})

        expected_category = (
            rc_truth.get("category")
            or rc_truth.get("root_cause_category")
            or rc_truth.get("type")
            or ""
        )

        if submission.category == expected_category and expected_category:
            score += category_match_credit
        else:
            symptom_aliases = rc_truth.get("symptom_aliases", {})
            if submission.category in symptom_aliases:
                score += float(symptom_aliases[submission.category])
            else:
                summary_norm = _normalize_text(submission.summary or "")
                for alias, partial_credit in symptom_aliases.items():
                    if _normalize_text(alias) in summary_norm:
                        score += float(partial_credit)
                        break

        detail_keywords = rc_truth.get("detail_keywords", [])
        score += detail_weight * _keyword_fraction(
            submission.summary or "",
            detail_keywords,
            self.min_keyword_fraction,
        )

        return min(1.0, max(0.0, score))

    def _fix_score(self, state: CICDTriageState, truth: dict[str, Any]) -> float:
        submission = state.fix_submission
        if submission is None:
            return 0.0

        if not submission.fix_type and not submission.summary:
            return 0.0

        cfg = self.cfg.get("grading", {}).get("fix", {})
        type_match_credit = float(cfg.get("type_match_credit", 0.7))
        detail_weight = float(cfg.get("detail_weight", 0.3))

        score = 0.0
        fix_truth = truth.get("fix", {})

        expected_fix_type = (
            fix_truth.get("fix_type")
            or fix_truth.get("type")
            or fix_truth.get("category")
            or ""
        )

        if submission.fix_type == expected_fix_type and expected_fix_type:
            score += type_match_credit
        else:
            fix_aliases = fix_truth.get("fix_aliases", {})
            if submission.fix_type in fix_aliases:
                score += float(fix_aliases[submission.fix_type])
            else:
                summary_norm = _normalize_text(submission.summary or "")
                for alias, partial_credit in fix_aliases.items():
                    if _normalize_text(alias) in summary_norm:
                        score += float(partial_credit)
                        break

        detail_keywords = fix_truth.get("detail_keywords", [])
        score += detail_weight * _keyword_fraction(
            submission.summary or "",
            detail_keywords,
            self.min_keyword_fraction,
        )

        return min(1.0, max(0.0, score))

    def _efficiency_score(self, state: CICDTriageState) -> float:
        ideal_steps = max(1, int(self.task.get("ideal_steps", state.max_steps or 1)))
        max_steps = max(ideal_steps, int(self.task.get("max_steps", ideal_steps)))
        repeated = sum(1 for entry in state.action_history if entry.get("repeated", False))
        step_overrun = max(0, state.step_count - ideal_steps)

        penalty_cfg = self.cfg["penalties"]
        overrun_factor = float(penalty_cfg["step_budget_overrun_factor"])
        repeated_factor = float(penalty_cfg["repeated_action"])

        normalized_overrun = step_overrun / max(1, max_steps - ideal_steps)
        penalty = min(1.0, normalized_overrun * overrun_factor + repeated * repeated_factor)
        return max(0.0, 1.0 - penalty)