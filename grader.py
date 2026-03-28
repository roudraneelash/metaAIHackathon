from __future__ import annotations

from models import EmailRecord, EmailTriageAction


CATEGORY_SIMILARITY = {
    ("billing", "payments"): 0.7,
    ("payments", "billing"): 0.7,
    ("technical_support", "account_access"): 0.5,
    ("account_access", "technical_support"): 0.5,
    ("spam", "sales"): 0.1,
    ("sales", "spam"): 0.1,
    ("billing", "finance"): 0.4,
    ("finance", "billing"): 0.4,
    ("general", "feedback"): 0.3,
    ("feedback", "general"): 0.3,
    ("onboarding", "account_access"): 0.4,
    ("account_access", "onboarding"): 0.4,
    ("compliance", "billing"): 0.2,
    ("billing", "compliance"): 0.2,
}

PRIORITY_SCORES = {
    ("critical", "high"): 0.6,
    ("high", "critical"): 0.6,
    ("high", "medium"): 0.5,
    ("medium", "high"): 0.5,
    ("medium", "low"): 0.4,
    ("low", "medium"): 0.4,
    ("critical", "medium"): 0.3,
    ("medium", "critical"): 0.3,
    ("critical", "low"): 0.1,
    ("low", "critical"): 0.1,
    ("high", "low"): 0.2,
    ("low", "high"): 0.2,
}


def _normalized(value: str | None) -> str:
    return (value or "").strip().lower()


def _score_exact_or_similar(predicted: str | None, expected: str) -> float:
    pred = _normalized(predicted)
    exp = _normalized(expected)
    if not pred:
        return 0.0
    if pred == exp:
        return 1.0
    return CATEGORY_SIMILARITY.get((pred, exp), 0.0)


def _score_priority(predicted: str | None, expected: str) -> float:
    pred = _normalized(predicted)
    exp = _normalized(expected)
    if not pred:
        return 0.0
    if pred == exp:
        return 1.0
    return PRIORITY_SCORES.get((pred, exp), 0.0)


def _score_exact(predicted: str | None, expected: str) -> float:
    return 1.0 if _normalized(predicted) == _normalized(expected) and predicted else 0.0


def grade_action(
    action: EmailTriageAction, email: EmailRecord, task_id: int
) -> tuple[float, dict[str, float]]:
    cat = _score_exact_or_similar(action.category, email.category)
    pri = _score_priority(action.priority, email.priority)
    dep = _score_exact(action.department, email.department)
    resp = _score_exact(action.response_action, email.response_action)

    if task_id == 1:
        return cat, {"category": cat}

    if task_id == 2:
        score = 0.6 * cat + 0.4 * pri
        return score, {"category": cat, "priority": pri}

    if task_id == 3:
        score = 0.35 * cat + 0.20 * pri + 0.25 * dep + 0.20 * resp
        return score, {
            "category": cat,
            "priority": pri,
            "department": dep,
            "response_action": resp,
        }

    raise ValueError(f"Unsupported task_id: {task_id}")
