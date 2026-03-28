from __future__ import annotations


def compute_step_reward(score: float) -> float:
    return max(0.0, min(1.0, score))


def compute_trajectory_reward(
    per_email_scores: list[float], inbox_size: int, steps_taken: int
) -> float:
    if not per_email_scores:
        return 0.0
    avg = sum(per_email_scores) / len(per_email_scores)
    overshoot = max(0, steps_taken - inbox_size)
    penalty = overshoot * 0.03
    return max(0.0, min(1.0, avg - penalty))
