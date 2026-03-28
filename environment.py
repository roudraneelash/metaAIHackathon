from __future__ import annotations

import random
import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

from models import (
    EmailRecord,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
)
from server.grader import grade_action
from server.reward import compute_step_reward, compute_trajectory_reward
from server.tasks import TASKS, get_task_definition, load_dataset


INBOX_SIZE_RANGE = (3, 5)


class EmailTriageEnvironment(
    Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    def __init__(self) -> None:
        super().__init__()
        self._dataset = load_dataset()
        self._dataset_by_id = {r.email_id: r for r in self._dataset}
        self._rng = random.Random()
        self._inbox: list[EmailRecord] = []
        self._state = EmailTriageState()

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        task_id: int = kwargs.get("task_id", 1)
        task = get_task_definition(task_id)

        if seed is not None:
            self._rng.seed(seed)

        inbox_size = self._rng.randint(*INBOX_SIZE_RANGE)
        self._inbox = self._rng.sample(self._dataset, min(inbox_size, len(self._dataset)))

        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_task_id=task_id,
            seed=seed,
            inbox_email_ids=[e.email_id for e in self._inbox],
            current_email_index=0,
            per_email_scores=[],
            total_reward=0.0,
        )

        return self._build_observation(task)

    def step(
        self,
        action: EmailTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        if not self._inbox or self._state.current_task_id is None:
            raise RuntimeError("Environment has not been reset.")

        idx = self._state.current_email_index
        if idx >= len(self._inbox):
            raise RuntimeError("Episode already done — call reset().")

        current_email = self._inbox[idx]
        task_id = self._state.current_task_id
        task = get_task_definition(task_id)

        score, breakdown = grade_action(action, current_email, task_id)
        step_reward = compute_step_reward(score)

        self._state.per_email_scores.append(score)
        self._state.step_count += 1
        self._state.current_email_index += 1

        is_done = self._state.current_email_index >= len(self._inbox)

        if is_done:
            traj_reward = compute_trajectory_reward(
                self._state.per_email_scores,
                len(self._inbox),
                self._state.step_count,
            )
            self._state.total_reward = traj_reward
            final_reward = traj_reward
        else:
            final_reward = step_reward

        history_entry = {
            "email_id": current_email.email_id,
            "score": score,
            "breakdown": breakdown,
        }

        return self._build_observation(
            task,
            done=is_done,
            reward=final_reward,
            extra_history=history_entry,
        )

    @property
    def state(self) -> EmailTriageState:
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        task: dict,
        done: bool = False,
        reward: float | None = None,
        extra_history: dict | None = None,
    ) -> EmailTriageObservation:
        idx = self._state.current_email_index
        inbox_size = len(self._inbox)

        if idx < inbox_size:
            email = self._inbox[idx]
            email_view = {
                "email_id": email.email_id,
                "subject": email.subject,
                "sender": email.sender,
                "body": email.body,
            }
        else:
            email_view = None

        history: list[dict] = []
        for i, s in enumerate(self._state.per_email_scores):
            history.append({"step": i + 1, "score": s})
        if extra_history:
            history.append(extra_history)

        return EmailTriageObservation(
            done=done,
            reward=reward,
            metadata={},
            task_id=task["id"],
            task_name=task["name"],
            instructions=task["instructions"],
            allowed_fields=list(task["allowed_fields"]),
            current_email=email_view,
            inbox_size=inbox_size,
            emails_remaining=max(0, inbox_size - idx),
            emails_processed=idx,
            history=history,
        )
