from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient, StepResult

from models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageEnvClient(
    EnvClient[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    def _step_payload(self, action: EmailTriageAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[EmailTriageObservation]:
        obs_data = payload.get("observation", payload)
        obs = EmailTriageObservation.model_validate(obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> EmailTriageState:
        return EmailTriageState.model_validate(payload)
