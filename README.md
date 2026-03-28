# Email Triage OpenEnv

A deterministic email triage environment built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. An agent receives a multi-email inbox and must classify, prioritize, route, and decide the response action for each email.

## Tasks

| ID | Name | Difficulty | Fields |
|---|---|---|---|
| 1 | Category Classification | easy | category |
| 2 | Category + Priority | medium | category, priority |
| 3 | Full Triage Routing | hard | category, priority, department, response_action |

## Action Schema

```json
{
  "category": "billing | account_access | technical_support | sales | spam | general | compliance | onboarding | feedback",
  "priority": "critical | high | medium | low",
  "department": "finance | support | engineering | sales | operations | compliance | hr",
  "response_action": "respond | escalate | forward | ignore | acknowledge"
}
```

## Project Structure

```
├── server/
│   ├── app.py              # FastAPI app via create_app()
│   ├── environment.py      # EmailTriageEnvironment
│   ├── grader.py           # Deterministic scoring
│   ├── reward.py           # Reward computation
│   ├── tasks.py            # Task definitions + dataset loader
│   └── Dockerfile          # Docker image for HF Spaces
├── models.py               # Action / Observation / State (extend openenv base)
├── client.py               # EnvClient subclass
├── inference.py            # LLM inference script (mandatory)
├── data/dataset.json       # 45 labeled business emails
├── openenv.yaml            # Environment metadata
├── pyproject.toml          # pip-installable
└── requirements.txt
```

## Local Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --reload --port 8000
```

## Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
python inference.py
```

Without LLM credentials, `inference.py` falls back to a heuristic baseline.

## Docker

```bash
docker build -f server/Dockerfile -t email-triage .
docker run -p 7860:7860 email-triage
```

## Episode Flow

1. `POST /reset` with `{"seed": 42, "task_id": 3}` — loads an inbox of 3–5 emails
2. `POST /step` with action for each email — grades and advances
3. When all emails are processed, `done=true` and final trajectory reward is returned

## Grading

- **Category**: exact match (1.0) or partial credit via similarity matrix
- **Priority**: exact match or proximity score
- **Department / Response Action**: exact match only
- Weighted composite per task; all scores in [0.0, 1.0]

## Team

**Team Hijibiji** — Roudraneel, Rijul, Tirthajoti