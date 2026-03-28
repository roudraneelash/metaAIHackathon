import sys
from pathlib import Path

# Ensure repo root is on sys.path so `models` and `server` are importable
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from openenv.core.env_server import create_app

from models import EmailTriageAction, EmailTriageObservation
from server.environment import EmailTriageEnvironment
from server.tasks import TASKS

app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email_triage",
)


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": t["id"],
                "name": t["name"],
                "difficulty": t["difficulty"],
                "instructions": t["instructions"],
                "allowed_fields": t["allowed_fields"],
            }
            for t in TASKS.values()
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)
