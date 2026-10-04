"""Authenticated loopback API; run one worker only. Not a public multi-tenant service."""

from contextlib import asynccontextmanager
import hmac
import json
from fastapi import Depends, FastAPI, Header, HTTPException
from .runtime import Runtime
from .runner import Runner
from .spec import TrainSpec
from .store import Store


def create_app(runtime=None, start_runner=True):
    runtime = runtime or Runtime.load()
    store = Store(runtime.root / "state")
    runner = Runner(runtime, store)

    @asynccontextmanager
    async def lifespan(app):
        if start_runner:
            runner.start()
        yield
        if start_runner:
            runner.close()

    app = FastAPI(title="Nibble", version="0.1.0", lifespan=lifespan)

    def auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization, f"Bearer {runtime.token}"):
            raise HTTPException(401, "invalid bearer token")

    def lookup(job_id):
        try:
            return store.get(job_id)
        except KeyError:
            raise HTTPException(404, "job not found")

    def validated(body):
        try:
            spec = TrainSpec.from_dict(body)
            plan = runtime.plan(spec)
            return spec, plan
        except ValueError as exc:
            raise HTTPException(422, str(exc))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/capabilities", dependencies=[Depends(auth)])
    def capabilities():
        return {
            "models": list(runtime.models),
            "datasets": list(runtime.datasets),
            "environments": list(runtime.environments),
            "rewards": list(runtime.rewards),
            "algorithms": ["grpo"],
            "score_centering": False,
            "resume": False,
            "managed_gpu_provisioning": False,
        }

    @app.post("/v1/plan", dependencies=[Depends(auth)])
    def plan(body: dict):
        return validated(body)[1]

    @app.post("/v1/jobs", status_code=202, dependencies=[Depends(auth)])
    def train(body: dict):
        spec, _ = validated(body)
        return store.create(spec.to_dict())

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(auth)])
    def status(job_id: str):
        return lookup(job_id)

    @app.post("/v1/jobs/{job_id}/cancel", dependencies=[Depends(auth)])
    def cancel(job_id: str):
        lookup(job_id)
        return store.cancel(job_id)

    @app.get("/v1/jobs/{job_id}/events", dependencies=[Depends(auth)])
    def events(job_id: str):
        lookup(job_id)
        path = store.root / "runs" / job_id / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # writer may be appending its last line
        return result

    @app.get("/v1/jobs/{job_id}/report", dependencies=[Depends(auth)])
    def report(job_id: str):
        lookup(job_id)
        path = store.root / "runs" / job_id / "report.json"
        if not path.exists():
            raise HTTPException(409, "report not ready")
        return json.loads(path.read_text())

    return app
