"""Small synchronous HTTP client for asynchronously executing training jobs."""

import json
import os
import time
import urllib.error
import urllib.request
from .spec import TrainSpec


class Client:
    def __init__(self, base_url="http://127.0.0.1:8765", token=None):
        self.base_url = base_url.rstrip("/")
        self.token = token or os.environ.get("NIBBLE_TOKEN")
        if not self.token:
            raise ValueError("pass token or set NIBBLE_TOKEN")

    def _request(self, method, path, payload=None):
        body = (
            None if payload is None else json.dumps(payload, allow_nan=False).encode()
        )
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Nibble HTTP {exc.code}: {exc.read().decode()}"
            ) from exc

    def plan(self, **kwargs):
        return self._request("POST", "/v1/plan", TrainSpec(**kwargs).to_dict())

    def train(self, **kwargs):
        data = self._request("POST", "/v1/jobs", TrainSpec(**kwargs).to_dict())
        return Job(self, data["id"])

    def job(self, job_id):
        return Job(self, job_id)


class Job:
    def __init__(self, client, job_id):
        self.client, self.id = client, job_id

    def status(self):
        return self.client._request("GET", f"/v1/jobs/{self.id}")

    def events(self):
        return self.client._request("GET", f"/v1/jobs/{self.id}/events")

    def cancel(self):
        return self.client._request("POST", f"/v1/jobs/{self.id}/cancel", {})

    def report(self):
        return self.client._request("GET", f"/v1/jobs/{self.id}/report")

    def wait(self, timeout=300, poll_seconds=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.status()
            if state["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                return state
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Job {self.id} is still running; wait timeout does not cancel it"
        )
