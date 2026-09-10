"""Shared protocol v1; copied into each image's app/maintenance.py (separate builds).
Never replace activity.lock: host and containers must keep the same inode.
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
from pathlib import Path


class MaintenanceMiddleware:
    def __init__(self, app, api_token=None):
        self.app = app
        self.api_token = api_token
        self.active: dict[str, int] = {}

    def runtime(self):
        directory = os.environ.get("GATEWAY_MAINTENANCE_DIR", "")
        result = {"protocol": 0, "lockId": "", "blocked": False,
                  "activeRequests": sum(self.active.values()), "requests": dict(self.active)}
        if not directory:
            return result
        try:
            with open(Path(directory) / "activity.lock", "rb") as lock:
                stat = os.fstat(lock.fileno())
                result.update(protocol=1, lockId=f"{stat.st_dev}:{stat.st_ino}",
                              blocked=(Path(directory) / "maintenance.json").exists())
        except OSError:
            pass
        return result

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        kind = scope.get("type")
        if kind == "http" and scope.get("method") == "GET" and path == "/admin/runtime":
            # Gateway uses its existing API token; workers are loopback-only services.
            expected = self.api_token if self.api_token is not None else os.environ.get("API_TOKEN", "")
            headers = dict(scope.get("headers", []))
            if expected and headers.get(b"authorization") != ("Bearer " + expected).encode():
                await self.reply(send, 401, {"detail": "Unauthorized"})
                return
            await self.reply(send, 200, self.runtime())
            return
        work = (kind == "http" and scope.get("method") not in {"GET", "HEAD", "OPTIONS"}) or (
            kind == "websocket" and not path.startswith(("/ws/jobs/", "/ws/uploads/")))
        if not work:
            await self.app(scope, receive, send)
            return
        lock = None
        directory = os.environ.get("GATEWAY_MAINTENANCE_DIR", "")
        try:
            if directory:
                try:
                    lock = open(Path(directory) / "activity.lock", "rb")
                    while True:
                        try:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            # A short preflight must not fail a running job's next step.
                            await asyncio.sleep(0.05)
                    if (Path(directory) / "maintenance.json").exists():
                        await self.reject(kind, send)
                        return
                except OSError:
                    await self.reject(kind, send)
                    return
            self.active[path] = self.active.get(path, 0) + 1
            try:
                await self.app(scope, receive, send)
            finally:
                self.active[path] -= 1
                if self.active[path] == 0:
                    del self.active[path]
        finally:
            if lock is not None:
                lock.close()

    @staticmethod
    async def reply(send, status, body):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    async def reject(self, kind, send):
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1013, "reason": "Gateway maintenance"})
        else:
            await self.reply(send, 503, {"detail": "Gateway maintenance", "code": "maintenance"})
