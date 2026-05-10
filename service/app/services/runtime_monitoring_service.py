from __future__ import annotations

import json
import logging
import os
from collections import deque
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(ts: datetime) -> str:
    return ts.isoformat()


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    return str(value)


def _logger() -> logging.Logger:
    logger = logging.getLogger("pickmeal.runtime")
    if not logger.handlers:
        level_name = os.getenv("PICKMEAL_LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=getattr(logging, level_name, logging.INFO), format="%(message)s")
    return logger


def _log_event(level: int, service_id: str, event: str, **payload: Any) -> None:
    body = {
        "timestamp": _to_iso(_utc_now()),
        "service_id": service_id,
        "event": event,
        **{key: _safe_json(value) for key, value in payload.items()},
    }
    _logger().log(level, json.dumps(body, ensure_ascii=False))


_STATE_LOCK = Lock()
_STATE: dict[str, dict[str, Any]] = {}


def _ensure_service(service_id: str) -> dict[str, Any]:
    with _STATE_LOCK:
        state = _STATE.get(service_id)
        if state is None:
            state = {
                "service_id": service_id,
                "started_at": _utc_now(),
                "request_count": 0,
                "status_counts": {},
                "paths": {},
                "stages": {},
                "recent_errors": deque(maxlen=20),
            }
            _STATE[service_id] = state
            _log_event(logging.INFO, service_id, "service_started")
        return state


def attach_runtime_monitoring(app: FastAPI, service_id: str | None = None) -> str:
    runtime_service_id = service_id or app.title
    _ensure_service(runtime_service_id)

    @app.middleware("http")
    async def runtime_monitoring_middleware(request: Request, call_next):
        start = perf_counter()
        status_code = 500
        exc_message: str | None = None
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
            return response
        except Exception as exc:
            exc_message = str(exc)
            raise
        finally:
            duration_ms = round((perf_counter() - start) * 1000.0, 2)
            record_request(
                runtime_service_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
                error=exc_message,
            )

    return runtime_service_id


def record_request(
    service_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    error: str | None = None,
) -> None:
    state = _ensure_service(service_id)
    key = f"{method.upper()} {path}"
    now = _utc_now()
    with _STATE_LOCK:
        path_state = state["paths"].setdefault(
            key,
            {
                "method": method.upper(),
                "path": path,
                "count": 0,
                "error_count": 0,
                "status_counts": {},
                "last_status": None,
                "last_duration_ms": None,
                "max_duration_ms": 0.0,
                "total_duration_ms": 0.0,
                "last_seen_at": None,
            },
        )
        path_state["count"] += 1
        path_state["status_counts"][str(status_code)] = int(path_state["status_counts"].get(str(status_code), 0)) + 1
        path_state["last_status"] = int(status_code)
        path_state["last_duration_ms"] = float(duration_ms)
        path_state["max_duration_ms"] = max(float(path_state["max_duration_ms"]), float(duration_ms))
        path_state["total_duration_ms"] += float(duration_ms)
        path_state["last_seen_at"] = _to_iso(now)
        if error:
            path_state["error_count"] += 1
            _append_error(state, component="request", operation=key, error=error, status_code=status_code)

        state["request_count"] += 1
        state["status_counts"][str(status_code)] = int(state["status_counts"].get(str(status_code), 0)) + 1
    log_level = logging.INFO if status_code < 400 else logging.WARNING
    _log_event(
        log_level,
        service_id,
        "http_request",
        method=method.upper(),
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
        error=error,
    )


def _append_error(state: dict[str, Any], component: str, operation: str, error: str, status_code: int | None = None) -> None:
    state["recent_errors"].appendleft(
        {
            "timestamp": _to_iso(_utc_now()),
            "component": component,
            "operation": operation,
            "status_code": status_code,
            "error": error,
        }
    )


def record_stage(
    service_id: str,
    component: str,
    operation: str,
    duration_ms: float,
    ok: bool = True,
    details: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    state = _ensure_service(service_id)
    key = f"{component}:{operation}"
    with _STATE_LOCK:
        stage_state = state["stages"].setdefault(
            key,
            {
                "component": component,
                "operation": operation,
                "count": 0,
                "ok_count": 0,
                "error_count": 0,
                "last_duration_ms": None,
                "max_duration_ms": 0.0,
                "total_duration_ms": 0.0,
                "last_details": None,
                "last_error": None,
                "last_seen_at": None,
            },
        )
        stage_state["count"] += 1
        if ok:
            stage_state["ok_count"] += 1
        else:
            stage_state["error_count"] += 1
        stage_state["last_duration_ms"] = float(duration_ms)
        stage_state["max_duration_ms"] = max(float(stage_state["max_duration_ms"]), float(duration_ms))
        stage_state["total_duration_ms"] += float(duration_ms)
        stage_state["last_seen_at"] = _to_iso(_utc_now())
        stage_state["last_details"] = _safe_json(details) if details else None
        stage_state["last_error"] = error
        if error:
            _append_error(state, component=component, operation=operation, error=error)
    _log_event(
        logging.INFO if ok else logging.WARNING,
        service_id,
        "stage",
        component=component,
        operation=operation,
        ok=ok,
        duration_ms=duration_ms,
        details=details,
        error=error,
    )


def measure_stage(
    service_id: str,
    component: str,
    operation: str,
    details: dict[str, Any] | None = None,
):
    start = perf_counter()

    class _StageTimer:
        def finish(self, ok: bool = True, error: str | None = None, extra_details: dict[str, Any] | None = None) -> float:
            merged = dict(details or {})
            if extra_details:
                merged.update(extra_details)
            duration_ms = round((perf_counter() - start) * 1000.0, 2)
            record_stage(
                service_id,
                component=component,
                operation=operation,
                duration_ms=duration_ms,
                ok=ok,
                details=merged or None,
                error=error,
            )
            return duration_ms

    return _StageTimer()


def load_runtime_stats(service_id: str) -> dict[str, Any]:
    state = _ensure_service(service_id)
    with _STATE_LOCK:
        started_at = state["started_at"]
        request_count = int(state["request_count"])
        status_counts = {str(k): int(v) for k, v in state["status_counts"].items()}
        recent_errors = list(state["recent_errors"])
        path_items = list(state["paths"].values())
        stage_items = list(state["stages"].values())
    uptime_seconds = round((_utc_now() - started_at).total_seconds(), 1)
    path_rows = []
    for row in path_items:
        count = int(row["count"])
        avg_ms = round(float(row["total_duration_ms"]) / count, 2) if count else 0.0
        path_rows.append(
            {
                "method": row["method"],
                "path": row["path"],
                "count": count,
                "error_count": int(row["error_count"]),
                "last_status": row["last_status"],
                "avg_duration_ms": avg_ms,
                "max_duration_ms": round(float(row["max_duration_ms"]), 2),
                "last_duration_ms": row["last_duration_ms"],
                "status_counts": row["status_counts"],
                "last_seen_at": row["last_seen_at"],
            }
        )
    path_rows.sort(key=lambda row: (-int(row["count"]), row["path"]))

    stage_rows = []
    for row in stage_items:
        count = int(row["count"])
        avg_ms = round(float(row["total_duration_ms"]) / count, 2) if count else 0.0
        stage_rows.append(
            {
                "component": row["component"],
                "operation": row["operation"],
                "count": count,
                "ok_count": int(row["ok_count"]),
                "error_count": int(row["error_count"]),
                "avg_duration_ms": avg_ms,
                "max_duration_ms": round(float(row["max_duration_ms"]), 2),
                "last_duration_ms": row["last_duration_ms"],
                "last_details": row["last_details"],
                "last_error": row["last_error"],
                "last_seen_at": row["last_seen_at"],
            }
        )
    stage_rows.sort(key=lambda row: (row["component"], row["operation"]))

    return {
        "service_id": service_id,
        "started_at": _to_iso(started_at),
        "uptime_seconds": uptime_seconds,
        "request_count": request_count,
        "status_counts": status_counts,
        "path_rows": path_rows,
        "stage_rows": stage_rows,
        "recent_errors": recent_errors,
    }
