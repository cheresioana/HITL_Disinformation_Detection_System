from flask import Blueprint, jsonify, current_app, make_response
from datetime import datetime
from threading import Lock

status_bp = Blueprint("status", __name__)

# --- shared state ---
_full_retrain = {
    "full_retrain_status": 0,   # 0 idle | 1 running | 2 done | -1 failed
    "full_retrain_error": None,
    "updated_at": None,
}
_full_retrain_lock = Lock()

def set_full_retrain(state: int, error: str | None = None):
    with _full_retrain_lock:
        _full_retrain["full_retrain_status"] = state
        _full_retrain["full_retrain_error"] = error
        _full_retrain["updated_at"] = datetime.utcnow().isoformat() + "Z"

@status_bp.get("/retrain/status")
def retrain_status():
    with _full_retrain_lock:
        payload = dict(_full_retrain)  # shallow copy for thread safety

    resp = make_response(jsonify(payload), 200)
    # prevent caching so the browser sees updates immediately
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# --- shared state ---
_update_model = {
    "update_status": 0,   # 0 idle | 1 running | 2 done | -1 failed
    "update_error": None,
    "updated_at": None,
}
_update_model_lock = Lock()

def set_update_model(state: int, error: str | None = None):
    with _update_model_lock:
        _update_model["update_status"] = state
        _update_model["update_error"] = error
        _update_model["updated_at"] = datetime.utcnow().isoformat() + "Z"

@status_bp.get("/update/status")
def update_status():
    with _update_model_lock:
        payload = dict(_update_model)  # shallow copy for thread safety

    resp = make_response(jsonify(payload), 200)
    # prevent caching so the browser sees updates immediately
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# --- hard reset shared state ---
_reset_state = {
    "reset_status": 0,   # 0 idle | 1 running | 2 done | -1 failed
    "reset_error": None,
    "updated_at": None,
}
_reset_lock = Lock()

def set_hard_reset(state: int, error: str | None = None):
    with _reset_lock:
        _reset_state["reset_status"] = state
        _reset_state["reset_error"] = error
        _reset_state["updated_at"] = datetime.utcnow().isoformat() + "Z"

@status_bp.get("/reset/status")
def reset_status():
    with _reset_lock:
        payload = dict(_reset_state)
    resp = make_response(jsonify(payload), 200)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# --- ingestion shared state ---
_ingest_state = {
    "ingest_status": 0,   # 0 idle | 1 running | 2 done | -1 failed
    "ingest_error": None,
    "updated_at": None,
}
_ingest_lock = Lock()

def set_ingest_status(state: int, error: str | None = None):
    with _ingest_lock:
        _ingest_state["ingest_status"] = state
        _ingest_state["ingest_error"] = error
        _ingest_state["updated_at"] = datetime.utcnow().isoformat() + "Z"

@status_bp.get("/ingest/status")
def ingest_status():
    with _ingest_lock:
        payload = dict(_ingest_state)
    resp = make_response(jsonify(payload), 200)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp