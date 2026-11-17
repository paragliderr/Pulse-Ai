"""Loads the exported model once; falls back to the stub when torch or the checkpoint is missing."""
import os
from pathlib import Path

CKPT = Path(os.getenv("PULSE_MODEL_PATH", Path(__file__).resolve().parent.parent / "model_store" / "pulse_model.pt"))
_state = {"svc": None, "tried": False, "reason": "not loaded yet"}


def get_service():
    if not _state["tried"]:
        _state["tried"] = True
        try:
            if not CKPT.exists():
                raise FileNotFoundError(f"{CKPT.name} not found - run experiments/export_model.py")
            from app.serving.pulse_service import PulseService   # imports torch
            _state["svc"] = PulseService(str(CKPT))
            _state["reason"] = "ok"
        except Exception as e:                                    # torch missing, no checkpoint, bad file...
            _state["reason"] = f"{type(e).__name__}: {e}"
            print(f"[pulse] model not loaded, serving stub scores. Reason: {_state['reason']}")
    return _state["svc"]


def status():
    svc = get_service()
    return {"mode": "model" if svc else "stub", "reason": _state["reason"],
            "meta": svc.meta if svc else None}
