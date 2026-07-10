from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .engine import DemoEngine
from .policy import recommend_k


DEMO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = DEMO_ROOT / "static"

app = FastAPI(title="RS-Token Adaptive Communication Demo", version="1.0.0")
engine: DemoEngine | None = None
engine_error: str | None = None


@app.on_event("startup")
def load_engine() -> None:
    global engine, engine_error
    try:
        engine = DemoEngine(device=os.environ.get("RSTOKEN_DEMO_DEVICE"))
        engine_error = None
    except Exception as exc:  # noqa: BLE001
        engine_error = str(exc)
        engine = None


@app.get("/api/health")
def health():
    if engine is None:
        return JSONResponse({"status": "loading" if engine_error is None else "error", "detail": engine_error}, status_code=503)
    return engine.health()


@app.get("/api/samples")
def samples():
    if engine is None:
        raise HTTPException(503, engine_error or "engine is loading")
    return engine.sample_manifest()


@app.get("/api/samples/{sample_id}/image")
def sample_image(sample_id: str):
    if engine is None:
        raise HTTPException(503, engine_error or "engine is loading")
    try:
        data, filename = engine.sample_image(sample_id)
    except KeyError as exc:
        raise HTTPException(404, "sample not found") from exc
    return Response(data, media_type="image/jpeg", headers={"Content-Disposition": f'inline; filename="{filename}"'})


@app.get("/api/policy")
def policy(channel: str = "awgn", snr_db: float = 5.0, protection: str = "none", priority: str = "balanced", max_transmitted_bits: int = 20480, previous_k: int | None = None):
    try:
        return recommend_k(
            channel=channel,
            snr_db=snr_db,
            protection=protection,
            priority=priority,
            max_transmitted_bits=max_transmitted_bits,
            previous_k=previous_k,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/infer")
async def infer(
    file: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    channel: str = Form(default="awgn"),
    snr_db: float = Form(default=5.0),
    protection: str = Form(default="none"),
    priority: str = Form(default="balanced"),
    max_transmitted_bits: int = Form(default=20480),
    auto_k: bool = Form(default=True),
    manual_k: int = Form(default=1),
    seed: int = Form(default=42),
    previous_k: int | None = Form(default=None),
):
    if engine is None:
        raise HTTPException(503, engine_error or "engine is loading")
    if file is not None:
        image_bytes = await file.read()
        filename = file.filename or "upload.jpg"
    elif sample_id:
        try:
            image_bytes, filename = engine.sample_image(sample_id)
        except KeyError as exc:
            raise HTTPException(404, "sample not found") from exc
    else:
        raise HTTPException(422, "upload a file or select a sample")
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(413, "image exceeds 20 MB")
    try:
        return engine.infer(
            image_bytes,
            filename=filename,
            channel=channel,
            snr_db=snr_db,
            protection=protection,
            priority=priority,
            max_transmitted_bits=max_transmitted_bits,
            auto_k=auto_k,
            manual_k=manual_k,
            seed=seed,
            previous_k=previous_k,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"inference failed: {exc}") from exc


if STATIC_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_ROOT / "assets"), name="assets")


@app.get("/{path:path}")
def frontend(path: str):
    target = STATIC_ROOT / path
    if path and target.is_file():
        return FileResponse(target)
    index = STATIC_ROOT / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"status": "frontend_not_built", "detail": "run demo/setup.ps1"}, status_code=503)
