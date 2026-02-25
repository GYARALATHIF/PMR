from collections import deque
from datetime import datetime, UTC
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import APP_ENV
from inference import RiskPredictor
from schemas import PredictRequest, RiskPrediction

app = FastAPI(title="Patient Risk Predictor API", version="2.0.0")

# Your files are at repo root (index.html, styles.css, app.js), so serve root as static/template dir
app.mount("/static", StaticFiles(directory="."), name="static")
templates = Jinja2Templates(directory=".")

predictor = None
prediction_history = deque(maxlen=100)


def get_predictor():
    global predictor
    if predictor is None:
        predictor = RiskPredictor()
    return predictor


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": APP_ENV,
        "model_loaded": predictor is not None,
        "history_size": len(prediction_history),
    }


@app.post("/api/predict", response_model=RiskPrediction)
async def predict(payload: PredictRequest):
    try:
        p = get_predictor()
        result, meta = p.predict(payload.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    prediction_history.appendleft(
        {
            "created_at": meta.created_at,
            "mortality_probability": result["mortality_probability"],
            "readmission_probability": result["readmission_probability"],
            "mortality_risk_tier": result["mortality_risk_tier"],
            "readmission_risk_tier": result["readmission_risk_tier"],
        }
    )
    return result


@app.get("/api/history")
async def history(limit: int = 20):
    return {"items": list(prediction_history)[: max(1, min(limit, 100))]}


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(
                {
                    "type": "heartbeat",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "history_size": len(prediction_history),
                }
            )
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        return
