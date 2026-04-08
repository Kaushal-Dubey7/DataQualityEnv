import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import Action, Observation, StepResponse
from app.environment import DataQualityEnvironment
from app.tasks import TaskManager

app = FastAPI(
    title="DataQualityEnv",
    description="Real-world data quality auditing environment for RL agents.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")

_env = DataQualityEnvironment()
_task_mgr = TaskManager()
_lock = asyncio.Lock()


# ------------------------------------------------------------------ #
# Health
# ------------------------------------------------------------------ #
@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ #
# Tasks listing
# ------------------------------------------------------------------ #
@app.get("/tasks")
async def list_tasks():
    return _task_mgr.list_tasks()


# ------------------------------------------------------------------ #
# Reset
# ------------------------------------------------------------------ #
@app.post("/reset", response_model=Observation)
async def reset(body: dict = None):
    async with _lock:
        task_id = "null_hunter"
        if body and "task_id" in body:
            task_id = body["task_id"]
        try:
            observation = _env.reset(task_id=task_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return observation


# ------------------------------------------------------------------ #
# Step
# ------------------------------------------------------------------ #
@app.post("/step", response_model=StepResponse)
async def step(action: Action):
    async with _lock:
        try:
            response = _env.step(action)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return response


# ------------------------------------------------------------------ #
# State
# ------------------------------------------------------------------ #
@app.get("/state")
async def state():
    async with _lock:
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(content=jsonable_encoder(_env.state()))
