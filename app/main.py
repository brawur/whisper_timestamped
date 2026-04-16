from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="Whisper Timestamped Service")
app.include_router(router)
