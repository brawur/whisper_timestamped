from fastapi import FastAPI

from app.services.transcription import GpuContextMiddleware
from app.maintenance import MaintenanceMiddleware

from app.api.routes import router

app = FastAPI(title="Whisper Timestamped Service")
app.include_router(router)

app.add_middleware(MaintenanceMiddleware)
app.add_middleware(GpuContextMiddleware)
